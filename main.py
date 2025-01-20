import argparse
import asyncio
import logging
import os
import time
from contextlib import contextmanager

import numpy as np
import wgpu
from colorama import Fore, Style, init
from PIL import Image
from wgpu import GPUDevice, GPUTexture


@contextmanager
def timer(name: str):
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    print(f"{name}: {(end - start) * 1000:.2f}ms")


init()


class ColoredFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": Fore.CYAN,
        "INFO": Fore.GREEN,
        "WARNING": Fore.YELLOW,
        "ERROR": Fore.RED,
        "CRITICAL": Fore.RED + Style.BRIGHT,
    }

    def format(self, record):
        color = self.COLORS.get(record.levelname, "")
        record.levelname = f"{color}{record.levelname}{Style.RESET_ALL}"
        return f"{record.levelname}: {record.getMessage()}"


logger = logging.getLogger("sift")


def setup_logger(debug: bool = False):
    if debug:
        logger.setLevel(logging.DEBUG)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(ColoredFormatter())
            logger.addHandler(handler)
        logger.debug("Debug logging enabled")


def save_mipmaps(
    device: GPUDevice,
    texture: GPUTexture,
    width: int,
    height: int,
    mip_level_count: int,
    image_index: int,
    output_dir: str,
) -> None:
    logger.debug(f"Saving mipmaps for image {image_index} with {mip_level_count} levels")
    base_dir = os.path.join(output_dir, f"image_{image_index}")
    os.makedirs(base_dir, exist_ok=True)

    current_width, current_height = width, height

    for mip_level in range(mip_level_count):
        try:
            data = device.queue.read_texture(
                {
                    "texture": texture,
                    "mip_level": mip_level,
                    "origin": (0, 0, 0),
                },
                {
                    "bytes_per_row": current_width,
                    "rows_per_image": current_height,
                },
                (current_width, current_height, 1),
            )

            data_array = np.frombuffer(data, dtype=np.uint8).reshape(current_height, current_width)

            output_path = os.path.join(base_dir, f"mip_{mip_level}.png")
            Image.fromarray(data_array).save(output_path)
        except Exception as e:
            logger.error(f"Failed to save mip level {mip_level}: {e}")
            continue

        current_width = max(1, current_width // 2)
        current_height = max(1, current_height // 2)


async def process_batch(
    device: GPUDevice,
    images: list[Image.Image],
    batch_size: int = 10,
    mip_levels: int = 4,
    save_dir: str | None = None,
) -> None:
    assert all(image.size == images[0].size for image in images), "All images must be the same size"
    width, height = images[0].size
    aligned_width = (width + 255) & ~255

    try:
        shader_code = open("mips.wgsl").read()
    except IOError as e:
        raise RuntimeError(f"Failed to load mipmap shader: {e}")

    shader_module = device.create_shader_module(code=shader_code)
    pipeline = device.create_render_pipeline(
        layout="auto",
        vertex={"module": shader_module, "entry_point": "vs"},
        fragment={
            "module": shader_module,
            "entry_point": "fs",
            "targets": [{"format": "r8unorm"}],
        },
        primitive={"topology": wgpu.PrimitiveTopology.triangle_strip},
    )
    sampler = device.create_sampler(
        min_filter="linear",
        mag_filter="linear",
        mipmap_filter="linear",
    )

    textures = []
    for idx in range(batch_size):
        texture = device.create_texture(
            size=(width, height, 1),
            format="r8unorm",
            usage=wgpu.TextureUsage.TEXTURE_BINDING
            | wgpu.TextureUsage.COPY_DST
            | wgpu.TextureUsage.RENDER_ATTACHMENT
            | wgpu.TextureUsage.COPY_SRC,
            mip_level_count=mip_levels,
        )
        views = []
        for level in range(mip_levels):
            views.append(
                texture.create_view(base_mip_level=level, mip_level_count=1, dimension="2d")
            )
        textures.append({"texture": texture, "views": views})

    for batch_start in range(0, len(images), batch_size):
        batch_end = min(batch_start + batch_size, len(images))
        batch = images[batch_start:batch_end]
        logger.debug(f"Processing batch {batch_start // batch_size + 1} with {len(batch)} images")

        encoder = device.create_command_encoder(label="Batch Mipmap Generation")

        for idx, image in enumerate(batch):
            padded_image = np.pad(image, ((0, 0), (0, aligned_width - width)), mode="constant")

            device.queue.write_texture(
                {"texture": textures[idx]["texture"], "mip_level": 0, "origin": (0, 0, 0)},
                padded_image,
                {"bytes_per_row": aligned_width, "rows_per_image": height},
                (width, height, 1),
            )

            for level in range(mip_levels - 1):
                src_view = textures[idx]["views"][level]
                dst_view = textures[idx]["views"][level + 1]

                bind_group = device.create_bind_group(
                    layout=pipeline.get_bind_group_layout(0),
                    entries=[
                        {"binding": 0, "resource": sampler},
                        {"binding": 1, "resource": src_view},
                    ],
                )

                render_pass_encoder = encoder.begin_render_pass(
                    color_attachments=[
                        {
                            "view": dst_view,
                            "clear_value": (0.0, 0.0, 0.0, 1.0),
                            "load_op": "load",
                            "store_op": "store",
                        }
                    ],
                )
                render_pass_encoder.set_pipeline(pipeline)
                render_pass_encoder.set_bind_group(0, bind_group)
                render_pass_encoder.draw(4, 1, 0, 0)
                render_pass_encoder.end()

        device.queue.submit([encoder.finish()])

        if save_dir is not None:
            mipmap_save_dir = os.path.join(save_dir, "mipmaps")
            os.makedirs(mipmap_save_dir, exist_ok=True)
            for idx in range(len(batch)):
                save_mipmaps(
                    device,
                    textures[idx]["texture"],
                    width,
                    height,
                    mip_levels,
                    batch_start + idx,
                    mipmap_save_dir,
                )


async def main():
    parser = argparse.ArgumentParser(description="Mipmap Generator")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--save", type=str, help="Dir to save the whole pipeline")

    args = parser.parse_args()

    setup_logger(args.debug)

    gpu = wgpu.GPU()
    adapter = await gpu.request_adapter_async(
        power_preference="high-performance",
        force_fallback_adapter=False,
    )

    if adapter is None:
        logger.error("No suitable GPU adapter found!")
        return

    logger.debug(f"Using adapter: {adapter.info}")
    device = await adapter.request_device_async()
    logger.debug("Device created successfully")

    images = []
    for i in range(82):
        try:
            img = Image.open(f"data/{i}.jpg").convert("L")
            images.append(img)
        except IOError as e:
            print(f"Failed to load image {i}: {e}")
            continue
    with timer("Processing batch"):
        await process_batch(
            device,
            images,
            save_dir=args.save,
        )


if __name__ == "__main__":
    asyncio.run(main())

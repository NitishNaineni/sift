import matplotlib.pyplot as plt
import numpy as np
import wgpu
from PIL import Image


def load_image_rgba8(path):
    image = Image.open(path).convert("RGBA")
    width, height = image.size
    data = np.array(image).astype(np.uint8)
    return width, height, data.tobytes()


# Generate mipmaps on the GPU using rendering passes
def generate_mipmaps(device, texture, width, height, mip_levels):
    shader = device.create_shader_module(code=open("mips.wgsl").read())
    pipeline = device.create_render_pipeline(
        layout="auto",
        vertex={"module": shader, "entry_point": "vs"},
        fragment={
            "module": shader,
            "entry_point": "fs",
            "targets": [{"format": wgpu.TextureFormat.rgba8unorm}],
        },
        primitive={"topology": wgpu.PrimitiveTopology.triangle_list},
    )
    sampler = device.create_sampler(min_filter="linear", mag_filter="linear")

    encoder = device.create_command_encoder(label="Mipmap Generation Encoder")
    src_level = 0
    dst_level = src_level + 1

    while dst_level < mip_levels:
        # Create views for the source and destination mip levels
        src_view = texture.create_view(base_mip_level=src_level, mip_level_count=1)
        dst_view = texture.create_view(base_mip_level=dst_level, mip_level_count=1)

        # Create a bind group for the source texture
        bind_group = device.create_bind_group(
            layout=pipeline.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": sampler},
                {"binding": 1, "resource": src_view},
            ],
        )

        # Begin the render pass for the next mip level
        render_pass = encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": dst_view,
                    "clear_value": (0.0, 0.0, 0.0, 1.0),  # Clear to black
                    "store_op": "store",
                    "load_op": "load",
                }
            ],
        )

        render_pass.set_pipeline(pipeline)
        render_pass.set_bind_group(0, bind_group)
        render_pass.draw(6, 1, 0, 0)
        render_pass.end()

        src_level += 1
        dst_level += 1

    # Submit the commands to the GPU
    device.queue.submit([encoder.finish()])


# Read mip levels from GPU and visualize with matplotlib
def read_and_plot_mipmaps(device, texture, width, height, mip_level_count):
    plt.figure(figsize=(10, 10))
    current_width, current_height = width, height

    for mip_level in range(mip_level_count):
        # Calculate bytes_per_row aligned to 256 bytes
        bytes_per_row = (current_width * 4 + 255) & ~255

        # Calculate the buffer size based on the aligned bytes_per_row
        buffer_size = bytes_per_row * current_height

        buffer = device.create_buffer(
            size=buffer_size, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ
        )
        encoder = device.create_command_encoder(label=f"Read Mip Level {mip_level}")
        encoder.copy_texture_to_buffer(
            {
                "texture": texture,
                "mip_level": mip_level,
                "origin": (0, 0, 0),
            },
            {
                "buffer": buffer,
                "offset": 0,
                "bytes_per_row": bytes_per_row,
                "rows_per_image": current_height,
            },
            (current_width, current_height, 1),
        )
        device.queue.submit([encoder.finish()])

        # Map the buffer and read the data
        buffer.map_sync(wgpu.MapMode.READ)
        data = np.frombuffer(buffer.read_mapped(), dtype=np.uint8)

        # Reshape the data, taking into account the padding
        data = data.reshape(current_height, bytes_per_row // 4, 4)[:, :current_width]

        buffer.unmap()

        # Plot the mip level
        plt.subplot(1, mip_level_count, mip_level + 1)
        plt.imshow(data)
        plt.axis("off")
        plt.title(f"Mip {mip_level}")

        current_width = max(1, current_width // 2)
        current_height = max(1, current_height // 2)

    plt.tight_layout()
    plt.show()


# Main function to load the image, generate mipmaps, and plot them
def main():
    # Initialize WGPU device
    # canvas = WgpuCanvas()
    gpu = wgpu.GPU()
    adapter = gpu.request_adapter_sync(canvas=None)
    device = adapter.request_device_sync()

    # Load the image and create the texture
    width, height, image_data = load_image_rgba8("test.png")
    texture = device.create_texture(
        size=(width, height, 1),
        format=wgpu.TextureFormat.rgba8unorm,
        usage=wgpu.TextureUsage.TEXTURE_BINDING
        | wgpu.TextureUsage.COPY_DST
        | wgpu.TextureUsage.RENDER_ATTACHMENT
        | wgpu.TextureUsage.COPY_SRC,
        mip_level_count=4,
    )

    # Upload the base level (level 0) image
    upload_buffer = device.create_buffer_with_data(data=image_data, usage=wgpu.BufferUsage.COPY_SRC)
    encoder = device.create_command_encoder(label="Base Level Upload")
    encoder.copy_buffer_to_texture(
        {
            "buffer": upload_buffer,
            "offset": 0,
            "bytes_per_row": width * 4,
            "rows_per_image": height,
        },
        {"texture": texture, "mip_level": 0, "origin": (0, 0, 0)},
        (width, height, 1),
    )
    device.queue.submit([encoder.finish()])

    generate_mipmaps(device, texture, width, height, 4)
    read_and_plot_mipmaps(device, texture, width, height, 4)


if __name__ == "__main__":
    main()

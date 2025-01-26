import os
import struct

import numpy as np
import wgpu
from PIL import Image

OCTAVES: int = 5
SCALES: int = 3
SIGMA_MIN: float = 1.6
SIGMA_BASE: float = 0.8
WORKGROUP_SIZE: tuple[int, int, int] = (8, 8, 1)


def generate_kernels(sigma_min: float, sigma_base: float, num_scales: int) -> np.ndarray:
    num_images_per_octave = num_scales + 3
    k = 2 ** (1.0 / num_scales)
    gaussian_kernels = np.zeros(num_images_per_octave, dtype=np.float32)
    gaussian_kernels[0] = np.sqrt(sigma_min**2 - sigma_base**2)
    for image_index in range(1, num_images_per_octave):
        sigma_previous = (k ** (image_index - 1)) * sigma_min
        sigma_total = k * sigma_previous
        sigma = np.sqrt(sigma_total**2 - sigma_previous**2)
        gaussian_kernels[image_index] = sigma
    return gaussian_kernels


def create_gaussian_pyramid_buffers(
    device: wgpu.GPUDevice, width: int, height: int
) -> tuple[list[list[wgpu.GPUBuffer]], list[list[wgpu.GPUBuffer]]]:
    pyramid, pyramid_read = [], []
    for octave in range(OCTAVES):
        octave_width = max(1, width >> octave)
        octave_height = max(1, height >> octave)
        buffer_size = octave_width * octave_height * 4
        buffers, buffers_read = [], []
        for _ in range(SCALES + 4):
            buf = device.create_buffer(
                size=buffer_size, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
            )
            buffers.append(buf)
            buf_read = device.create_buffer(
                size=buffer_size, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ
            )
            buffers_read.append(buf_read)
        pyramid.append(buffers)
        pyramid_read.append(buffers_read)
    return pyramid, pyramid_read


def create_dog_buffers(
    device: wgpu.GPUDevice, width: int, height: int
) -> tuple[list[list[wgpu.GPUBuffer]], list[list[wgpu.GPUBuffer]]]:
    pyramid, pyramid_read = [], []
    for octave in range(OCTAVES):
        octave_width = max(1, width >> octave)
        octave_height = max(1, height >> octave)
        buffer_size = octave_width * octave_height * 4
        buffers, buffers_read = [], []
        for _ in range(SCALES + 2):
            buf = device.create_buffer(
                size=buffer_size, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
            )
            buffers.append(buf)
            buf_read = device.create_buffer(
                size=buffer_size, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ
            )
            buffers_read.append(buf_read)
        pyramid.append(buffers)
        pyramid_read.append(buffers_read)
    return pyramid, pyramid_read


def create_param_buffers(
    device: wgpu.GPUDevice, width: int, height: int
) -> tuple[dict[int, wgpu.GPUBuffer], dict[int, wgpu.GPUBuffer], dict[int, wgpu.GPUBuffer]]:
    octave_params, scale_params, box_params = {}, {}, {}

    for octave in range(OCTAVES):
        octave_width = max(1, width >> octave)
        octave_height = max(1, height >> octave)
        data = struct.pack("II", octave_width, octave_height)
        buf = device.create_buffer_with_data(
            data=data, usage="UNIFORM", label=f"octave_{octave}_params"
        )
        octave_params[octave] = buf

    kernels = generate_kernels(SIGMA_MIN, SIGMA_BASE, SCALES)
    for scale in range(SCALES + 3):
        sigma = kernels[scale]
        data = struct.pack("f", sigma)
        buf = device.create_buffer_with_data(
            data=data, usage="UNIFORM", label=f"scale_{scale}_params"
        )
        scale_params[scale] = buf

    for octave in range(1, OCTAVES):
        prev_octave_width = max(1, width >> (octave - 1))
        prev_octave_height = max(1, height >> (octave - 1))
        octave_width = max(1, width >> octave)
        octave_height = max(1, height >> octave)
        data = struct.pack(
            "IIII", prev_octave_width, prev_octave_height, octave_width, octave_height
        )
        buf = device.create_buffer_with_data(
            data=data, usage="UNIFORM", label=f"box_params_{octave - 1}_to_{octave}"
        )
        box_params[octave] = buf

    return octave_params, scale_params, box_params


def create_compute_pipeline(
    device: wgpu.GPUDevice, shader_path: str, entry_point: str, layout: wgpu.GPUPipelineLayout
) -> wgpu.GPUComputePipeline:
    with open(shader_path, "r") as f:
        code = f.read()
    module = device.create_shader_module(code=code)
    pipeline = device.create_compute_pipeline(
        layout=layout, compute={"module": module, "entry_point": entry_point}
    )
    return pipeline


def dispatch(
    command_encoder: wgpu.GPUCommandEncoder,
    pipeline: wgpu.GPUComputePipeline,
    bind_group: wgpu.GPUBindGroup,
    width: int,
    height: int,
) -> None:
    workgroup_count_x = (width + WORKGROUP_SIZE[0] - 1) // WORKGROUP_SIZE[0]
    workgroup_count_y = (height + WORKGROUP_SIZE[1] - 1) // WORKGROUP_SIZE[1]
    pass_enc = command_encoder.begin_compute_pass()
    pass_enc.set_pipeline(pipeline)
    pass_enc.set_bind_group(0, bind_group)
    pass_enc.dispatch_workgroups(workgroup_count_x, workgroup_count_y)
    pass_enc.end()


def save_gaussian_pyramid(
    device: wgpu.GPUDevice,
    pyramid: list[list[wgpu.GPUBuffer]],
    pyramid_read: list[list[wgpu.GPUBuffer]],
    width: int,
    height: int,
) -> None:
    command_encoder = device.create_command_encoder()
    for octave in range(OCTAVES):
        octave_width = max(1, width >> octave)
        octave_height = max(1, height >> octave)
        for scale in range(SCALES + 3):
            size_bytes = octave_width * octave_height * 4
            command_encoder.copy_buffer_to_buffer(
                source=pyramid[octave][scale],
                source_offset=0,
                destination=pyramid_read[octave][scale],
                destination_offset=0,
                size=size_bytes,
            )
    device.queue.submit([command_encoder.finish()])
    os.makedirs("output/scale_space", exist_ok=True)
    for octave in range(OCTAVES):
        octave_width = max(1, width >> octave)
        octave_height = max(1, height >> octave)
        for scale in range(SCALES + 3):
            buf = pyramid_read[octave][scale]
            buf.map_sync(mode=wgpu.MapMode.READ)
            data = buf.read_mapped()
            img_data = np.frombuffer(data, dtype=np.float32).reshape((octave_height, octave_width))
            img_data = (img_data * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(img_data).save(f"output/scale_space/octave_{octave}_scale_{scale}.png")
            buf.unmap()


def save_dog_pyramid(
    device: wgpu.GPUDevice,
    pyramid: list[list[wgpu.GPUBuffer]],
    pyramid_read: list[list[wgpu.GPUBuffer]],
    width: int,
    height: int,
) -> None:
    command_encoder = device.create_command_encoder()
    for octave in range(OCTAVES):
        octave_width = max(1, width >> octave)
        octave_height = max(1, height >> octave)
        for scale in range(SCALES + 2):
            size_bytes = octave_width * octave_height * 4
            command_encoder.copy_buffer_to_buffer(
                source=pyramid[octave][scale],
                source_offset=0,
                destination=pyramid_read[octave][scale],
                destination_offset=0,
                size=size_bytes,
            )
    device.queue.submit([command_encoder.finish()])
    os.makedirs("output/dog", exist_ok=True)
    for octave in range(OCTAVES):
        octave_width = max(1, width >> octave)
        octave_height = max(1, height >> octave)
        for scale in range(SCALES + 2):
            buf = pyramid_read[octave][scale]
            buf.map_sync(mode=wgpu.MapMode.READ)
            data = buf.read_mapped()
            img_data = np.frombuffer(data, dtype=np.float32).reshape((octave_height, octave_width))
            img_data = (img_data + 1) / 2
            img_data = (img_data * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(img_data).save(f"output/dog/octave_{octave}_scale_{scale}.png")
            buf.unmap()


def main() -> None:
    gpu = wgpu.GPU()
    adapter = gpu.request_adapter_sync(
        power_preference="high-performance", force_fallback_adapter=False
    )
    device = adapter.request_device_sync()
    image = Image.open("data/0.jpg").convert("L")
    width, height = image.size
    base_image_np = np.array(image, dtype=np.float32) / 255.0
    base_image_buffer = device.create_buffer_with_data(
        data=base_image_np, usage="STORAGE", label="base_image_buffer"
    )
    gaussian_pyramid, gaussian_pyramid_read = create_gaussian_pyramid_buffers(device, width, height)
    dog_pyramid, dog_pyramid_read = create_dog_buffers(device, width, height)
    octave_params, scale_params, box_params = create_param_buffers(device, width, height)
    gaussian_bind_group_layout = device.create_bind_group_layout(
        entries=[
            {
                "binding": 0,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.uniform},
            },
            {
                "binding": 1,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.uniform},
            },
            {
                "binding": 2,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
            },
            {
                "binding": 3,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.storage},
            },
        ]
    )
    gaussian_pipeline_layout = device.create_pipeline_layout(
        bind_group_layouts=[gaussian_bind_group_layout]
    )
    box_bind_group_layout = device.create_bind_group_layout(
        entries=[
            {
                "binding": 0,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.uniform},
            },
            {
                "binding": 1,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
            },
            {
                "binding": 2,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.storage},
            },
        ]
    )
    box_pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[box_bind_group_layout])
    dog_bind_group_layout = device.create_bind_group_layout(
        entries=[
            {
                "binding": 0,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.uniform},
            },
            {
                "binding": 1,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
            },
            {
                "binding": 2,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
            },
            {
                "binding": 3,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.storage},
            },
        ]
    )
    dog_pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[dog_bind_group_layout])
    hblur_pipeline = create_compute_pipeline(
        device, "shaders/gaussian.wgsl", "hblur", gaussian_pipeline_layout
    )
    vblur_pipeline = create_compute_pipeline(
        device, "shaders/gaussian.wgsl", "vblur", gaussian_pipeline_layout
    )
    box_pipeline = create_compute_pipeline(device, "shaders/box.wgsl", "main", box_pipeline_layout)
    dog_pipeline = create_compute_pipeline(device, "shaders/diff.wgsl", "main", dog_pipeline_layout)
    command_encoder = device.create_command_encoder()
    for octave in range(OCTAVES):
        octave_width = max(1, width >> octave)
        octave_height = max(1, height >> octave)
        if octave > 0:
            bgroup = device.create_bind_group(
                layout=box_bind_group_layout,
                entries=[
                    {"binding": 0, "resource": {"buffer": box_params[octave]}},
                    {"binding": 1, "resource": {"buffer": gaussian_pyramid[octave - 1][SCALES]}},
                    {"binding": 2, "resource": {"buffer": gaussian_pyramid[octave][0]}},
                ],
            )
            dispatch(command_encoder, box_pipeline, bgroup, octave_width, octave_height)
        else:
            hblur_bgroup = device.create_bind_group(
                layout=gaussian_bind_group_layout,
                entries=[
                    {"binding": 0, "resource": {"buffer": octave_params[0]}},
                    {"binding": 1, "resource": {"buffer": scale_params[0]}},
                    {"binding": 2, "resource": {"buffer": base_image_buffer}},
                    {"binding": 3, "resource": {"buffer": gaussian_pyramid[0][-1]}},
                ],
            )
            dispatch(command_encoder, hblur_pipeline, hblur_bgroup, octave_width, octave_height)
            vblur_bgroup = device.create_bind_group(
                layout=gaussian_bind_group_layout,
                entries=[
                    {"binding": 0, "resource": {"buffer": octave_params[0]}},
                    {"binding": 1, "resource": {"buffer": scale_params[0]}},
                    {"binding": 2, "resource": {"buffer": gaussian_pyramid[0][-1]}},
                    {"binding": 3, "resource": {"buffer": gaussian_pyramid[0][0]}},
                ],
            )
            dispatch(command_encoder, vblur_pipeline, vblur_bgroup, octave_width, octave_height)
        for scale in range(1, SCALES + 3):
            hblur_bgroup = device.create_bind_group(
                layout=gaussian_bind_group_layout,
                entries=[
                    {"binding": 0, "resource": {"buffer": octave_params[octave]}},
                    {"binding": 1, "resource": {"buffer": scale_params[scale]}},
                    {"binding": 2, "resource": {"buffer": gaussian_pyramid[octave][scale - 1]}},
                    {"binding": 3, "resource": {"buffer": gaussian_pyramid[octave][-1]}},
                ],
            )
            dispatch(command_encoder, hblur_pipeline, hblur_bgroup, octave_width, octave_height)
            vblur_bgroup = device.create_bind_group(
                layout=gaussian_bind_group_layout,
                entries=[
                    {"binding": 0, "resource": {"buffer": octave_params[octave]}},
                    {"binding": 1, "resource": {"buffer": scale_params[scale]}},
                    {"binding": 2, "resource": {"buffer": gaussian_pyramid[octave][-1]}},
                    {"binding": 3, "resource": {"buffer": gaussian_pyramid[octave][scale]}},
                ],
            )
            dispatch(command_encoder, vblur_pipeline, vblur_bgroup, octave_width, octave_height)

    for octave in range(OCTAVES):
        octave_width = max(1, width >> octave)
        octave_height = max(1, height >> octave)
        for scale in range(SCALES + 2):
            dog_bgroup = device.create_bind_group(
                layout=dog_bind_group_layout,
                entries=[
                    {"binding": 0, "resource": {"buffer": octave_params[octave]}},
                    {"binding": 1, "resource": {"buffer": gaussian_pyramid[octave][scale + 1]}},
                    {"binding": 2, "resource": {"buffer": gaussian_pyramid[octave][scale]}},
                    {"binding": 3, "resource": {"buffer": dog_pyramid[octave][scale]}},
                ],
            )
            dispatch(command_encoder, dog_pipeline, dog_bgroup, octave_width, octave_height)

    device.queue.submit([command_encoder.finish()])
    save_gaussian_pyramid(device, gaussian_pyramid, gaussian_pyramid_read, width, height)
    save_dog_pyramid(device, dog_pyramid, dog_pyramid_read, width, height)


if __name__ == "__main__":
    main()

import os
import struct

import numpy as np
import wgpu
from PIL import Image

OCTAVES = 4

gpu = wgpu.GPU()
adapter = gpu.request_adapter_sync(
    power_preference="high-performance", force_fallback_adapter=False
)
device = adapter.request_device_sync()


image = Image.open("data/0.jpg").convert("L")
width, height = image.size


image_pyramid = []
for level in range(OCTAVES):
    buffer_size = max(1, width >> level) * max(1, height >> level) * 4
    usage = (
        wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC
        if level == 0
        else wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
    )
    buffer = device.create_buffer(size=buffer_size, usage=usage)
    image_pyramid.append(buffer)

image_pyramid_read = []
for level in range(OCTAVES):
    buffer_size = max(1, width >> level) * max(1, height >> level) * 4
    usage = wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ
    buffer = device.create_buffer(size=buffer_size, usage=usage)
    image_pyramid_read.append(buffer)

upload_buffer = device.create_buffer(size=width * height * 4, usage="MAP_WRITE | COPY_SRC")
upload_buffer.map_sync(mode="WRITE", offset=0, size=upload_buffer.size)
upload_buffer.write_mapped(np.asarray(image).astype(np.float32))
upload_buffer.unmap()

param_buffers = {}
for level in range(1, OCTAVES):
    src_width = max(1, width >> (level - 1))
    src_height = max(1, height >> (level - 1))
    dst_width = max(1, width >> level)
    dst_height = max(1, height >> level)

    param_buffer = device.create_buffer_with_data(
        label=f"param_buffer_{level}",
        data=struct.pack("4I", src_width, src_height, dst_width, dst_height),
        usage="UNIFORM",
    )
    param_buffers[level] = param_buffer

with open("shaders/box.wgsl", "r") as f:
    box_code = f.read()
box_module = device.create_shader_module(code=box_code)

bind_group_layout = device.create_bind_group_layout(
    entries=[
        {
            "binding": 0,
            "visibility": wgpu.ShaderStage.COMPUTE,
            "buffer": {
                "type": wgpu.BufferBindingType.uniform,
            },
        },
        {
            "binding": 1,
            "visibility": wgpu.ShaderStage.COMPUTE,
            "buffer": {
                "type": wgpu.BufferBindingType.read_only_storage,
            },
        },
        {
            "binding": 2,
            "visibility": wgpu.ShaderStage.COMPUTE,
            "buffer": {
                "type": wgpu.BufferBindingType.storage,
            },
        },
    ]
)

pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[bind_group_layout])

compute_pipeline = device.create_compute_pipeline(
    layout=pipeline_layout,
    compute={
        "module": box_module,
        "entry_point": "main",
    },
)

command_encoder = device.create_command_encoder()

command_encoder.copy_buffer_to_buffer(
    source=upload_buffer,
    source_offset=0,
    destination=image_pyramid[0],
    destination_offset=0,
    size=image_pyramid[0].size,
)
for level in range(1, OCTAVES):
    bind_group = device.create_bind_group(
        layout=bind_group_layout,
        entries=[
            {"binding": 0, "resource": {"buffer": param_buffers[level]}},
            {"binding": 1, "resource": {"buffer": image_pyramid[level - 1]}},
            {"binding": 2, "resource": {"buffer": image_pyramid[level]}},
        ],
    )

    compute_pass = command_encoder.begin_compute_pass()
    compute_pass.set_pipeline(compute_pipeline)
    compute_pass.set_bind_group(index=0, bind_group=bind_group)

    level_width = max(1, width >> level)
    level_height = max(1, height >> level)
    workgroup_size = (8, 8, 1)
    workgroup_count_x = (level_width + workgroup_size[0] - 1) // workgroup_size[0]
    workgroup_count_y = (level_height + workgroup_size[1] - 1) // workgroup_size[1]
    compute_pass.dispatch_workgroups(
        workgroup_count_x=workgroup_count_x, workgroup_count_y=workgroup_count_y
    )

    compute_pass.end()

command_buffer = command_encoder.finish()
device.queue.submit([command_buffer])

command_encoder = device.create_command_encoder()
for level in range(OCTAVES):
    command_encoder.copy_buffer_to_buffer(
        source=image_pyramid[level],
        source_offset=0,
        destination=image_pyramid_read[level],
        destination_offset=0,
        size=image_pyramid[level].size,
    )
command_buffer = command_encoder.finish()
device.queue.submit([command_buffer])

os.makedirs("results", exist_ok=True)
results = []
for level in range(OCTAVES):
    image_pyramid_read[level].map_sync(mode="READ", offset=0, size=image_pyramid_read[level].size)
    image_data = image_pyramid_read[level].read_mapped()
    image_pyramid_read[level].unmap()
    level_width = max(1, width >> level)
    level_height = max(1, height >> level)
    image_array = np.frombuffer(image_data, dtype=np.float32).reshape(level_height, level_width)
    image = Image.fromarray(image_array.astype(np.uint8))
    image.save(f"results/level_{level}.png")

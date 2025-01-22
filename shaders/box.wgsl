struct Params {
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> src_level: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst_level: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= params.dst_width || global_id.y >= params.dst_height) {
        return;
    }

    let dst_idx = global_id.y * params.dst_width + global_id.x;

    let src_x = global_id.x * 2;
    let src_y = global_id.y * 2;

    let src_x0 = min(src_x, params.src_width - 1);
    let src_y0 = min(src_y, params.src_height - 1);
    let src_x1 = min(src_x0 + 1, params.src_width - 1);
    let src_y1 = min(src_y0 + 1, params.src_height - 1);

    let idx00 = src_y0 * params.src_width + src_x0;
    let idx01 = idx00 + 1;
    let idx10 = idx00 + params.src_width;
    let idx11 = idx10 + 1;

    let c00 = src_level[idx00];
    let c01 = src_level[idx01];
    let c10 = src_level[idx10];
    let c11 = src_level[idx11];

    let c = (c00 + c01 + c10 + c11) * 0.25;

    dst_level[dst_idx] = c;
}
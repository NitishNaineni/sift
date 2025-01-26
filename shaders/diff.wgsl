struct OctaveParams {
    width: u32,
    height: u32,
}

@group(0) @binding(0) var<uniform> octave_params: OctaveParams;
@group(0) @binding(1) var<storage, read> high: array<f32>;
@group(0) @binding(2) var<storage, read> low: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;

fn index(x: u32, y: u32, width: u32) -> u32 {
    return y * width + x;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= octave_params.width || global_id.y >= octave_params.height) {
        return;
    }

    let x = global_id.x;
    let y = global_id.y;

    dst[index(x, y, octave_params.width)] = high[index(x, y, octave_params.width)] - low[index(x, y, octave_params.width)];
}
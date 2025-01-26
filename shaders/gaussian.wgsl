struct OctaveParams {
    width: u32,
    height: u32,
}

struct ScaleParams {
    sigma: f32,
}

fn index(x: u32, y: u32, width: u32) -> u32 {
    return y * width + x;
}

fn gaussian_weight(x: f32, sigma: f32) -> f32 {
    let sigma_sq = sigma * sigma;
    let exponent = -(x * x) / (2.0 * sigma_sq);
    return (1.0 / sqrt(2.0 * 3.14159 * sigma_sq)) * exp(exponent);
}

@group(0) @binding(0) var<uniform> octave_params: OctaveParams;
@group(0) @binding(1) var<uniform> scale_params: ScaleParams;
@group(0) @binding(2) var<storage, read> src: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn hblur(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= octave_params.width || global_id.y >= octave_params.height) {
        return;
    }

    let kernel_radius = u32(ceil(3.0 * scale_params.sigma));
    let kernel_size = 2u * kernel_radius + 1u;
    
    var sum = 0.0;
    var weight_sum = 0.0;

    for (var i = 0u; i < kernel_size; i++) {
        let offset = i32(i) - i32(kernel_radius);
        let x = i32(global_id.x) + offset;
        
        if (x >= 0 && x < i32(octave_params.width)) {
            let weight = gaussian_weight(f32(offset), scale_params.sigma);
            let sample = src[index(u32(x), global_id.y, octave_params.width)];
            sum += sample * weight;
            weight_sum += weight;
        }
    }

    dst[index(global_id.x, global_id.y, octave_params.width)] = sum / weight_sum;
}

@compute @workgroup_size(8, 8, 1)
fn vblur(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= octave_params.width || global_id.y >= octave_params.height) {
        return;
    }

    let kernel_radius = u32(ceil(3.0 * scale_params.sigma));
    let kernel_size = 2u * kernel_radius + 1u;

    var sum = 0.0;
    var weight_sum = 0.0;

    for (var i = 0u; i < kernel_size; i++) {
        let offset = i32(i) - i32(kernel_radius);
        let y = i32(global_id.y) + offset;

        if (y >= 0 && y < i32(octave_params.height)) {
            let weight = gaussian_weight(f32(offset), scale_params.sigma);
            let sample = src[index(global_id.x, u32(y), octave_params.width)];
            sum += sample * weight;
            weight_sum += weight;
        }
    }

    dst[index(global_id.x, global_id.y, octave_params.width)] = sum / weight_sum;
}

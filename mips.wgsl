struct VSOutput {
    @builtin(position) position: vec4f,
    @location(0) texcoord: vec2f,
};

@vertex
fn vs(@builtin(vertex_index) vertexIndex: u32) -> VSOutput {
    var pos: vec2f;
    switch (vertexIndex) {
        case 0u: { pos = vec2f(0.0, 0.0); }
        case 1u: { pos = vec2f(1.0, 0.0); }
        case 2u: { pos = vec2f(0.0, 1.0); }
        case 3u: { pos = vec2f(0.0, 1.0); }
        case 4u: { pos = vec2f(1.0, 0.0); }
        case 5u: { pos = vec2f(1.0, 1.0); }
        default: { pos = vec2f(0.0, 0.0); }
    }

    var vsOutput: VSOutput;
    vsOutput.position = vec4f(pos * 2.0 - 1.0, 0.0, 1.0);
    vsOutput.texcoord = vec2f(pos.x, 1.0 - pos.y);
    return vsOutput;
}

@group(0) @binding(0) var ourSampler: sampler;
@group(0) @binding(1) var ourTexture: texture_2d<f32>;

@fragment
fn fs(fsInput: VSOutput) -> @location(0) vec4f {
    return textureSample(ourTexture, ourSampler, fsInput.texcoord);
}

struct VSOutput {
    @builtin(position) position: vec4f,
    @location(0) texcoord: vec2f,
};

var<private> POSITIONS: array<vec2f, 4> = array<vec2f, 4>(
    vec2f(0.0, 0.0),
    vec2f(1.0, 0.0),
    vec2f(0.0, 1.0),
    vec2f(1.0, 1.0),
);

@vertex
fn vs(@builtin(vertex_index) vertexIndex: u32) -> VSOutput {
    let pos = POSITIONS[vertexIndex];
    var vsOutput: VSOutput;
    let transformedPos = pos * 2.0 - 1.0;
    vsOutput.position = vec4f(transformedPos, 0.0, 1.0);
    vsOutput.texcoord = vec2f(pos.x, 1.0 - pos.y);
    return vsOutput;
}

@group(0) @binding(0) var ourSampler: sampler;
@group(0) @binding(1) var ourTexture: texture_2d<f32>;

@fragment
fn fs(fsInput: VSOutput) -> @location(0) vec4f {
    return textureSample(ourTexture, ourSampler, fsInput.texcoord);
}
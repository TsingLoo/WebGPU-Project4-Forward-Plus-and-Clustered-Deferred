@group(0) @binding(0) var sourceTexture: texture_2d<f32>;
@group(0) @binding(1) var sourceSampler: sampler;

struct FragmentInput {
    @location(0) uv: vec2f,
}

@fragment
fn main(in: FragmentInput) -> @location(0) vec4f {
    let flipped_uv = vec2f(in.uv.x, 1.0 - in.uv.y);
    return textureSample(sourceTexture, sourceSampler, flipped_uv);
}
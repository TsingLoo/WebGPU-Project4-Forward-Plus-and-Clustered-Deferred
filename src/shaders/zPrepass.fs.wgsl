@group(2) @binding(0) var diffuseTex: texture_2d<f32>;
@group(2) @binding(1) var diffuseTexSampler: sampler;

struct FragmentInput {
    @location(2) uv: vec2f, 
}

@fragment
fn main(in: FragmentInput) {
    let alpha = textureSample(diffuseTex, diffuseTexSampler, in.uv).a;
    
    if (alpha < 0.5) {
        discard;
    }
}
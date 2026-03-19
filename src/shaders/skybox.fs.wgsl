// Skybox fragment shader - samples environment cubemap

@group(0) @binding(1) var envMap: texture_cube<f32>;
@group(0) @binding(2) var envSampler: sampler;

struct FragmentInput {
    @location(0) localPos: vec3f,
}

@fragment
fn main(in: FragmentInput) -> @location(0) vec4f {
    let dir = normalize(in.localPos);
    let color = textureSample(envMap, envSampler, dir).rgb;
    
    // Simple tone mapping for HDR sky
    let mapped = color / (color + vec3f(1.0));
    
    // Gamma correction
    let gamma = pow(mapped, vec3f(1.0/2.2));
    
    return vec4f(gamma, 1.0);
}

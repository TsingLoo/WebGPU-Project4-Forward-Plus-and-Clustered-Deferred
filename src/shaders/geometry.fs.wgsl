@group(${bindGroup_material}) @binding(0) var diffuseTex: texture_2d<f32>;
@group(${bindGroup_material}) @binding(1) var diffuseTexSampler: sampler;

struct PBRParams {
    roughness: f32,
    metallic: f32,
    has_mr_texture: f32,
    has_normal_texture: f32,
    base_color_factor: vec4f,
    _reserved: vec4f,
}
@group(${bindGroup_material}) @binding(2) var<uniform> pbrParams: PBRParams;
@group(${bindGroup_material}) @binding(3) var metallicRoughnessTex: texture_2d<f32>;
@group(${bindGroup_material}) @binding(4) var metallicRoughnessTexSampler: sampler;
@group(${bindGroup_material}) @binding(5) var normalTex: texture_2d<f32>;
@group(${bindGroup_material}) @binding(6) var normalTexSampler: sampler;

struct FragmentInput
{
    @location(0) pos: vec3f,
    @location(1) nor: vec3f,
    @location(2) uv: vec2f,
    @location(3) tangent_world: vec4f
}

struct GBufferOutput {
    @location(0) albedo : vec4f,
    @location(1) normal : vec4f,
    @location(2) position : vec4f,
    @location(3) specular_material : vec4f,
}

@fragment
fn main(in: FragmentInput) ->  GBufferOutput
{
    let diffuseColor = textureSample(diffuseTex, diffuseTexSampler, in.uv) * pbrParams.base_color_factor;
    if (diffuseColor.a < 0.5f) {
        discard;
    }

    // Per-pixel metallic/roughness from texture (glTF: G = roughness, B = metallic)
    var metallic = pbrParams.metallic;
    var roughness = pbrParams.roughness;
    if (pbrParams.has_mr_texture > 0.5) {
        let mrSample = textureSample(metallicRoughnessTex, metallicRoughnessTexSampler, in.uv);
        roughness = roughness * mrSample.g;
        metallic = metallic * mrSample.b;
    }

    // Normal mapping: build TBN matrix and sample normal map
    var N = normalize(in.nor);
    if (pbrParams.has_normal_texture > 0.5) {
        let T = normalize(in.tangent_world.xyz);
        let B = cross(N, T) * in.tangent_world.w;
        let tbn = mat3x3f(T, B, N);
        let normalSample = textureSample(normalTex, normalTexSampler, in.uv).rgb;
        let tangentNormal = normalSample * 2.0 - 1.0;
        N = normalize(tbn * tangentNormal);
    }

    var output: GBufferOutput;
    output.albedo = diffuseColor;
    output.normal = vec4f(N, 1.0); 
    output.position = vec4f(in.pos, 1.0);

    // Store PBR params from material uniform
    output.specular_material = vec4f(roughness, metallic, 0.0, 1.0);
    
    return output;
}

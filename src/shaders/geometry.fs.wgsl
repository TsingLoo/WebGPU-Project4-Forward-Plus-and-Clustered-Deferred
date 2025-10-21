@group(${bindGroup_material}) @binding(0) var diffuseTex: texture_2d<f32>;
@group(${bindGroup_material}) @binding(1) var diffuseTexSampler: sampler;

struct FragmentInput
{
    @location(0) pos: vec3f,
    @location(1) nor: vec3f,
    @location(2) uv: vec2f
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
    let diffuseColor = textureSample(diffuseTex, diffuseTexSampler, in.uv);
    if (diffuseColor.a < 0.5f) {
        discard;
    }

    var output: GBufferOutput;

    output.albedo = diffuseColor;

    output.normal = vec4f(in.nor, 1.0); 

    output.position = vec4f(in.pos, 1.0);

    let roughness = 0.5f;
    let metallic = 0.0f;
    output.specular_material = vec4f(roughness, metallic, 0.0, 1.0);
    
    return output;
}

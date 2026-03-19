// CHECKITOUT: you can use this vertex shader for all of the renderers

// TODO-1.3: add a uniform variable here for camera uniforms (of type CameraUniforms)
// make sure to use ${bindGroup_scene} for the group
@group(${bindGroup_scene}) @binding(0) var<uniform> camera: CameraUniforms;

@group(${bindGroup_model}) @binding(0) var<uniform> modelMat: mat4x4f;

struct VertexInput
{
    @location(0) pos: vec3f,
    @location(1) nor: vec3f,
    @location(2) uv: vec2f,
    @location(3) tangent: vec4f
}

struct VertexOutput
{
    @builtin(position) fragPos: vec4f,
    @location(0) pos: vec3f,
    @location(1) nor: vec3f,
    @location(2) uv: vec2f,
    @location(3) tangent_world: vec4f
}

@vertex
fn main(in: VertexInput) -> VertexOutput
{
    let modelPos = modelMat * vec4(in.pos, 1);

    var out: VertexOutput;
    out.fragPos = camera.view_proj_mat * modelPos;
    out.pos = modelPos.xyz / modelPos.w;
    out.nor = normalize((modelMat * vec4(in.nor, 0.0)).xyz);
    out.uv = in.uv;
    // Transform tangent direction by model matrix, preserve handedness in w
    out.tangent_world = vec4f(normalize((modelMat * vec4(in.tangent.xyz, 0.0)).xyz), in.tangent.w);
    return out;
}

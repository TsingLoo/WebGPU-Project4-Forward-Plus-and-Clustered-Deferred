// VSM Shadow Vertex Shader
// Transforms geometry to light clip space for a given clipmap level.
// The VP matrix is set per-clipmap-level draw call.

@group(0) @binding(0) var<uniform> lightViewProj: mat4x4f;
@group(1) @binding(0) var<uniform> modelMat: mat4x4f;

@group(2) @binding(0) var diffuseTex: texture_2d<f32>;
@group(2) @binding(1) var diffuseTexSampler: sampler;

struct VertexInput {
    @location(0) pos: vec3f,
    @location(1) nor: vec3f,
    @location(2) uv: vec2f,
    @location(3) tangent: vec4f
}

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
}

@vertex
fn main(in: VertexInput) -> VertexOutput {
    let worldPos = modelMat * vec4f(in.pos, 1.0);
    let clipPos = lightViewProj * worldPos;

    var out: VertexOutput;
    out.position = clipPos;
    out.uv = in.uv;
    return out;
}

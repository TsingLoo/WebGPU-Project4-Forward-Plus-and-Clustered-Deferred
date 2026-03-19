// Skybox vertex shader - generates fullscreen triangle

@group(0) @binding(0) var<uniform> camera: CameraUniforms;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) localPos: vec3f,
}

@vertex
fn main(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
    // fullscreen triangle positions in NDC
    var positions = array<vec2f, 3>(
        vec2f(-1.0, -1.0),
        vec2f( 3.0, -1.0),
        vec2f(-1.0,  3.0)
    );

    let pos = positions[vertexIndex];
    
    // Reconstruct world direction from clip coords
    // Invert the VP matrix to go from NDC → World
    let invProj = camera.inv_proj_mat;
    let invView = transpose(mat4x4f(
        camera.view_mat[0],
        camera.view_mat[1],
        camera.view_mat[2],
        camera.view_mat[3]
    ));

    // Unproject from clip space to view space (at far plane)
    let viewSpace = invProj * vec4f(pos.x, pos.y, 1.0, 1.0);
    let viewDir = viewSpace.xyz / viewSpace.w;

    // Transform from view space to world space (rotation only)
    let worldDir = (invView * vec4f(viewDir, 0.0)).xyz;

    var output: VertexOutput;
    output.position = vec4f(pos, 1.0, 1.0); // z=1 = far plane (for depth test)
    output.localPos = worldDir;
    return output;
}

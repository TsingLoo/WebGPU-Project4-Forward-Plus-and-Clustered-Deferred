struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
}

@vertex
fn main(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
    const pos = array(
        vec2f(-1.0, -1.0),
        vec2f( 3.0, -1.0),
        vec2f(-1.0,  3.0)
    );

    const uv = array(
        vec2f(0.0, 0.0),
        vec2f(2.0, 0.0),
        vec2f(0.0, 2.0)
    );

    var out: VertexOutput;
    out.position = vec4f(pos[vertexIndex], 0.0, 1.0);
    out.uv = uv[vertexIndex];
    return out;
}
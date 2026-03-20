struct VertexOutput {
    @builtin(position) fragcoord: vec4f,
    @location(0) uv: vec2f,
}

@vertex
fn main(
    @builtin(vertex_index) VertexIndex : u32
) -> VertexOutput {
    var pos = array<vec2f, 3>(
        vec2f(-1.0, -1.0),
        vec2f(3.0, -1.0),
        vec2f(-1.0, 3.0)
    );
    
    var output : VertexOutput;
    output.fragcoord = vec4f(pos[VertexIndex], 0.0, 1.0);
    output.uv = pos[VertexIndex] * vec2f(0.5, -0.5) + vec2f(0.5, 0.5);
    return output;
}

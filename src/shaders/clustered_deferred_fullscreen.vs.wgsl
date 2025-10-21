struct VertexOutput {
    @builtin(position) position: vec4f,
}

@vertex
fn main(
    @builtin(vertex_index) vertexIndex: u32 
) -> VertexOutput {

    const pos = array(
        vec2f(-1.0, -1.0),
        vec2f( 3.0, -1.0),
        vec2f(-1.0,  3.0)
    );

    var out: VertexOutput;
    
    out.position = vec4f(pos[vertexIndex], 0.0, 1.0);
    
    return out;
}
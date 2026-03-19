// Equirectangular to Cubemap compute shader
// Converts a 2D equirectangular environment map (HDRI) into a 6-face cubemap array texture

@group(0) @binding(0) var equiTex: texture_2d<f32>;
@group(0) @binding(1) var outputTex: texture_storage_2d_array<rgba16float, write>;

const PI: f32 = 3.14159265359;
const TWO_PI: f32 = 6.28318530718;

// Convert 3D direction to equirectangular UV coordinates
fn directionToEquirectangularUV(dir: vec3f) -> vec2f {
    let normalizedDir = normalize(dir);
    
    // atan2(z, x) returns [-pi, pi], adding pi to get [0, 2pi], dividing by 2pi to get [0, 1]
    let u = (atan2(normalizedDir.z, normalizedDir.x) + PI) / TWO_PI;
    
    // asin(y) returns [-pi/2, pi/2], map to [0, 1]
    // v=0 at south pole (y=-1), v=1 at north pole (y=1)
    let v = asin(clamp(normalizedDir.y, -1.0, 1.0)) / PI + 0.5;
    
    return vec2f(u, v);
}

// Manual bilinear interpolation using textureLoad
fn sampleEquirectBilinear(uv: vec2f) -> vec4f {
    let texDims = textureDimensions(equiTex);
    let w = f32(texDims.x);
    let h = f32(texDims.y);

    // Convert UV to pixel coordinates (continuous)
    let px = uv.x * w - 0.5;
    let py = uv.y * h - 0.5;

    // Integer pixel coordinates for the four neighbors
    let x0 = i32(floor(px));
    let y0 = i32(floor(py));
    let x1 = x0 + 1;
    let y1 = y0 + 1;

    // Fractional part for interpolation weights
    let fx = px - floor(px);
    let fy = py - floor(py);

    // Wrap U (horizontal) for seamless 360° wrapping, clamp V (vertical) at poles
    let sx0 = ((x0 % i32(texDims.x)) + i32(texDims.x)) % i32(texDims.x);
    let sx1 = ((x1 % i32(texDims.x)) + i32(texDims.x)) % i32(texDims.x);
    let sy0 = clamp(y0, 0, i32(texDims.y) - 1);
    let sy1 = clamp(y1, 0, i32(texDims.y) - 1);

    // Load four neighbors
    let c00 = textureLoad(equiTex, vec2i(sx0, sy0), 0);
    let c10 = textureLoad(equiTex, vec2i(sx1, sy0), 0);
    let c01 = textureLoad(equiTex, vec2i(sx0, sy1), 0);
    let c11 = textureLoad(equiTex, vec2i(sx1, sy1), 0);

    // Bilinear interpolation
    let top = mix(c00, c10, fx);
    let bottom = mix(c01, c11, fx);
    return mix(top, bottom, fy);
}

// Map 2D face coordinates to 3D direction for a given cubemap face
fn getDirectionForFace(coord: vec2f, face: u32) -> vec3f {
    // coord is [0, 1], map to [-1, 1]
    let c = coord * 2.0 - 1.0;
    
    // Cubemap faces: 
    // 0: +X (Right)
    // 1: -X (Left)
    // 2: +Y (Top)
    // 3: -Y (Bottom)
    // 4: +Z (Front)
    // 5: -Z (Back)
    var dir = vec3f(0.0);
    switch face {
        case 0u: { dir = vec3f( 1.0, -c.y, -c.x); }
        case 1u: { dir = vec3f(-1.0, -c.y,  c.x); }
        case 2u: { dir = vec3f( c.x,  1.0,  c.y); }
        case 3u: { dir = vec3f( c.x, -1.0, -c.y); }
        case 4u: { dir = vec3f( c.x, -c.y,  1.0); }
        case 5u: { dir = vec3f(-c.x, -c.y, -1.0); }
        default: {}
    }
    return normalize(dir);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let dims = textureDimensions(outputTex);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }

    let face = gid.z;
    // Map pixel coordinates to [0, 1] for face UV
    let faceUV = (vec2f(gid.xy) + 0.5) / vec2f(f32(dims.x), f32(dims.y));
    
    // Get corresponding 3D direction vector
    let dir = getDirectionForFace(faceUV, face);
    
    // Convert 3D direction to equirectangular UV
    let equiUV = directionToEquirectangularUV(dir);
    
    // Flip V: image row 0 = north pole (top), but our v=0 = south pole
    let sampleV = 1.0 - equiUV.y;
    
    // Sample with bilinear interpolation
    let color = sampleEquirectBilinear(vec2f(equiUV.x, sampleV));
    
    // Write to the cubemap face array layer
    textureStore(outputTex, vec2i(gid.xy), face, color);
}

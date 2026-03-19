// Procedural sky cubemap generation compute shader
// Writes to a 2D texture array (6 layers = cubemap faces)

struct GenParams {
    face_size: u32,
    sun_dir_x: f32,
    sun_dir_y: f32,
    sun_dir_z: f32,
}

@group(0) @binding(0) var outputTex: texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(1) var<uniform> params: GenParams;

const PI: f32 = 3.14159265359;

// Get cubemap direction from face index and UV
fn getCubeDir(face: u32, uv: vec2f) -> vec3f {
    let u = uv.x * 2.0 - 1.0;
    let v = uv.y * 2.0 - 1.0;

    switch (face) {
        case 0u: { return normalize(vec3f( 1.0, -v,  -u)); } // +X
        case 1u: { return normalize(vec3f(-1.0, -v,   u)); } // -X
        case 2u: { return normalize(vec3f(  u,  1.0,  v)); } // +Y
        case 3u: { return normalize(vec3f(  u, -1.0, -v)); } // -Y
        case 4u: { return normalize(vec3f(  u,  -v, 1.0)); } // +Z
        case 5u: { return normalize(vec3f( -u,  -v,-1.0)); } // -Z
        default: { return vec3f(0.0); }
    }
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let face = gid.z;
    let size = params.face_size;
    if (gid.x >= size || gid.y >= size || face >= 6u) { return; }

    let uv = (vec2f(f32(gid.x), f32(gid.y)) + 0.5) / f32(size);
    let dir = getCubeDir(face, uv);

    let sunDir = normalize(vec3f(params.sun_dir_x, params.sun_dir_y, params.sun_dir_z));

    // Sky gradient
    let upness = dir.y;

    // Horizon to zenith gradient
    let skyColorHorizon = vec3f(0.6, 0.75, 0.95);
    let skyColorZenith = vec3f(0.15, 0.3, 0.65);
    let groundColor = vec3f(0.25, 0.22, 0.2);

    var color: vec3f;
    if (upness > 0.0) {
        // Sky
        let t = pow(upness, 0.5);
        color = mix(skyColorHorizon, skyColorZenith, t);
    } else {
        // Ground
        let t = pow(-upness, 0.4);
        color = mix(skyColorHorizon * 0.5, groundColor, t);
    }

    // Sun disk
    let sunDot = dot(dir, sunDir);
    let sunAngularRadius = 0.02;
    let sunGlow = 0.15;
    
    // Hard sun disk
    if (sunDot > (1.0 - sunAngularRadius)) {
        let sunIntensity = 15.0;
        let sunColor = vec3f(1.0, 0.95, 0.85) * sunIntensity;
        color = sunColor;
    } else if (sunDot > (1.0 - sunGlow)) {
        // Sun glow
        let t = (sunDot - (1.0 - sunGlow)) / (sunGlow - sunAngularRadius);
        let glowColor = vec3f(1.0, 0.85, 0.6) * t * t * 2.0;
        color += glowColor;
    }

    // Subtle atmospheric scattering near horizon
    let horizonGlow = exp(-abs(upness) * 5.0) * 0.3;
    color += vec3f(1.0, 0.85, 0.7) * horizonGlow * max(dot(dir, sunDir) * 0.5 + 0.5, 0.0);

    textureStore(outputTex, vec2i(gid.xy), i32(face), vec4f(color, 1.0));
}

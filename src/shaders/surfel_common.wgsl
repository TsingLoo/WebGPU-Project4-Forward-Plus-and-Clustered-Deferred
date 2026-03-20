// surfel_common.wgsl
// This file is prepended to all surfel compute shaders.

struct Surfel {
    position: vec3f,
    radius: f32,
    normal: vec3f,
    age: f32,
    irradiance: vec3f, // Long-term MSME mean
    variance: f32,     // Short-term MSME variance
    shortMean: vec3f,  // Short-term MSME mean
    pGuide: f32,       // Guiding probability
};

struct SurfelGridConstants {
    gridMin: vec3f,
    pad0: f32,
    gridMax: vec3f,
    pad1: f32,
    cellsX: u32,
    cellsY: u32,
    cellsZ: u32,
    maxSurfelsPerCell: u32,
    maxSurfels: u32, // Total allowed
    allocatedCount: u32, // Current active count
    raysPerSurfel: u32,
    pad2: f32,
};

struct SurfelGuidanceTile {
    // 8x8 = 64 bins. Packed as 16 vec4<f32>.
    luminanceBits: array<vec4f, 16>,
};

struct SurfelDepthTile {
    // 4x4 = 16 bins. Packed as 16 vec4<f32> for Moment Shadow Mapping.
    // E[z], E[z^2], E[z^3], E[z^4]
    moments: array<vec4f, 16>,
};

// Math utilities
fn hemiOctEncode(v: vec3f) -> vec2f {
    let l1norm = abs(v.x) + abs(v.y) + abs(v.z);
    let p = v.xy * (1.0 / max(l1norm, 0.0001));
    if (v.z < 0.0) {
        let flipped = (1.0 - abs(p.yx)) * sign(p.xy);
        return flipped;
    }
    return p;
}

fn hemiOctDecode(p: vec2f) -> vec3f {
    var v = vec3f(p.xy, 1.0 - abs(p.x) - abs(p.y));
    if (v.z < 0.0) {
        v.x = (1.0 - abs(v.y)) * sign(v.x);
        v.y = (1.0 - abs(v.x)) * sign(v.y);
    }
    return normalize(v);
}

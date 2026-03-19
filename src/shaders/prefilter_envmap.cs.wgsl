// Specular prefilter environment map compute shader
// Generates mip levels with increasing roughness using importance sampling (GGX)

struct PrefilterParams {
    roughness: f32,
    face_size: u32,
    num_samples: u32,
    _pad: u32,
}

@group(0) @binding(0) var envMap: texture_cube<f32>;
@group(0) @binding(1) var envSampler: sampler;
@group(0) @binding(2) var outputTex: texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(3) var<uniform> params: PrefilterParams;

const PI: f32 = 3.14159265359;

fn getCubeDir(face: u32, uv: vec2f) -> vec3f {
    let u = uv.x * 2.0 - 1.0;
    let v = uv.y * 2.0 - 1.0;

    switch (face) {
        case 0u: { return normalize(vec3f( 1.0, -v,  -u)); }
        case 1u: { return normalize(vec3f(-1.0, -v,   u)); }
        case 2u: { return normalize(vec3f(  u,  1.0,  v)); }
        case 3u: { return normalize(vec3f(  u, -1.0, -v)); }
        case 4u: { return normalize(vec3f(  u,  -v, 1.0)); }
        case 5u: { return normalize(vec3f( -u,  -v,-1.0)); }
        default: { return vec3f(0.0); }
    }
}

// radical inverse Van der Corput
fn radicalInverseVdC(bits_in: u32) -> f32 {
    var bits = bits_in;
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return f32(bits) * 2.3283064365386963e-10; // / 0x100000000
}

fn hammersley(i: u32, N: u32) -> vec2f {
    return vec2f(f32(i) / f32(N), radicalInverseVdC(i));
}

fn importanceSampleGGX(Xi: vec2f, N: vec3f, roughness: f32) -> vec3f {
    let a = roughness * roughness;

    let phi = 2.0 * PI * Xi.x;
    let cosTheta = sqrt((1.0 - Xi.y) / (1.0 + (a * a - 1.0) * Xi.y));
    let sinTheta = sqrt(1.0 - cosTheta * cosTheta);

    // from spherical coordinates to cartesian coordinates
    let H_tangent = vec3f(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);

    // from tangent-space vector to world-space sample vector
    var up = vec3f(0.0, 1.0, 0.0);
    if (abs(N.y) > 0.99) { up = vec3f(1.0, 0.0, 0.0); }
    let tangent = normalize(cross(up, N));
    let bitangent = cross(N, tangent);

    return normalize(tangent * H_tangent.x + bitangent * H_tangent.y + N * H_tangent.z);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let face = gid.z;
    let size = params.face_size;
    if (gid.x >= size || gid.y >= size || face >= 6u) { return; }

    let uv = (vec2f(f32(gid.x), f32(gid.y)) + 0.5) / f32(size);
    let N = getCubeDir(face, uv);
    let R = N;
    let V = R;

    let roughness = params.roughness;
    let totalSamples = params.num_samples;

    var prefilteredColor = vec3f(0.0);
    var totalWeight: f32 = 0.0;

    for (var i = 0u; i < totalSamples; i += 1u) {
        let Xi = hammersley(i, totalSamples);
        let H = importanceSampleGGX(Xi, N, roughness);
        let L = normalize(2.0 * dot(V, H) * H - V);

        let NdotL = max(dot(N, L), 0.0);
        if (NdotL > 0.0) {
            var sampledColor = textureSampleLevel(envMap, envSampler, L, 0.0).rgb;
            // Clamp extreme HDR values to prevent firefly/bright spot artifacts
            let maxLuminance = 100.0;
            let luminance = dot(sampledColor, vec3f(0.2126, 0.7152, 0.0722));
            if (luminance > maxLuminance) {
                sampledColor = sampledColor * (maxLuminance / luminance);
            }
            prefilteredColor += sampledColor * NdotL;
            totalWeight += NdotL;
        }
    }

    prefilteredColor = prefilteredColor / totalWeight;
    textureStore(outputTex, vec2i(gid.xy), i32(face), vec4f(prefilteredColor, 1.0));
}

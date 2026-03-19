// DDGI Visibility (Depth) Probe Update
// Each workgroup processes one probe. Each thread handles one texel in the visibility map.
// Stores (mean_distance, mean_distance^2) for Chebyshev visibility test.

@group(0) @binding(0) var<uniform> ddgi: DDGIUniforms;
@group(0) @binding(1) var<uniform> randomRotation: mat4x4f;
@group(0) @binding(2) var<storage, read> rayData: array<vec4f>;
@group(0) @binding(3) var visibilityAtlasRead: texture_2d<f32>;
@group(0) @binding(4) var visibilityAtlasWrite: texture_storage_2d<rgba16float, write>;

const DDGI_RAYS_PER_PROBE: u32 = ${ddgiRaysPerProbe}u;
const VISIBILITY_TEXELS: u32 = ${ddgiVisibilityTexels}u;
const VISIBILITY_WITH_BORDER: u32 = VISIBILITY_TEXELS + 2u;
const GOLDEN_RATIO: f32 = 1.618033988749895;

fn fibonacciSphereDir(index: u32, total: u32) -> vec3f {
    let i = f32(index);
    let n = f32(total);
    let theta = 2.0 * PI * i / GOLDEN_RATIO;
    let phi = acos(1.0 - 2.0 * (i + 0.5) / n);
    return vec3f(
        sin(phi) * cos(theta),
        cos(phi),
        sin(phi) * sin(theta)
    );
}

@compute @workgroup_size(${ddgiVisibilityTexels}, ${ddgiVisibilityTexels}, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3u,
    @builtin(workgroup_id) wgid: vec3u
) {
    let probeIndex = i32(wgid.z);
    let totalProbes = ddgi.grid_count.w;
    if (probeIndex >= totalProbes) { return; }

    let texelX = lid.x;
    let texelY = lid.y;

    let octUV = vec2f(
        (f32(texelX) + 0.5) / f32(VISIBILITY_TEXELS),
        (f32(texelY) + 0.5) / f32(VISIBILITY_TEXELS)
    );
    let texelDir = octDecode(octUV);

    var weightedDist = 0.0;
    var weightedDist2 = 0.0;
    var totalWeight = 0.0;

    let rayBaseIdx = u32(probeIndex) * DDGI_RAYS_PER_PROBE;

    for (var r = 0u; r < DDGI_RAYS_PER_PROBE; r++) {
        let baseDir = fibonacciSphereDir(r, DDGI_RAYS_PER_PROBE);
        let rayDir = normalize((randomRotation * vec4f(baseDir, 0.0)).xyz);

        let rayResult = rayData[rayBaseIdx + r];
        let dist = rayResult.w;

        let weight = max(dot(texelDir, rayDir), 0.0);

        if (weight > 0.0) {
            weightedDist += dist * weight;
            weightedDist2 += dist * dist * weight;
            totalWeight += weight;
        }
    }

    if (totalWeight > 0.0) {
        weightedDist /= totalWeight;
        weightedDist2 /= totalWeight;
    }

    // Atlas texel coordinate
    let probesPerRow = ddgi.grid_count.x;
    let probeRow = probeIndex / probesPerRow;
    let probeCol = probeIndex % probesPerRow;

    let atlasX = probeCol * i32(VISIBILITY_WITH_BORDER) + 1 + i32(texelX);
    let atlasY = probeRow * i32(VISIBILITY_WITH_BORDER) + 1 + i32(texelY);

    // Hysteresis blending
    let prevVal = textureLoad(visibilityAtlasRead, vec2i(atlasX, atlasY), 0).rg;
    let hysteresis = ddgi.hysteresis.y;
    let blendedDist = mix(weightedDist, prevVal.x, hysteresis);
    let blendedDist2 = mix(weightedDist2, prevVal.y, hysteresis);

    textureStore(visibilityAtlasWrite, vec2i(atlasX, atlasY), vec4f(blendedDist, blendedDist2, 0.0, 1.0));
}

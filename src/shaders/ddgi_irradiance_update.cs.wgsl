// DDGI Irradiance Probe Update
// Each workgroup processes one probe. Each thread handles one texel in the octahedral irradiance map.

@group(0) @binding(0) var<uniform> ddgi: DDGIUniforms;
@group(0) @binding(1) var<uniform> randomRotation: mat4x4f;
@group(0) @binding(2) var<storage, read> rayData: array<vec4f>;
@group(0) @binding(3) var irradianceAtlasRead: texture_2d<f32>;
@group(0) @binding(4) var irradianceAtlasWrite: texture_storage_2d<rgba16float, write>;

const DDGI_RAYS_PER_PROBE: u32 = ${ddgiRaysPerProbe}u;
const IRRADIANCE_TEXELS: u32 = ${ddgiIrradianceTexels}u;
const IRRADIANCE_WITH_BORDER: u32 = IRRADIANCE_TEXELS + 2u;
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

@compute @workgroup_size(${ddgiIrradianceTexels}, ${ddgiIrradianceTexels}, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3u,
    @builtin(workgroup_id) wgid: vec3u
) {
    let probeIndex = i32(wgid.z);
    let totalProbes = ddgi.grid_count.w;
    if (probeIndex >= totalProbes) { return; }

    // Texel position within this probe's octahedral irradiance map
    let texelX = lid.x;
    let texelY = lid.y;

    // Convert texel to octahedral UV [0,1]^2
    let octUV = vec2f(
        (f32(texelX) + 0.5) / f32(IRRADIANCE_TEXELS),
        (f32(texelY) + 0.5) / f32(IRRADIANCE_TEXELS)
    );

    // Decode to world-space direction
    let texelDir = octDecode(octUV);

    // Accumulate irradiance from all rays
    var weightedIrradiance = vec3f(0.0);
    var totalWeight = 0.0;

    let rayBaseIdx = u32(probeIndex) * DDGI_RAYS_PER_PROBE;

    for (var r = 0u; r < DDGI_RAYS_PER_PROBE; r++) {
        // Reconstruct the ray direction (must match probe_trace)
        let baseDir = fibonacciSphereDir(r, DDGI_RAYS_PER_PROBE);
        let rayDir = normalize((randomRotation * vec4f(baseDir, 0.0)).xyz);

        let rayResult = rayData[rayBaseIdx + r];
        let radiance = rayResult.xyz;

        // Weight by cosine of angle between texel direction and ray direction
        let weight = max(dot(texelDir, rayDir), 0.0);

        if (weight > 0.0) {
            weightedIrradiance += radiance * weight;
            totalWeight += weight;
        }
    }

    if (totalWeight > 0.0) {
        weightedIrradiance /= totalWeight;
    }

    // Atlas texel coordinate (interior, +1 for border)
    let probesPerRow = ddgi.grid_count.x;
    let probeRow = probeIndex / probesPerRow;
    let probeCol = probeIndex % probesPerRow;

    let atlasX = probeCol * i32(IRRADIANCE_WITH_BORDER) + 1 + i32(texelX);
    let atlasY = probeRow * i32(IRRADIANCE_WITH_BORDER) + 1 + i32(texelY);

    // Hysteresis blending with previous value
    let prevColor = textureLoad(irradianceAtlasRead, vec2i(atlasX, atlasY), 0).rgb;
    let hysteresis = ddgi.hysteresis.x;
    let blended = mix(weightedIrradiance, prevColor, hysteresis);

    textureStore(irradianceAtlasWrite, vec2i(atlasX, atlasY), vec4f(blended, 1.0));
}

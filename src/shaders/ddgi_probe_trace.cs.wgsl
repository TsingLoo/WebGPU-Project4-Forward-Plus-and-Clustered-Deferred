// DDGI Probe Ray Tracing — screen-space ray march against G-buffer
// Each thread casts one ray for one probe

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<uniform> ddgi: DDGIUniforms;
@group(0) @binding(2) var<uniform> randomRotation: mat4x4f;
@group(0) @binding(3) var depthTex: texture_depth_2d;
@group(0) @binding(4) var normalTex: texture_2d<f32>;
@group(0) @binding(5) var albedoTex: texture_2d<f32>;
@group(0) @binding(6) var positionTex: texture_2d<f32>;
@group(0) @binding(7) var envMap: texture_cube<f32>;
@group(0) @binding(8) var envSampler: sampler;
@group(0) @binding(9) var<storage, read_write> rayData: array<vec4f>; // [radiance.rgb, hitDist]
@group(0) @binding(10) var<uniform> sunLight: SunLight;
@group(0) @binding(11) var vsmPhysAtlas: texture_depth_2d;
@group(0) @binding(12) var<uniform> vsmUniforms: VSMUniforms;

const DDGI_RAYS_PER_PROBE: u32 = ${ddgiRaysPerProbe}u;
const GOLDEN_RATIO: f32 = 1.618033988749895;

// Fibonacci sphere: uniform distribution on sphere
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

@compute @workgroup_size(${ddgiRaysPerProbe}, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3u,
    @builtin(workgroup_id) wgid: vec3u
) {
    let rayIndex = gid.x;
    let probeIndex = i32(wgid.y);
    let totalProbes = ddgi.grid_count.w;

    if (probeIndex >= totalProbes || rayIndex >= DDGI_RAYS_PER_PROBE) {
        return;
    }

    // Probe 3D index from linear index
    let gridX = ddgi.grid_count.x;
    let gridY = ddgi.grid_count.y;
    let pz = probeIndex / (gridX * gridY);
    let py = (probeIndex % (gridX * gridY)) / gridX;
    let px = probeIndex % gridX;
    let probeGridIdx = vec3i(px, py, pz);
    let probeWorldPos = ddgiProbePosition(probeGridIdx, ddgi);

    // Generate ray direction with random rotation for temporal jitter
    let baseDir = fibonacciSphereDir(rayIndex, DDGI_RAYS_PER_PROBE);
    let rotatedDir = normalize((randomRotation * vec4f(baseDir, 0.0)).xyz);

    // Screen-space ray march
    let screenDims = vec2f(f32(textureDimensions(albedoTex).x), f32(textureDimensions(albedoTex).y));

    // Project probe origin to view space
    let probeView = (camera.view_mat * vec4f(probeWorldPos, 1.0)).xyz;
    let rayEndView = (camera.view_mat * vec4f(probeWorldPos + rotatedDir * 50.0, 1.0)).xyz;

    // Project to clip/screen space
    let probeClip = camera.proj_mat * vec4f(probeView, 1.0);
    let endClip = camera.proj_mat * vec4f(rayEndView, 1.0);

    let probeNDC = probeClip.xy / probeClip.w;
    let endNDC = endClip.xy / endClip.w;

    let probeScreen = (probeNDC * vec2f(0.5, -0.5) + 0.5) * screenDims;
    let endScreen = (endNDC * vec2f(0.5, -0.5) + 0.5) * screenDims;

    // Ray march parameters
    let maxSteps = 64;
    let stepSize = 1.0 / f32(maxSteps);

    var hitRadiance = vec3f(0.0);
    var hitDist = -1.0; // -1 means miss

    // Only march if the probe is in front of the camera
    if (probeClip.w > 0.0) {
        for (var step = 1; step <= maxSteps; step++) {
            let t = f32(step) * stepSize;
            let sampleScreen = mix(probeScreen, endScreen, t);
            let ssi = vec2i(sampleScreen);

            // Bounds check
            if (ssi.x < 0 || ssi.y < 0 || ssi.x >= i32(screenDims.x) || ssi.y >= i32(screenDims.y)) {
                continue;
            }

            // Sample depth
            let depth = textureLoad(depthTex, ssi, 0);
            if (depth >= 1.0) {
                continue; // sky
            }

            // Reconstruct the position at this screen pixel
            let hitPosWorld = textureLoad(positionTex, ssi, 0).xyz;
            let toHit = hitPosWorld - probeWorldPos;
            let hitDistCandidate = length(toHit);

            // Check if the ray direction matches (dot product with direction to hit)
            let dirToHit = normalize(toHit);
            let alignment = dot(dirToHit, rotatedDir);

            if (alignment > 0.3 && hitDistCandidate > 0.1) {
                // Read surface data
                let hitAlbedo = textureLoad(albedoTex, ssi, 0).rgb;
                let hitNormal = normalize(textureLoad(normalTex, ssi, 0).xyz);

                // Compute lighting at the hit point for proper bounce radiance
                var hitLighting = vec3f(0.0);

                // Sun light contribution at hit surface (with shadow)
                if (sunLight.color.a > 0.5) {
                    let hitPos = textureLoad(positionTex, ssi, 0).xyz;
                    let sunShadow = calculateShadowVSMSimple(vsmPhysAtlas, vsmUniforms, sunLight, hitPos, hitNormal);
                    let sunL = normalize(sunLight.direction.xyz);
                    let sunNdotL = max(dot(hitNormal, sunL), 0.0);
                    hitLighting += sunLight.color.rgb * sunLight.direction.w * sunNdotL * sunShadow;
                }

                // Add ambient term so shadowed areas still contribute some bounce
                hitLighting += vec3f(ddgi.ddgi_enabled.z);

                hitRadiance = hitAlbedo * hitLighting;
                hitDist = hitDistCandidate;
                break;
            }
        }
    }

    // If miss, sample environment sky
    if (hitDist < 0.0) {
        hitRadiance = textureSampleLevel(envMap, envSampler, rotatedDir, 0.0).rgb;
        // Clamp extreme HDR values (sun disk = 15.0) but allow normal sky brightness through
        hitRadiance = min(hitRadiance, vec3f(3.0));
        hitDist = 1000.0; // far
    }

    // Store ray result
    let outputIdx = u32(probeIndex) * DDGI_RAYS_PER_PROBE + rayIndex;
    rayData[outputIdx] = vec4f(hitRadiance, hitDist);
}

// DDGI Probe Ray Tracing — World Space Coarse Scene Voxel DDA
// Each thread casts one ray for one probe

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<uniform> ddgi: DDGIUniforms;
@group(0) @binding(2) var<uniform> randomRotation: mat4x4f;
@group(0) @binding(3) var voxelGrid: texture_3d<f32>;
@group(0) @binding(4) var envMap: texture_cube<f32>;
@group(0) @binding(5) var envSampler: sampler;
@group(0) @binding(6) var<storage, read_write> rayData: array<vec4f>; // [radiance.rgb, hitDist]
@group(0) @binding(7) var<uniform> sunLight: SunLight;
@group(0) @binding(8) var vsmPhysAtlas: texture_depth_2d;
@group(0) @binding(9) var<uniform> vsmUniforms: VSMUniforms;

const DDGI_RAYS_PER_PROBE: u32 = ${ddgiRaysPerProbe}u;
const GOLDEN_RATIO: f32 = 1.618033988749895;

fn fibonacciSphereDir(index: u32, total: u32) -> vec3f {
    let i = f32(index);
    let n = f32(total);
    let theta = 2.0 * PI * i / GOLDEN_RATIO;
    let phi = acos(1.0 - 2.0 * (i + 0.5) / n);
    return vec3f(sin(phi) * cos(theta), cos(phi), sin(phi) * sin(theta));
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

    let gridX = ddgi.grid_count.x;
    let gridY = ddgi.grid_count.y;
    let pz = probeIndex / (gridX * gridY);
    let py = (probeIndex % (gridX * gridY)) / gridX;
    let px = probeIndex % gridX;
    let probeWorldPos = ddgiProbePosition(vec3i(px, py, pz), ddgi);

    let baseDir = fibonacciSphereDir(rayIndex, DDGI_RAYS_PER_PROBE);
    let rotatedDir = normalize((randomRotation * vec4f(baseDir, 0.0)).xyz);

    var hitRadiance = vec3f(0.0);
    var hitDist = -1.0; 

    // Scene Voxel Bounds (matching scene.ts limits)
    let vMin = vec3f(-15.0, 0.0, -10.0);
    let vMax = vec3f(15.0, 15.0, 10.0);
    let textureDims = vec3f(128.0, 128.0, 128.0);
    let vExtent = vMax - vMin;

    // AABB Intersection to narrow ray bounds
    var tMin = 0.0;
    var tMax = 100.0;
    for (var j = 0; j < 3; j++) {
        if (abs(rotatedDir[j]) > 0.0001) {
            let invD = 1.0 / rotatedDir[j];
            let t0 = (vMin[j] - probeWorldPos[j]) * invD;
            let t1 = (vMax[j] - probeWorldPos[j]) * invD;
            tMin = max(tMin, min(t0, t1));
            tMax = min(tMax, max(t0, t1));
        } else {
            if (probeWorldPos[j] < vMin[j] || probeWorldPos[j] > vMax[j]) {
                tMax = -1.0; // miss completely
            }
        }
    }

    if (tMax >= tMin && tMax > 0.0) {
        let maxRayDist = min(tMax, 100.0);
        let rayStart = max(0.0, tMin) + 0.1; // push slightly over probe epsilon
        let stepSize = min(min(vExtent.x, vExtent.y), vExtent.z) / 256.0; // half voxel
        
        var t = rayStart;
        while (t <= maxRayDist) {
            let pos = probeWorldPos + rotatedDir * t;
            let uvw = (pos - vMin) / vExtent;
            let coord = vec3i(uvw * textureDims);
            
            // Bounds check just to be safe
            if (coord.x >= 0 && coord.y >= 0 && coord.z >= 0 && coord.x < 128 && coord.y < 128 && coord.z < 128) {
                let voxel = textureLoad(voxelGrid, coord, 0);
                if (voxel.a > 0.5) {
                    hitDist = t;
                    
                    let normal = normalize(voxel.rgb * 2.0 - 1.0);
                    var hitLighting = vec3f(0.0);
                    
                    // Direct Sun evaluation
                    if (sunLight.color.a > 0.5) {
                        let sunShadow = calculateShadowVSMSimple(vsmPhysAtlas, vsmUniforms, sunLight, pos, normal);
                        let sunL = normalize(sunLight.direction.xyz);
                        let sunNdotL = max(dot(normal, sunL), 0.0);
                        hitLighting += sunLight.color.rgb * sunLight.direction.w * sunNdotL * sunShadow;
                    }
                    
                    hitLighting += vec3f(ddgi.ddgi_enabled.z); // minimal ambient
                    hitRadiance = vec3f(0.5) * hitLighting; // diffuse 50% albedo proxy
                    break;
                }
            }
            t += stepSize;
        }
    }

    if (hitDist < 0.0) {
        hitRadiance = textureSampleLevel(envMap, envSampler, rotatedDir, 0.0).rgb;
        hitRadiance = min(hitRadiance, vec3f(3.0));
        hitDist = 1000.0; // sky
    }

    let outputIdx = u32(probeIndex) * DDGI_RAYS_PER_PROBE + rayIndex;
    rayData[outputIdx] = vec4f(hitRadiance, hitDist);
}

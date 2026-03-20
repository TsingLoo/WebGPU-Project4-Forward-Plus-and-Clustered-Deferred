// ============================
// NRC Training Sample Scatter — Generate training data from G-buffer
// ============================
// Reads the G-buffer at subsampled positions and computes target radiance
// from direct lighting (sun + ambient). Writes (input_features, target) pairs
// to the training buffer.

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<uniform> nrc: NRCUniforms;
@group(0) @binding(2) var depthTex: texture_depth_2d;
@group(0) @binding(3) var normalTex: texture_2d<f32>;
@group(0) @binding(4) var albedoTex: texture_2d<f32>;
@group(0) @binding(5) var positionTex: texture_2d<f32>;
@group(0) @binding(6) var<uniform> sunLight: SunLight;
@group(0) @binding(7) var vsmPhysAtlas: texture_depth_2d;
@group(0) @binding(8) var<uniform> vsmUniforms: VSMUniforms;
@group(0) @binding(9) var<storage, read_write> trainingSamples: array<f32>;
@group(0) @binding(10) var<storage, read_write> sampleCounter: atomic<u32>;
@group(0) @binding(11) var envMap: texture_cube<f32>;
@group(0) @binding(12) var envSampler: sampler;

const MAX_TRAINING_SAMPLES: u32 = ${nrcMaxTrainingSamples}u;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let strideX = u32(nrc.screen_dims.z);
    let strideY = u32(nrc.screen_dims.w);
    let screenW = u32(nrc.screen_dims.x);
    let screenH = u32(nrc.screen_dims.y);

    // Simple integer hash for temporal frame jitter
    let frameCount = u32(nrc.params.w);
    let hash = frameCount * 1664525u + 1013904223u;
    let jitterX = hash % strideX;
    let jitterY = (hash / strideX) % strideY;

    // Map thread to subsampled pixel position with temporal jitter
    let pixelX = gid.x * strideX + jitterX;
    let pixelY = gid.y * strideY + jitterY;

    if (pixelX >= screenW || pixelY >= screenH) {
        return;
    }

    let ssi = vec2i(i32(pixelX), i32(pixelY));

    // Skip sky pixels
    let depth = textureLoad(depthTex, ssi, 0);
    if (depth >= 1.0) {
        return;
    }

    // Read G-buffer
    let posWorld = textureLoad(positionTex, ssi, 0).xyz;
    let normal = normalize(textureLoad(normalTex, ssi, 0).xyz);
    let albedo = textureLoad(albedoTex, ssi, 0).rgb;

    // View direction
    let viewDir = normalize(camera.camera_pos.xyz - posWorld);

    // ---- Compute target radiance (what the network should predict) ----
    // This is the Total outgoing radiance at this surface point

    var targetRadiance = vec3f(0.0);

    // Sun light contribution with shadow
    if (sunLight.color.a > 0.5) {
        let sunShadow = calculateShadowVSMSimple(vsmPhysAtlas, vsmUniforms, sunLight, posWorld, normal);
        let sunL = normalize(sunLight.direction.xyz);
        let sunNdotL = max(dot(normal, sunL), 0.0);
        let sunContrib = sunLight.color.rgb * sunLight.direction.w * sunNdotL * sunShadow;
        targetRadiance += sunContrib / PI;
    }

    // Tone map target to [0,1] so sigmoid output can represent it
    // Using Reinhard: x / (x + 1)
    targetRadiance = targetRadiance / (targetRadiance + vec3f(1.0));
    
    // Prevent bad samples from poisoning the training buffer
    if (targetRadiance.x != targetRadiance.x || targetRadiance.y != targetRadiance.y || targetRadiance.z != targetRadiance.z) {
        targetRadiance = vec3f(0.0);
    }

    // Early out for sky pixels so we don't dilute the training dataset with zeros
    if (depth >= 1.0) {
        return;
    }

    // ---- Allocate sample slot ----
    let sampleIdx = atomicAdd(&sampleCounter, 1u);
    if (sampleIdx >= MAX_TRAINING_SAMPLES) {
        return;
    }

    // ---- Encode features ----
    let features = nrcEncodeInput(posWorld, normal, viewDir, nrc.scene_min.xyz, nrc.scene_max.xyz);

    // ---- Write to training buffer ----
    let baseOffset = sampleIdx * NRC_SAMPLE_STRIDE;

    // Input features [0..14]
    for (var i = 0u; i < 15u; i++) {
        trainingSamples[baseOffset + i] = features[i];
    }

    // Target radiance [15..17]
    trainingSamples[baseOffset + 15u] = targetRadiance.x;
    trainingSamples[baseOffset + 16u] = targetRadiance.y;
    trainingSamples[baseOffset + 17u] = targetRadiance.z;

    // Sample weight [18]
    trainingSamples[baseOffset + 18u] = 1.0;

    // Pad [19]
    trainingSamples[baseOffset + 19u] = 0.0;
}

// ============================
// Neural Radiance Caching — Common Definitions
// ============================
// MLP architecture: 4 layers fully-connected
//   Input (15D) → [32] → [64] → [64] → [3] (RGB radiance)
//   Activations: ReLU (hidden), Sigmoid (output)
//
// Input encoding (15D):
//   [0..2]  = position (normalized to scene bbox)
//   [3..5]  = surface normal
//   [6..8]  = view direction
//   [9..14] = frequency encoding of position (sin/cos at 3 frequencies)
//
// Weight buffer layout (contiguous f32 array):
//   Layer 0: weights[15×32] + bias[32]  = 480 + 32  = 512
//   Layer 1: weights[32×64] + bias[64]  = 2048 + 64 = 2112
//   Layer 2: weights[64×64] + bias[64]  = 4096 + 64 = 4160
//   Layer 3: weights[64×3]  + bias[3]   = 192 + 3   = 195
//   Total = 6979 floats → round up to 6980 (aligned)

const NRC_INPUT_DIM: u32 = 15u;
const NRC_LAYER0_OUT: u32 = 32u;
const NRC_LAYER1_OUT: u32 = 64u;
const NRC_LAYER2_OUT: u32 = 64u;
const NRC_OUTPUT_DIM: u32 = 3u;

const NRC_L0_W_OFFSET: u32 = 0u;                      // 15*32 = 480
const NRC_L0_B_OFFSET: u32 = 480u;                     // 32
const NRC_L1_W_OFFSET: u32 = 512u;                     // 32*64 = 2048
const NRC_L1_B_OFFSET: u32 = 2560u;                    // 64
const NRC_L2_W_OFFSET: u32 = 2624u;                    // 64*64 = 4096
const NRC_L2_B_OFFSET: u32 = 6720u;                    // 64
const NRC_L3_W_OFFSET: u32 = 6784u;                    // 64*3 = 192
const NRC_L3_B_OFFSET: u32 = 6976u;                    // 3
const NRC_TOTAL_PARAMS: u32 = 6980u;

// Training sample struct: input features + target radiance
// Packed as: [input: 15 floats][target: 3 floats][weight: 1 float][pad: 1 float] = 20 floats
const NRC_SAMPLE_STRIDE: u32 = 20u;

// NRCUniforms struct is defined in common.wgsl (prepended to all NRC shaders)

// ============================
// Feature Encoding Helpers
// ============================

// Normalize a world-space position to [0, 1] within the scene bounding box
fn nrcNormalizePosition(pos: vec3f, sceneMin: vec3f, sceneMax: vec3f) -> vec3f {
    let range = sceneMax - sceneMin;
    let safeRange = max(range, vec3f(0.001));
    return clamp((pos - sceneMin) / safeRange, vec3f(0.0), vec3f(1.0));
}

// Encode input features (15D) from position, normal, view direction
fn nrcEncodeInput(pos: vec3f, normal: vec3f, viewDir: vec3f, sceneMin: vec3f, sceneMax: vec3f) -> array<f32, 15> {
    let normPos = nrcNormalizePosition(pos, sceneMin, sceneMax);
    var features: array<f32, 15>;

    // Raw position (normalized)
    features[0] = normPos.x;
    features[1] = normPos.y;
    features[2] = normPos.z;

    // Surface normal
    features[3] = normal.x;
    features[4] = normal.y;
    features[5] = normal.z;

    // View direction
    features[6] = viewDir.x;
    features[7] = viewDir.y;
    features[8] = viewDir.z;

    // Frequency encoding of position (symmetric across all axes)
    // Using a moderate frequency to capture spatial variations without excessive high-freq noise
    let freq = 2.0;
    features[9]  = sin(normPos.x * PI * freq);
    features[10] = sin(normPos.y * PI * freq);
    features[11] = sin(normPos.z * PI * freq);
    features[12] = cos(normPos.x * PI * freq);
    features[13] = cos(normPos.y * PI * freq);
    features[14] = cos(normPos.z * PI * freq);

    return features;
}

// ============================
// MLP Forward Pass (read-only weights)
// ============================
// Returns predicted RGB radiance in [0, 1] range

fn nrcForward(input: array<f32, 15>, weights: ptr<storage, array<f32>, read>) -> vec3f {
    // Layer 0: input[15] → hidden0[32], ReLU
    var h0: array<f32, 32>;
    for (var j = 0u; j < NRC_LAYER0_OUT; j++) {
        var sum = (*weights)[NRC_L0_B_OFFSET + j];
        for (var i = 0u; i < NRC_INPUT_DIM; i++) {
            sum += input[i] * (*weights)[NRC_L0_W_OFFSET + i * NRC_LAYER0_OUT + j];
        }
        h0[j] = max(sum, 0.0); // ReLU
    }

    // Layer 1: hidden0[32] → hidden1[64], ReLU
    var h1: array<f32, 64>;
    for (var j = 0u; j < NRC_LAYER1_OUT; j++) {
        var sum = (*weights)[NRC_L1_B_OFFSET + j];
        for (var i = 0u; i < NRC_LAYER0_OUT; i++) {
            sum += h0[i] * (*weights)[NRC_L1_W_OFFSET + i * NRC_LAYER1_OUT + j];
        }
        h1[j] = max(sum, 0.0); // ReLU
    }

    // Layer 2: hidden1[64] → hidden2[64], ReLU
    var h2: array<f32, 64>;
    for (var j = 0u; j < NRC_LAYER2_OUT; j++) {
        var sum = (*weights)[NRC_L2_B_OFFSET + j];
        for (var i = 0u; i < NRC_LAYER1_OUT; i++) {
            sum += h1[i] * (*weights)[NRC_L2_W_OFFSET + i * NRC_LAYER2_OUT + j];
        }
        h2[j] = max(sum, 0.0); // ReLU
    }

    // Layer 3: hidden2[64] → output[3], Sigmoid
    var out = vec3f(0.0);
    for (var c = 0u; c < NRC_OUTPUT_DIM; c++) {
        var sum = (*weights)[NRC_L3_B_OFFSET + c];
        for (var i = 0u; i < NRC_LAYER2_OUT; i++) {
            sum += h2[i] * (*weights)[NRC_L3_W_OFFSET + i * NRC_OUTPUT_DIM + c];
        }
        // Sigmoid activation
        let s = 1.0 / (1.0 + exp(-clamp(sum, -10.0, 10.0)));
        if (c == 0u) { out.x = s; }
        else if (c == 1u) { out.y = s; }
        else { out.z = s; }
    }

    return out;
}

// ============================
// NRC Train — MLP Forward + Backprop + SGD Weight Update
// ============================
// Each workgroup processes one training sample.
// Performs forward pass, computes MSE loss gradient, backpropagates,
// and atomically accumulates weight gradients.
//
// A separate lightweight pass applies the accumulated gradients to weights.
// For simplicity, this shader does both in a single dispatch:
// each thread handles a subset of parameters.

@group(0) @binding(0) var<uniform> nrc: NRCUniforms;
@group(0) @binding(1) var<storage, read_write> weights: array<f32>;     // MLP weights + biases
@group(0) @binding(2) var<storage, read> trainingSamples: array<f32>; // training data
@group(0) @binding(3) var<storage, read_write> gradAccum: array<f32>;  // gradient accumulator (same size as weights)
@group(0) @binding(4) var<storage, read_write> momentum: array<f32>;   // momentum buffer (same size as weights)

const WG_SIZE: u32 = 64u;

// ---- Shared workgroup memory ----
var<workgroup> wg_input: array<f32, 15>;
var<workgroup> wg_target: array<f32, 3>;
var<workgroup> wg_h0: array<f32, 32>;       // layer 0 output (pre-ReLU stored as post-ReLU)
var<workgroup> wg_h0_pre: array<f32, 32>;   // layer 0 pre-activation
var<workgroup> wg_h1: array<f32, 64>;       // layer 1 output
var<workgroup> wg_h1_pre: array<f32, 64>;
var<workgroup> wg_h2: array<f32, 64>;       // layer 2 output
var<workgroup> wg_h2_pre: array<f32, 64>;
var<workgroup> wg_out: array<f32, 3>;       // network output (post-sigmoid)

// Gradients w.r.t. layer outputs
var<workgroup> wg_dh0: array<f32, 32>;
var<workgroup> wg_dh1: array<f32, 64>;
var<workgroup> wg_dh2: array<f32, 64>;
var<workgroup> wg_dout: array<f32, 3>;

@compute @workgroup_size(WG_SIZE, 1, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3u
) {
    let threadIdx = lid.x;
    let numSamples = u32(nrc.params.y);
    let lr = nrc.params.x;
    let momentumDecay = nrc.params.z;

    // Initialize gradient accumulator to 0
    var cIdx = threadIdx;
    while (cIdx < NRC_TOTAL_PARAMS) {
        gradAccum[cIdx] = 0.0;
        cIdx += WG_SIZE;
    }
    workgroupBarrier();

    for (var sampleIdx = 0u; sampleIdx < numSamples; sampleIdx++) {
        let baseOffset = sampleIdx * NRC_SAMPLE_STRIDE;

    // ---- Step 1: Load sample into shared memory ----
    if (threadIdx < 15u) {
        wg_input[threadIdx] = trainingSamples[baseOffset + threadIdx];
    }
    if (threadIdx < 3u) {
        wg_target[threadIdx] = trainingSamples[baseOffset + 15u + threadIdx];
    }
    workgroupBarrier();

    // ---- Step 2: Forward Pass ----

    // Layer 0: input[15] → h0[32], ReLU
    if (threadIdx < NRC_LAYER0_OUT) {
        var sum = weights[NRC_L0_B_OFFSET + threadIdx];
        for (var i = 0u; i < NRC_INPUT_DIM; i++) {
            sum += wg_input[i] * weights[NRC_L0_W_OFFSET + i * NRC_LAYER0_OUT + threadIdx];
        }
        wg_h0_pre[threadIdx] = sum;
        wg_h0[threadIdx] = max(sum, 0.0);
    }
    workgroupBarrier();

    // Layer 1: h0[32] → h1[64], ReLU
    if (threadIdx < NRC_LAYER1_OUT) {
        var sum = weights[NRC_L1_B_OFFSET + threadIdx];
        for (var i = 0u; i < NRC_LAYER0_OUT; i++) {
            sum += wg_h0[i] * weights[NRC_L1_W_OFFSET + i * NRC_LAYER1_OUT + threadIdx];
        }
        wg_h1_pre[threadIdx] = sum;
        wg_h1[threadIdx] = max(sum, 0.0);
    }
    workgroupBarrier();

    // Layer 2: h1[64] → h2[64], ReLU
    if (threadIdx < NRC_LAYER2_OUT) {
        var sum = weights[NRC_L2_B_OFFSET + threadIdx];
        for (var i = 0u; i < NRC_LAYER1_OUT; i++) {
            sum += wg_h1[i] * weights[NRC_L2_W_OFFSET + i * NRC_LAYER2_OUT + threadIdx];
        }
        wg_h2_pre[threadIdx] = sum;
        wg_h2[threadIdx] = max(sum, 0.0);
    }
    workgroupBarrier();

    // Layer 3: h2[64] → out[3], Sigmoid
    if (threadIdx < NRC_OUTPUT_DIM) {
        var sum = weights[NRC_L3_B_OFFSET + threadIdx];
        for (var i = 0u; i < NRC_LAYER2_OUT; i++) {
            sum += wg_h2[i] * weights[NRC_L3_W_OFFSET + i * NRC_OUTPUT_DIM + threadIdx];
        }
        let sig = 1.0 / (1.0 + exp(-clamp(sum, -10.0, 10.0)));
        wg_out[threadIdx] = sig;
    }
    workgroupBarrier();

    // ---- Step 3: Compute Loss Gradient ----
    // MSE loss = 0.5 * sum((out - target)^2)
    // d_loss/d_out = (out - target)
    if (threadIdx < NRC_OUTPUT_DIM) {
        let o = wg_out[threadIdx];
        let t = wg_target[threadIdx];
        // Gradient of MSE * sigmoid derivative: (o - t) * o * (1 - o)
        wg_dout[threadIdx] = (o - t) * o * (1.0 - o);
    }
    workgroupBarrier();

    // ---- Step 4: Backpropagation ----

    // Layer 3 backward: dh2 from dout
    if (threadIdx < NRC_LAYER2_OUT) {
        var grad = 0.0;
        for (var c = 0u; c < NRC_OUTPUT_DIM; c++) {
            grad += wg_dout[c] * weights[NRC_L3_W_OFFSET + threadIdx * NRC_OUTPUT_DIM + c];
        }
        // ReLU derivative
        if (wg_h2_pre[threadIdx] <= 0.0) { grad = 0.0; }
        wg_dh2[threadIdx] = grad;
    }
    workgroupBarrier();

    // Layer 2 backward: dh1 from dh2
    if (threadIdx < NRC_LAYER1_OUT) {
        var grad = 0.0;
        for (var j = 0u; j < NRC_LAYER2_OUT; j++) {
            grad += wg_dh2[j] * weights[NRC_L2_W_OFFSET + threadIdx * NRC_LAYER2_OUT + j];
        }
        if (wg_h1_pre[threadIdx] <= 0.0) { grad = 0.0; }
        wg_dh1[threadIdx] = grad;
    }
    workgroupBarrier();

    // Layer 1 backward: dh0 from dh1
    if (threadIdx < NRC_LAYER0_OUT) {
        var grad = 0.0;
        for (var j = 0u; j < NRC_LAYER1_OUT; j++) {
            grad += wg_dh1[j] * weights[NRC_L1_W_OFFSET + threadIdx * NRC_LAYER1_OUT + j];
        }
        if (wg_h0_pre[threadIdx] <= 0.0) { grad = 0.0; }
        wg_dh0[threadIdx] = grad;
    }
    workgroupBarrier();

    // ---- Step 5: Accumulate Weight Gradients in Storage Buffer ----
    // Each thread accumulates gradients for a subset of parameters into gradAccum

    // Layer 3 weights
    {
        var idx = threadIdx;
        while (idx < NRC_LAYER2_OUT * NRC_OUTPUT_DIM) {
            let i = idx / NRC_OUTPUT_DIM;
            let j = idx % NRC_OUTPUT_DIM;
            let grad = wg_h2[i] * wg_dout[j];
            gradAccum[NRC_L3_W_OFFSET + idx] += grad;
            idx += WG_SIZE;
        }
    }
    if (threadIdx < NRC_OUTPUT_DIM) {
        gradAccum[NRC_L3_B_OFFSET + threadIdx] += wg_dout[threadIdx];
    }
    workgroupBarrier();

    // Layer 2 weights
    {
        var idx = threadIdx;
        while (idx < NRC_LAYER1_OUT * NRC_LAYER2_OUT) {
            let i = idx / NRC_LAYER2_OUT;
            let j = idx % NRC_LAYER2_OUT;
            let grad = wg_h1[i] * wg_dh2[j];
            gradAccum[NRC_L2_W_OFFSET + idx] += grad;
            idx += WG_SIZE;
        }
    }
    if (threadIdx < NRC_LAYER2_OUT) {
        gradAccum[NRC_L2_B_OFFSET + threadIdx] += wg_dh2[threadIdx];
    }
    workgroupBarrier();

    // Layer 1 weights
    {
        var idx = threadIdx;
        while (idx < NRC_LAYER0_OUT * NRC_LAYER1_OUT) {
            let i = idx / NRC_LAYER1_OUT;
            let j = idx % NRC_LAYER1_OUT;
            let grad = wg_h0[i] * wg_dh1[j];
            gradAccum[NRC_L1_W_OFFSET + idx] += grad;
            idx += WG_SIZE;
        }
    }
    if (threadIdx < NRC_LAYER1_OUT) {
        gradAccum[NRC_L1_B_OFFSET + threadIdx] += wg_dh1[threadIdx];
    }
    workgroupBarrier();

    // Layer 0 weights
    {
        var idx = threadIdx;
        while (idx < NRC_INPUT_DIM * NRC_LAYER0_OUT) {
            let i = idx / NRC_LAYER0_OUT;
            let j = idx % NRC_LAYER0_OUT;
            let grad = wg_input[i] * wg_dh0[j];
            gradAccum[NRC_L0_W_OFFSET + idx] += grad;
            idx += WG_SIZE;
        }
    }
    if (threadIdx < NRC_LAYER0_OUT) {
        gradAccum[NRC_L0_B_OFFSET + threadIdx] += wg_dh0[threadIdx];
    }
    workgroupBarrier(); // Sync at end of sample iteration
    } // End of sample loop

    // ---- Step 6: Apply Global Update ----
    // Apply average gradient over the batch + momentum
    var uIdx = threadIdx;
    while (uIdx < NRC_TOTAL_PARAMS) {
        let avgGrad = gradAccum[uIdx] / max(f32(numSamples), 1.0);
        
        // Clip gradient to prevent exploding updates
        let clippedGrad = clamp(avgGrad, -1.0, 1.0);
        
        let m = momentumDecay * momentum[uIdx] + clippedGrad;
        momentum[uIdx] = m;
        
        // Update weights and clip them to prevent NaNs or infinite growth
        let newWeight = weights[uIdx] - lr * m;
        weights[uIdx] = clamp(newWeight, -100.0, 100.0);
        
        uIdx += WG_SIZE;
    }
}

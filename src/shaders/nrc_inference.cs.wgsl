// ============================
// NRC Inference — Full-screen MLP evaluation
// ============================
// For every visible pixel, reads G-buffer, encodes features,
// evaluates the trained MLP, and writes predicted indirect radiance
// to an output texture.

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<uniform> nrc: NRCUniforms;
@group(0) @binding(2) var depthTex: texture_depth_2d;
@group(0) @binding(3) var normalTex: texture_2d<f32>;
@group(0) @binding(4) var positionTex: texture_2d<f32>;
@group(0) @binding(5) var<storage, read> weights: array<f32>;
@group(0) @binding(6) var outputTex: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let screenW = u32(nrc.screen_dims.x);
    let screenH = u32(nrc.screen_dims.y);

    if (gid.x >= screenW || gid.y >= screenH) {
        return;
    }

    let ssi = vec2i(i32(gid.x), i32(gid.y));

    // Skip sky pixels — output black
    let depth = textureLoad(depthTex, ssi, 0);
    if (depth >= 1.0) {
        textureStore(outputTex, ssi, vec4f(0.0, 0.0, 0.0, 0.0));
        return;
    }

    // Read G-buffer
    let posWorld = textureLoad(positionTex, ssi, 0).xyz;
    let normal = normalize(textureLoad(normalTex, ssi, 0).xyz);
    let viewDir = normalize(camera.camera_pos.xyz - posWorld);

    // Encode input features
    let features = nrcEncodeInput(posWorld, normal, viewDir, nrc.scene_min.xyz, nrc.scene_max.xyz);

    // Run MLP forward pass
    let predicted = nrcForward(features, &weights);

    // Inverse Reinhard to recover HDR: x / (1 - x)
    // We clamp predicted to [0.0, 0.95] to prevent unconstrained oscillations 
    // from blowing up to infinity (0.95 -> HDR 19.0 max).
    let clampedPred = clamp(predicted, vec3f(0.0), vec3f(0.95));
    let hdrPredicted = clampedPred / max(vec3f(1.0) - clampedPred, vec3f(0.001));

    // Write to output texture
    textureStore(outputTex, ssi, vec4f(hdrPredicted, 1.0));
}

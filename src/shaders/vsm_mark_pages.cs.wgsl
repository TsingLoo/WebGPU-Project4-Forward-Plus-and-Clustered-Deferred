// VSM Page Marking Compute Shader
// Reads depth buffer, reconstructs world position, projects into clipmap levels,
// and marks the needed virtual pages.

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var depthTex: texture_depth_2d;
@group(0) @binding(2) var<storage, read_write> pageRequestFlags: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> vsmUniforms: VSMUniforms;

// Reconstruct world position from depth buffer
fn reconstructWorldPos(fragCoord: vec2u, depth: f32) -> vec3f {
    let screenSize = vec2f(f32(textureDimensions(depthTex).x), f32(textureDimensions(depthTex).y));
    // fragCoord to NDC: [0, width] -> [-1, 1], [0, height] -> [1, -1] (Y flipped)
    let ndc = vec2f(
        (f32(fragCoord.x) + 0.5) / screenSize.x * 2.0 - 1.0,
        1.0 - (f32(fragCoord.y) + 0.5) / screenSize.y * 2.0
    );

    // Unproject using inverse view-projection
    let clipPos = vec4f(ndc.x, ndc.y, depth, 1.0);
    let worldPos4 = vsmUniforms.inv_view_proj * clipPos;
    return worldPos4.xyz / worldPos4.w;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let texSize = textureDimensions(depthTex);
    if (gid.x >= texSize.x || gid.y >= texSize.y) {
        return;
    }

    let depth = textureLoad(depthTex, vec2i(gid.xy), 0);

    // Skip skybox pixels (depth = 1.0 in WebGPU)
    if (depth >= 1.0) {
        return;
    }

    let worldPos = reconstructWorldPos(gid.xy, depth);

    // Mark pages for the finest applicable clipmap level
    let numLevels = vsmUniforms.clipmap_count;
    for (var level = 0u; level < numLevels; level++) {
        let lightClip = vsmUniforms.clipmap_vp[level] * vec4f(worldPos, 1.0);
        let lightNDC = lightClip.xyz / lightClip.w;

        // Check if within NDC bounds [-1, 1]
        if (lightNDC.x < -1.0 || lightNDC.x > 1.0 ||
            lightNDC.y < -1.0 || lightNDC.y > 1.0 ||
            lightNDC.z < 0.0  || lightNDC.z > 1.0) {
            continue;
        }

        // NDC to UV [0, 1]
        let uv = vec2f(lightNDC.x * 0.5 + 0.5, -lightNDC.y * 0.5 + 0.5);

        // Compute virtual page coordinate
        let pagesPerAxis = vsmUniforms.pages_per_axis;
        let pageX = min(u32(uv.x * f32(pagesPerAxis)), pagesPerAxis - 1u);
        let pageY = min(u32(uv.y * f32(pagesPerAxis)), pagesPerAxis - 1u);

        // Linear index into page request buffer
        let pageIdx = level * pagesPerAxis * pagesPerAxis + pageY * pagesPerAxis + pageX;
        atomicMax(&pageRequestFlags[pageIdx], 1u);

        // Only mark the finest level that contains this pixel
        break;
    }
}

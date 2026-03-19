// DDGI Border Copy
// Copies interior edge texels to the 1px border for seamless bilinear sampling.
// One dispatch for irradiance atlas, one for visibility atlas.
// Each thread handles one border texel.

@group(0) @binding(0) var sourceAtlas: texture_2d<f32>;
@group(0) @binding(1) var destAtlas: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<uniform> params: vec4u; // (texelDim, texelDimWithBorder, probesPerRow, totalProbes)

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let texelDim = params.x;         // 8 or 16
    let texelDimBorder = params.y;   // 10 or 18
    let probesPerRow = params.z;
    let totalProbes = params.w;

    // Total border texels per probe = perimeter of (texelDim+2) square minus corners
    // = 4 * texelDim + 4 (the full border ring)
    let borderTexelsPerProbe = 4u * texelDim + 4u;
    let totalBorderTexels = borderTexelsPerProbe * totalProbes;

    let globalIdx = gid.x;
    if (globalIdx >= totalBorderTexels) { return; }

    let probeIdx = globalIdx / borderTexelsPerProbe;
    let borderIdx = globalIdx % borderTexelsPerProbe;

    let probeRow = probeIdx / probesPerRow;
    let probeCol = probeIdx % probesPerRow;

    // Probe top-left in atlas (including border)
    let probeOriginX = i32(probeCol * texelDimBorder);
    let probeOriginY = i32(probeRow * texelDimBorder);

    // Map border index to border texel (x, y) within the (texelDim+2)x(texelDim+2) block
    // and the corresponding interior source texel
    var borderX: i32;
    var borderY: i32;
    var srcX: i32;
    var srcY: i32;

    let td = i32(texelDim);
    let tdb = i32(texelDimBorder);

    if (borderIdx < texelDim + 2u) {
        // Top row (y=0)
        borderX = i32(borderIdx);
        borderY = 0;
        // Mirror: copy from row 1, with column mirrored for octahedral
        srcX = clamp(tdb - 1 - borderX, 1, td);
        srcY = 1;
    } else if (borderIdx < 2u * (texelDim + 2u)) {
        // Bottom row (y=texelDim+1)
        let localIdx = borderIdx - (texelDim + 2u);
        borderX = i32(localIdx);
        borderY = td + 1;
        srcX = clamp(tdb - 1 - borderX, 1, td);
        srcY = td;
    } else if (borderIdx < 2u * (texelDim + 2u) + texelDim) {
        // Left column (x=0, excluding corners already handled)
        let localIdx = borderIdx - 2u * (texelDim + 2u);
        borderX = 0;
        borderY = i32(localIdx) + 1;
        srcX = 1;
        srcY = clamp(tdb - 1 - borderY, 1, td);
    } else {
        // Right column (x=texelDim+1, excluding corners)
        let localIdx = borderIdx - 2u * (texelDim + 2u) - texelDim;
        borderX = td + 1;
        borderY = i32(localIdx) + 1;
        srcX = td;
        srcY = clamp(tdb - 1 - borderY, 1, td);
    }

    let dstCoord = vec2i(probeOriginX + borderX, probeOriginY + borderY);
    let srcCoord = vec2i(probeOriginX + srcX, probeOriginY + srcY);

    let srcColor = textureLoad(sourceAtlas, srcCoord, 0);
    textureStore(destAtlas, dstCoord, srcColor);
}

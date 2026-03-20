@group(0) @binding(0) var volumetricTex: texture_2d<f32>;
@group(0) @binding(1) var depthTex: texture_depth_2d;
@group(0) @binding(2) var<uniform> camera: CameraUniforms;

struct FragmentInput {
    @builtin(position) fragcoord: vec4f,
    @location(0) uv: vec2f,
}

// Convert non-linear depth to linear view Z
fn linearizeDepth(depth: f32) -> f32 {
    let clipPos = vec4f(0.0, 0.0, depth, 1.0);
    let viewPos = camera.inv_proj_mat * clipPos;
    return viewPos.z / viewPos.w;
}

@fragment
fn main(in: FragmentInput) -> @location(0) vec4f {
    let fullResCoord = vec2i(in.fragcoord.xy);
    let myRawDepth = textureLoad(depthTex, fullResCoord, 0);
    let myLinearDepth = linearizeDepth(myRawDepth);

    let lowResSize = vec2f(textureDimensions(volumetricTex));
    let lowResCoord = in.uv * lowResSize - 0.5;
    let baseCoord = floor(lowResCoord);
    let fracCoord = fract(lowResCoord);

    var totalScattering = vec3f(0.0);
    var totalWeight = 0.0;

    // 2x2 neighborhood joint bilateral upsampling
    for (var y = 0u; y <= 1u; y++) {
        for (var x = 0u; x <= 1u; x++) {
            let offset = vec2f(f32(x), f32(y));
            let sampleCoordLowRes = baseCoord + offset;
            
            // Map neighbor back to full-res UV to get its depth for comparison
            let sampleUV = (sampleCoordLowRes + 0.5) / lowResSize;
            let fullResDim = vec2f(textureDimensions(depthTex));
            let fullResSampleCoord = vec2i(sampleUV * fullResDim);
            
            let neighborRawDepth = textureLoad(depthTex, fullResSampleCoord, 0);
            let neighborLinearDepth = linearizeDepth(neighborRawDepth);
            
            // Depth weight: exponential falloff based on absolute depth difference in view space (meters)
            let depthDiff = abs(myLinearDepth - neighborLinearDepth);
            let depthWeight = exp(-depthDiff * 2.0); // Strongly reject neighbors with different depths
            
            // Spatial weight (bilinear)
            let spatialDist = abs(offset - fracCoord);
            let spatialWeight = (1.0 - spatialDist.x) * (1.0 - spatialDist.y);
            
            let weight = spatialWeight * depthWeight;
            
            let color = textureLoad(volumetricTex, vec2i(sampleCoordLowRes), 0).rgb;
            
            totalScattering += color * weight;
            totalWeight += weight;
        }
    }

    var finalScattering = vec3f(0.0);
    if (totalWeight < 0.0001) {
        // Fallback to nearest neighbor if depth discontinuity rejected all bilinear taps
        finalScattering = textureLoad(volumetricTex, vec2i(baseCoord + round(fracCoord)), 0).rgb;
    } else {
        finalScattering = totalScattering / totalWeight;
    }

    // Tone mapping and gamma correction 
    // We apply this only AFTER filtering linear HDR values to preserve physically correct blending
    let mapped = finalScattering / (finalScattering + vec3f(1.0));
    let corrected = pow(mapped, vec3f(1.0/2.2));
    
    return vec4f(corrected, 1.0);
}

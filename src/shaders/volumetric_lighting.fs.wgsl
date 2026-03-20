@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var depthTex: texture_depth_2d;
@group(0) @binding(2) var<uniform> sunLight: SunLight;
@group(0) @binding(3) var vsmPhysAtlas: texture_depth_2d;
@group(0) @binding(4) var<storage, read> vsmPageTable: array<u32>;
@group(0) @binding(5) var<uniform> vsmUniforms: VSMUniforms;

struct FragmentInput {
    @builtin(position) fragcoord: vec4f,
    @location(0) uv: vec2f,
}

// Interleaved Gradient Noise for dithering
fn interleavedGradientNoise(uv: vec2f) -> f32 {
    let magic = vec3f(0.06711056, 0.00583715, 52.9829189);
    return fract(magic.z * fract(dot(uv, magic.xy)));
}

@fragment
fn main(in: FragmentInput) -> @location(0) vec4f {
    if (sunLight.color.a < 0.5) { return vec4f(0.0); }

    // Fetch full-res depth using screen UV, making this pass resolution-independent
    let texDims = vec2f(textureDimensions(depthTex));
    let fullResCoord = vec2i(in.uv * texDims);
    let depth = textureLoad(depthTex, fullResCoord, 0);
    let clipPos = vec4f(in.uv.x * 2.0 - 1.0, 1.0 - in.uv.y * 2.0, depth, 1.0);
    
    // Convert clip space to view space
    let viewPos4 = camera.inv_proj_mat * clipPos;
    let viewPos = viewPos4.xyz / viewPos4.w;
    
    // Convert view space to world space direction using WGSL vector * matrix trick for transpose
    let worldDirRaw = vec4f(viewPos, 0.0) * camera.view_mat;
    let worldPos = camera.camera_pos.xyz + worldDirRaw.xyz;
    
    let startPos = camera.camera_pos.xyz;
    let endPos = worldPos;
    
    let rayDirUnnorm = endPos - startPos;
    let dist = length(rayDirUnnorm);
    
    // Define max distance for sky/deep background
    let MAX_DIST = sunLight.volumetric_params.w;
    
    // If depth is 1.0, it's the skybox. Raymarch up to MAX_DIST
    let marchDist = select(min(dist, MAX_DIST), MAX_DIST, depth >= 1.0);
    let rayDir = rayDirUnnorm / dist;
    
    let NUM_STEPS = u32(max(1.0, sunLight.shadow_params.z));
    let stepSize = marchDist / f32(NUM_STEPS);
    
    let L = normalize(sunLight.direction.xyz);
    let V = -rayDir;
    let cosTheta = dot(L, V);
    
    // Mie scattering parameters
    let g = 0.8;
    let g2 = g * g;
    let miePhase = (1.0 - g2) / (4.0 * PI * pow(1.0 + g2 - 2.0 * g * cosTheta, 1.5));
    
    var scattering = vec3f(0.0);
    
    let noise = interleavedGradientNoise(in.fragcoord.xy);
    let startOffset = stepSize * noise;
    
    let sun = sunLight;
    
    for (var i = 0u; i < NUM_STEPS; i++) {
        let distFromCam = startOffset + f32(i) * stepSize;
        let samplePos = startPos + rayDir * distFromCam;
        
        let shadow = calculateShadowVSMSimple(vsmPhysAtlas, vsmUniforms, sun, samplePos, L);
        
        // Height Fog: Dense near ground (y=0), fading exponentially upwards
        let heightFog = exp(-max(samplePos.y, -2.0) * sun.volumetric_params.y) * sun.volumetric_params.z + 0.1;
        
        // Distance attenuation: Fade out seamlessly at MAX_DIST
        let distanceAttenuation = 1.0 - smoothstep(MAX_DIST * 0.7, MAX_DIST, distFromCam);
        
        // Accumulate scattering
        let scatteringDensity = sun.volumetric_params.x * heightFog;
        scattering += sun.color.rgb * sun.direction.w * shadow * miePhase * stepSize * scatteringDensity * distanceAttenuation;
    }
    
    // Output raw linear HDR scattering. Tonemapping will happen in the composite pass!
    return vec4f(scattering, 1.0); 
}

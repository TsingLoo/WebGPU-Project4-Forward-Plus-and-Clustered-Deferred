// No includes needed, TS handles it
@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<uniform> constants: SurfelGridConstants;
@group(0) @binding(2) var<storage, read_write> surfels: array<Surfel>;

// BVH bindings
@group(1) @binding(0) var<storage, read> bvhNodes: array<BVHNode>;
@group(1) @binding(1) var<storage, read> bvhPositions: array<vec4f>;
@group(1) @binding(2) var<storage, read> bvhIndices: array<vec4u>;

// Env map
@group(2) @binding(0) var envMap: texture_cube<f32>;
@group(2) @binding(1) var envSampler: sampler;

// Randomness
@group(3) @binding(0) var<uniform> randoms: vec4f;

// Sun & Shadow
@group(0) @binding(3) var<uniform> sun: SunLight;
@group(0) @binding(4) var vsmPhysAtlas: texture_depth_2d;
@group(0) @binding(5) var<uniform> vsm: VSMUniforms;

fn pcg_hash(seed: u32) -> u32 {
    var state = seed * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn randFloat(seed: ptr<function, u32>) -> f32 {
    *seed = pcg_hash(*seed);
    return f32(*seed) / 4294967296.0;
}

fn sampleCosineHemisphere(n: vec3f, r1: f32, r2: f32) -> vec3f {
    let z = sqrt(1.0 - r2);
    let phi = 2.0 * 3.14159265 * r1;
    let x = cos(phi) * sqrt(r2);
    let y = sin(phi) * sqrt(r2);
    
    // Build tangent space
    var up = vec3f(0.0, 1.0, 0.0);
    if (abs(n.y) > 0.99) { up = vec3f(1.0, 0.0, 0.0); }
    let right = normalize(cross(up, n));
    up = cross(n, right);
    
    return right * x + up * y + n * z;
}

// Integrator Pass
@compute @workgroup_size(64, 1, 1)
fn integratorMain(@builtin(global_invocation_id) global_id: vec3u) {
    let idx = global_id.x;
    if (idx >= constants.maxSurfels) { return; }
    
    let time_ms = bitcast<u32>(randoms.y); 
    let frame_idx = time_ms % 4u;
    
    // Time slicing: only evaluate a quarter of surfels per frame
    if ((idx % 4u) != frame_idx) { return; }
    
    let surfel = surfels[idx];
    if (surfel.age == 0.0) { return; }
    
    var seed = pcg_hash(idx ^ bitcast<u32>(randoms.x));
    
    var totalIrradiance = vec3f(0.0);
    let RAYS = constants.raysPerSurfel;
    
    for (var r = 0u; r < RAYS; r++) {
        let r1 = randFloat(&seed);
        let r2 = randFloat(&seed);
        
        let dir = sampleCosineHemisphere(surfel.normal, r1, r2);
        
        var ray: Ray;
        // Bump bias along normal to avoid self-intersection
        ray.origin = surfel.position + surfel.normal * 0.1;
        ray.direction = dir;
        
        // Trace against BVH
        let hit = bvhIntersectFirstHit(&bvhNodes, &bvhPositions, &bvhIndices, ray);
        
        var sampleColor = vec3f(0.0);
        if (hit.didHit) {
            let hit_pos = ray.origin + ray.direction * hit.dist;
            
            // Assume generic bounce albedo: Sponza average reflectance is roughly 0.6
            let hit_albedo = vec3f(0.6); 
            
            // Check shadow using VSM clipmap
            let shadow = calculateShadowVSMSimple(vsmPhysAtlas, vsm, sun, hit_pos, hit.normal);
            
            if (shadow > 0.0) {
                // Diffuse lighting from sun
                let NdotL = max(dot(hit.normal, normalize(sun.direction.xyz)), 0.0);
                let directLighting = sun.color.rgb * sun.direction.w * NdotL * shadow;
                sampleColor = (hit_albedo / 3.14159) * directLighting;
            } else {
                // Dimmer fallback for deepest multi-bounce pockets
                sampleColor = vec3f(0.01);
            }
        } else {
            // Environment map directly (Sky hit)
            let env = textureSampleLevel(envMap, envSampler, dir, 0.0);
            sampleColor = env.rgb;
        }
        
        // Accumulate
        totalIrradiance += sampleColor;
    }
    
    let currentIrradiance = totalIrradiance / f32(RAYS);
    
    // MSME (Multi-Scale Mean Estimator)
    let lrSlow = 0.02;
    let lrFast = 0.1;
    let variance = surfels[idx].variance;
    
    // Update short-term and long-term
    let diff = currentIrradiance - surfels[idx].shortMean;
    let currentVar = dot(diff, diff);
    
    // Dynamic learning rate based on variance spike (detect lighting changes e.g. moving sun)
    // If irradiance jumps significantly compared to historical variance, we discard history
    let varianceRatio = clamp(currentVar / (variance + 0.001), 1.0, 50.0);
    let dynamicLr = clamp(lrSlow * varianceRatio, lrSlow, 1.0);
    
    // Fast-track lighting for newborn surfels over the first 60 frames (1.0 sec) to stabilize perfectly.
    // Instead of jumping abruptly, blend strongly so 8-ray variances can quickly cancel out.
    if (surfel.age <= 60.0) {
        surfels[idx].variance = currentVar;
        surfels[idx].shortMean = mix(surfels[idx].shortMean, currentIrradiance, 0.25);
        surfels[idx].irradiance = mix(surfels[idx].irradiance, currentIrradiance, 0.25);
    } else {
        surfels[idx].variance = mix(variance, currentVar, 0.1);
        surfels[idx].shortMean = mix(surfels[idx].shortMean, currentIrradiance, lrFast);
        surfels[idx].irradiance = mix(surfels[idx].irradiance, currentIrradiance, dynamicLr);
    }
}

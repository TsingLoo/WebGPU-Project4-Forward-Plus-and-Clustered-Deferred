@group(${bindGroup_scene}) @binding(0) var<uniform> camera: CameraUniforms;
@group(${bindGroup_scene}) @binding(1) var<storage, read> lightSet: LightSet;
@group(${bindGroup_scene}) @binding(2) var<storage, read> tileOffsets: array<TileMeta>;
@group(${bindGroup_scene}) @binding(3) var<storage, read> globalLightIndices: LightIndexListReadOnly;
@group(${bindGroup_scene}) @binding(4) var<uniform> clusterSet: ClusterSet;
@group(${bindGroup_scene}) @binding(5) var irradianceMap: texture_cube<f32>;
@group(${bindGroup_scene}) @binding(6) var prefilteredMap: texture_cube<f32>;
@group(${bindGroup_scene}) @binding(7) var brdfLut: texture_2d<f32>;
@group(${bindGroup_scene}) @binding(8) var iblSampler: sampler;
@group(${bindGroup_scene}) @binding(9) var ddgiIrradianceAtlas: texture_2d<f32>;
@group(${bindGroup_scene}) @binding(10) var ddgiVisibilityAtlas: texture_2d<f32>;
@group(${bindGroup_scene}) @binding(11) var<uniform> ddgiParams: DDGIUniforms;
@group(${bindGroup_scene}) @binding(12) var ddgiSampler: sampler;
@group(${bindGroup_scene}) @binding(13) var<uniform> sunLight: SunLight;
// VSM bindings
@group(${bindGroup_scene}) @binding(14) var vsmPhysAtlas: texture_depth_2d;
@group(${bindGroup_scene}) @binding(15) var vsmShadowSampler: sampler_comparison;
@group(${bindGroup_scene}) @binding(16) var<storage, read> vsmPageTable: array<u32>;
@group(${bindGroup_scene}) @binding(17) var<uniform> vsmUniforms: VSMUniforms;
// NRC bindings
@group(${bindGroup_scene}) @binding(18) var nrcInferenceTex: texture_2d<f32>;
@group(${bindGroup_scene}) @binding(19) var<uniform> nrcParams: NRCUniforms;
// G-buffer bindings for SSRT
@group(${bindGroup_scene}) @binding(20) var gBufferPosition: texture_2d<f32>;
@group(${bindGroup_scene}) @binding(21) var gBufferNormal: texture_2d<f32>;
@group(${bindGroup_scene}) @binding(22) var gBufferAlbedo: texture_2d<f32>;

@group(${bindGroup_material}) @binding(0) var diffuseTex: texture_2d<f32>;
@group(${bindGroup_material}) @binding(1) var diffuseTexSampler: sampler;

struct PBRParams {
    roughness: f32,
    metallic: f32,
    has_mr_texture: f32,
    has_normal_texture: f32,
    base_color_factor: vec4f,
    _reserved: vec4f,
}
@group(${bindGroup_material}) @binding(2) var<uniform> pbrParams: PBRParams;
@group(${bindGroup_material}) @binding(3) var metallicRoughnessTex: texture_2d<f32>;
@group(${bindGroup_material}) @binding(4) var metallicRoughnessTexSampler: sampler;
@group(${bindGroup_material}) @binding(5) var normalTex: texture_2d<f32>;
@group(${bindGroup_material}) @binding(6) var normalTexSampler: sampler;

struct FragmentInput
{
    @builtin(position) fragcoord: vec4f,
    @location(0) pos_world: vec3f,
    @location(1) nor_world: vec3f,
    @location(2) uv: vec2f,
    @location(3) tangent_world: vec4f
}

// SSGI Helper: Hash functions for random sampling
fn hash22(p: vec2f) -> vec2f {
    var p3 = fract(vec3f(p.xyx) * vec3f(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.xx + p3.yz) * p3.zy);
}

// SSGI Helper: Cosine-weighted hemisphere sample
fn getCosHemisphereSample(n: vec3f, u: vec2f) -> vec3f {
    let r = sqrt(u.x);
    let theta = 2.0 * 3.14159265359 * u.y;
    let x = r * cos(theta);
    let y = r * sin(theta);
    let z = sqrt(max(0.0, 1.0 - u.x));
    let up = select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 0.0, 1.0), abs(n.z) < 0.999);
    let tangent = normalize(cross(up, n));
    let bitangent = cross(n, tangent);
    return tangent * x + bitangent * y + n * z;
}

@fragment
fn main(in: FragmentInput) -> @location(0) vec4f
{
    let diffuseColor = textureSample(diffuseTex, diffuseTexSampler, in.uv) * pbrParams.base_color_factor;
    if (diffuseColor.a < 0.5f) {
        discard;
    }

    let albedo = diffuseColor.rgb;

    // Per-pixel metallic/roughness/AO from texture (glTF ORM packing: R = occlusion, G = roughness, B = metallic)
    var metallic = pbrParams.metallic;
    var roughness = pbrParams.roughness;
    var ao = 1.0;
    if (pbrParams.has_mr_texture > 0.5) {
        let mrSample = textureSample(metallicRoughnessTex, metallicRoughnessTexSampler, in.uv);
        // glTF spec: metallicRoughness texture has G=roughness, B=metallic
        // Occlusion is a separate texture (not bound here), so keep ao = 1.0
        roughness = roughness * mrSample.g; // scalar * texture per glTF spec
        metallic = metallic * mrSample.b;
    }
    roughness = max(roughness, 0.04); // clamp to avoid singularity

    // Normal mapping: build TBN matrix and sample normal map
    var N = normalize(in.nor_world);
    let vertexNormal = N; // save for debug
    if (pbrParams.has_normal_texture > 0.5) {
        // Re-orthogonalize tangent against normal (Gram-Schmidt)
        let T_raw = in.tangent_world.xyz - N * dot(in.tangent_world.xyz, N);
        let T_len = length(T_raw);
        var T = vec3f(0.0);
        if (T_len > 0.001) {
            T = T_raw / T_len;
        } else {
            // Degenerate tangent - pick one orthogonal to N
            let refVec = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(N.y) > 0.9);
            T = normalize(cross(N, refVec));
        }
        // Handedness: default to 1.0 if tangent.w is zero (missing data)
        let handedness = select(in.tangent_world.w, 1.0, abs(in.tangent_world.w) < 0.5);
        let B = normalize(cross(T, N)) * handedness;
        let tbn = mat3x3f(T, B, N);
        // Sample normal map (stored as [0,1], convert to [-1,1])
        let normalSample = textureSample(normalTex, normalTexSampler, in.uv).rgb;
        let tangentNormal = normalSample * 2.0 - 1.0;
        N = normalize(tbn * tangentNormal);
    }

    let V = normalize(camera.camera_pos.xyz - in.pos_world);

    // ---- Cluster lookup ----
    let screen_width = f32(clusterSet.screen_width);
    let screen_height = f32(clusterSet.screen_height);

    let num_clusters_X = clusterSet.num_clusters_X;
    let num_clusters_Y = clusterSet.num_clusters_Y;
    let num_clusters_Z = clusterSet.num_clusters_Z;

    let screen_size_cluster_x = screen_width / f32(num_clusters_X);
    let screen_size_cluster_y = screen_height / f32(num_clusters_Y);

    let clusterid_x = u32(in.fragcoord.x / screen_size_cluster_x);
    let clusterid_y_unflipped = u32(in.fragcoord.y / screen_size_cluster_y);
    let clusterid_y = clamp((num_clusters_Y - 1u) - clusterid_y_unflipped, 0u, num_clusters_Y - 1u);

    let pos_view = (camera.view_mat * vec4f(in.pos_world, 1.0)).xyz;
    let z_view = pos_view.z;

    let near = camera.near_plane; 
    let far = camera.far_plane;
    let clamped_Z_positive = clamp(-z_view, near, far);

    let logFN = log(far/near);
    let SCALE = f32(num_clusters_Z) / logFN;
    let BIAS = SCALE * log(near);
    let slice = log(clamped_Z_positive) * SCALE - BIAS;
    let cluster_z = clamp(u32(floor(slice)), 0u, num_clusters_Z - 1u);

    let cluster_index = cluster_z * (num_clusters_X * num_clusters_Y) +
                          clusterid_y * num_clusters_X +
                          clusterid_x;

    let lightmeta = tileOffsets[cluster_index];
    let offset = lightmeta.offset;
    let count = lightmeta.count;

    // ---- Direct lighting (PBR Cook-Torrance) ----
    var Lo = vec3f(0.0);
    for (var i = 0u; i < count; i += 1u) {
        let light_idx = globalLightIndices.indices[offset + i];
        let light = lightSet.lights[light_idx];
        Lo += calculateLightContribPBR(light, in.pos_world, N, V, albedo, metallic, roughness);
    }

    // Sun/directional light with VSM shadow
    let shadow = calculateShadowVSM(vsmPhysAtlas, vsmUniforms, sunLight, in.pos_world, N);
    Lo += calculateSunLightPBR(sunLight, in.pos_world, N, V, albedo, metallic, roughness, shadow);

    // ---- IBL Ambient (split-sum approximation) ----
    let F0 = mix(vec3f(0.04), albedo, metallic);
    let NdotV = max(dot(N, V), 0.0);
    let F = fresnelSchlickRoughness(NdotV, F0, roughness);

    let kS = F;
    let kD = (vec3f(1.0) - kS) * (1.0 - metallic);

    // Diffuse IBL from preconvolved irradiance map
    // Use geometric normal for low-frequency diffuse - avoids normal map modulating ambient
    let iblIrradiance = textureSample(irradianceMap, iblSampler, vertexNormal).rgb;

    // Specular IBL (split-sum)
    let R = reflect(-V, N);
    let maxLod = 4.0; // PREFILTER_MIP_LEVELS - 1
    let prefilteredColor = textureSampleLevel(prefilteredMap, iblSampler, R, roughness * maxLod).rgb;
    let brdf = textureSample(brdfLut, iblSampler, vec2f(NdotV, roughness)).rg;
    let specularIBL = prefilteredColor * (F * brdf.x + brdf.y);

    // Build ambient: scale IBL independently from DDGI
    var diffuseAmbient = vec3f(0.0);
    if (ddgiParams.ddgi_enabled.x > 0.5) {
        // Inline DDGI irradiance sampling (trilinear probe interpolation + Chebyshev visibility)
        let ddgi_spacing = ddgiParams.grid_spacing.xyz;
        let ddgi_gridMin = ddgiParams.grid_min.xyz;
        let ddgi_normalBias = ddgiParams.hysteresis.z;
        let ddgi_viewBias = ddgiParams.hysteresis.w;
        // Apply normal bias and view bias to avoid self-intersection (DDGI shadow acne)
        let ddgi_biasedPos = in.pos_world + vertexNormal * ddgi_normalBias + V * ddgi_viewBias;
        let ddgi_fractIdx = (ddgi_biasedPos - ddgi_gridMin) / ddgi_spacing;
        let ddgi_baseIdx = vec3i(floor(ddgi_fractIdx));
        let ddgi_alpha = ddgi_fractIdx - floor(ddgi_fractIdx);

        var ddgi_totalIrr = vec3f(0.0);
        var ddgi_totalW = 0.0;

        for (var dz = 0; dz < 2; dz++) {
            for (var dy = 0; dy < 2; dy++) {
                for (var dx = 0; dx < 2; dx++) {
                    let p_offset = vec3i(dx, dy, dz);
                    let p_gridIdx = clamp(ddgi_baseIdx + p_offset, vec3i(0), ddgiParams.grid_count.xyz - vec3i(1));
                    let p_pos = ddgiProbePosition(p_gridIdx, ddgiParams);
                    let p_idx = ddgiProbeLinearIndex(p_gridIdx, ddgiParams);

                    let p_dir = ddgi_biasedPos - p_pos;
                    let p_dist = length(p_dir);
                    let p_dirN = select(vertexNormal, normalize(p_dir), p_dist > 0.001);

                    // Use geometric normal for wrap-around test
                    let p_wrap = (dot(p_dirN, vertexNormal) + 1.0) * 0.5;
                    if (p_wrap <= 0.0) { continue; }

                    let p_tri = vec3f(
                        select(1.0 - ddgi_alpha.x, ddgi_alpha.x, dx == 1),
                        select(1.0 - ddgi_alpha.y, ddgi_alpha.y, dy == 1),
                        select(1.0 - ddgi_alpha.z, ddgi_alpha.z, dz == 1)
                    );
                    var p_w = p_tri.x * p_tri.y * p_tri.z;

                    // Chebyshev visibility
                    let p_visUV = ddgiVisibilityTexelCoord(p_idx, octEncode(p_dirN), ddgiParams);
                    let p_vis = textureSampleLevel(ddgiVisibilityAtlas, ddgiSampler, p_visUV, 0.0).rg;
                    if (p_dist > p_vis.x) {
                        let p_var = max(p_vis.y - p_vis.x * p_vis.x, 0.0001);
                        let p_d = p_dist - p_vis.x;
                        let p_cheb = p_var / (p_var + p_d * p_d);
                        p_w *= max(p_cheb * p_cheb * p_cheb, 0.0);
                    }

                    p_w *= p_wrap;
                    if (p_w < 0.00001) { continue; } // skip probes with negligible weight

                    // Use geometric normal for irradiance lookup - DDGI probes are low-res
                    // and normal map variations would sample wildly different atlas regions
                    let p_irrUV = ddgiIrradianceTexelCoord(p_idx, octEncode(vertexNormal), ddgiParams);
                    let p_irr_encoded = textureSampleLevel(ddgiIrradianceAtlas, ddgiSampler, p_irrUV, 0.0).rgb;
                    // Clamp to [0,1] before pow decode: atlas border interpolation can
                    // produce values slightly > 1, and pow(1.1, 5) = 1.61 creates fireflies
                    let p_irr = pow(clamp(p_irr_encoded, vec3f(0.0), vec3f(1.0)), vec3f(5.0));
                    // Clamp decoded irradiance to prevent HDR fireflies
                    let p_irr_clamped = min(p_irr, vec3f(10.0));
                    ddgi_totalIrr += p_irr_clamped * p_w;
                    ddgi_totalW += p_w;
                }
            }
        }
        if (ddgi_totalW > 0.0) { ddgi_totalIrr /= ddgi_totalW; }

        // ==== Phase 1: SSGI Final Gather ====
        // Screen-Space Ray Tracing: Shoot a few rays per pixel
        var ssgiRadiance = vec3f(0.0);
        var ssgiHitCount = 0.0;
        
        let numSSGIRays = select(0, 2, ddgiParams.ddgi_enabled.w > 0.5); 
        for (var i = 0; i < numSSGIRays; i++) {
            // Using a structured dither pattern to eliminate chaotic fireflies (since we lack a denoiser)
            let bayer = fract(in.fragcoord.x * 0.5 + in.fragcoord.y * 0.25);
            let u = vec2f(fract(bayer + f32(i)*0.5), fract(bayer*1.618 + f32(i)*0.618));
            
            let rayDir = getCosHemisphereSample(vertexNormal, u);
            let rayOrigin = in.pos_world + vertexNormal * 0.05;
            let rayEnd = rayOrigin + rayDir * 10.0; // short max trace distance for speed
            
            let originView = (camera.view_mat * vec4f(rayOrigin, 1.0)).xyz;
            let endView = (camera.view_mat * vec4f(rayEnd, 1.0)).xyz;
            
            if (originView.z < 0.0) { // Forward object
                let p0Clip = camera.proj_mat * vec4f(originView, 1.0);
                let p1Clip = camera.proj_mat * vec4f(endView, 1.0);
                let p0NDC = p0Clip.xy / p0Clip.w;
                let p1NDC = p1Clip.xy / p1Clip.w;
                
                let screenDims = vec2f(f32(clusterSet.screen_width), f32(clusterSet.screen_height));
                let p0Screen = (p0NDC * vec2f(0.5, -0.5) + 0.5) * screenDims;
                let p1Screen = (p1NDC * vec2f(0.5, -0.5) + 0.5) * screenDims;
                
                let deltaScreen = p1Screen - p0Screen;
                let steps = min(20.0, max(abs(deltaScreen.x), abs(deltaScreen.y))); // Cap max steps to 20 for huge performance boost
                
                if (steps > 1.0) {
                    let stepSize = 1.0 / steps;
                    let z0 = originView.z;
                    let z1 = min(endView.z, -0.1); 
                    let invZ0 = 1.0 / z0;
                    let invZ1 = 1.0 / z1;
                    
                    var rayHit = false;
                    for (var s = 1; s <= 32; s++) {
                        let t = f32(s) * stepSize;
                        if (t > 1.0) { break; }
                        
                        let ssi = vec2i(mix(p0Screen, p1Screen, t));
                        if (ssi.x < 0 || ssi.y < 0 || ssi.x >= i32(screenDims.x) || ssi.y >= i32(screenDims.y)) { break; }
                        
                        let hitPos = textureLoad(gBufferPosition, ssi, 0).xyz;
                        if (dot(hitPos, hitPos) < 0.1) { continue; }
                        
                        let hitViewZ = (camera.view_mat * vec4f(hitPos, 1.0)).z;
                        let expectedInvZ = mix(invZ0, invZ1, t);
                        let expectedZ = 1.0 / expectedInvZ;
                        
                        let thickness = 1.0; 
                        if (hitViewZ > expectedZ && hitViewZ < expectedZ + thickness) {
                            let hitAlbedo = textureLoad(gBufferAlbedo, ssi, 0).rgb;
                            
                            // Cheap strong color bleeding proxy: 
                            // Amplified DDGI base irradiance + IBL floor to guarantee bright vivid bounces
                            let hitIrradiance = ddgi_totalIrr * 2.5 + iblIrradiance * 0.5;
                            ssgiRadiance += hitAlbedo * hitIrradiance;
                            
                            ssgiHitCount += 1.0;
                            rayHit = true;
                            break;
                        }
                    }
                }
            }
        }
        
        // Blend SSGI (Hits) and DDGI (Misses)
        let ddgiBounce = ddgi_totalIrr * albedo;
        let avgSSGI = select(vec3f(0.0), ssgiRadiance / max(ssgiHitCount, 1.0), ssgiHitCount > 0.1);
        let hitRatio = ssgiHitCount / f32(max(numSSGIRays, 1));
        
        // Final indirect diffuse = smoothly mix DDGI (missed rays) and SSGI (hit rays)
        let directBounces = mix(ddgiBounce, avgSSGI * albedo, hitRatio);
        
        let iblFloor = iblIrradiance * albedo * 0.25;
        diffuseAmbient = max(directBounces, iblFloor);
    } else if (nrcParams.scene_min.w > 0.5) {
        // NRC mode: sample the neural radiance cache inference texture
        let screenUV = in.fragcoord.xy / vec2f(nrcParams.screen_dims.x, nrcParams.screen_dims.y);
        let nrcTexSize = textureDimensions(nrcInferenceTex);
        let nrcCoord = vec2i(i32(screenUV.x * f32(nrcTexSize.x)), i32(screenUV.y * f32(nrcTexSize.y)));
        let nrcIrradiance = textureLoad(nrcInferenceTex, nrcCoord, 0).rgb;
        // NRC provides cached irradiance; apply albedo modulation
        let nrcBounce = nrcIrradiance * albedo;
        let iblFloor2 = iblIrradiance * albedo * 0.15;
        diffuseAmbient = max(nrcBounce, iblFloor2);
    } else {
        // No DDGI: use IBL irradiance with moderate scaling
        diffuseAmbient = iblIrradiance * albedo * 1.0;
    }

    // ---- DDGI Debug Visualization ----
    let debugMode = i32(ddgiParams.ddgi_enabled.y);
    if (debugMode == 1) {
        // Mode 1: Raw DDGI irradiance (should NOT be black if probes have data)
        if (ddgiParams.ddgi_enabled.x > 0.5) {
            // Re-sample center probe for this fragment to show raw irradiance
            let dbg_spacing = ddgiParams.grid_spacing.xyz;
            let dbg_gridMin = ddgiParams.grid_min.xyz;
            let dbg_fractIdx = (in.pos_world - dbg_gridMin) / dbg_spacing;
            let dbg_baseIdx = clamp(vec3i(floor(dbg_fractIdx)), vec3i(0), ddgiParams.grid_count.xyz - vec3i(1));
            let dbg_probeIdx = ddgiProbeLinearIndex(dbg_baseIdx, ddgiParams);
            let dbg_irrUV = ddgiIrradianceTexelCoord(dbg_probeIdx, octEncode(N), ddgiParams);
            let dbg_raw = textureSampleLevel(ddgiIrradianceAtlas, ddgiSampler, dbg_irrUV, 0.0).rgb;
            // Show raw atlas value (gamma-encoded) amplified
            return vec4f(dbg_raw * 3.0, 1.0);
        }
        return vec4f(1.0, 0.0, 1.0, 1.0); // Magenta = DDGI disabled
    }
    if (debugMode == 2) {
        // Mode 2: Decoded DDGI irradiance (trilinear sampled, after pow 5)
        if (ddgiParams.ddgi_enabled.x > 0.5) {
            let dbg2_spacing = ddgiParams.grid_spacing.xyz;
            let dbg2_gridMin = ddgiParams.grid_min.xyz;
            let dbg2_fractIdx = (in.pos_world - dbg2_gridMin) / dbg2_spacing;
            let dbg2_baseIdx = clamp(vec3i(floor(dbg2_fractIdx)), vec3i(0), ddgiParams.grid_count.xyz - vec3i(1));
            let dbg2_probeIdx = ddgiProbeLinearIndex(dbg2_baseIdx, ddgiParams);
            let dbg2_irrUV = ddgiIrradianceTexelCoord(dbg2_probeIdx, octEncode(N), ddgiParams);
            let dbg2_encoded = textureSampleLevel(ddgiIrradianceAtlas, ddgiSampler, dbg2_irrUV, 0.0).rgb;
            let dbg2_decoded = pow(max(dbg2_encoded, vec3f(0.0)), vec3f(5.0));
            let dbg2_mapped = dbg2_decoded / (dbg2_decoded + vec3f(1.0));
            return vec4f(pow(dbg2_mapped, vec3f(1.0/2.2)), 1.0);
        }
        return vec4f(0.0, 1.0, 1.0, 1.0); // Cyan = no DDGI data
    }
    if (debugMode == 3) {
        // Mode 3: IBL irradiance only
        let dbg_ibl = iblIrradiance;
        let dbg_mapped2 = dbg_ibl / (dbg_ibl + vec3f(1.0));
        return vec4f(pow(dbg_mapped2, vec3f(1.0/2.2)), 1.0);
    }
    if (debugMode == 4) {
        // Mode 4: Final mapped world-space normal as RGB
        return vec4f(N * 0.5 + 0.5, 1.0);
    }
    if (debugMode == 5) {
        // Mode 5: Vertex normal (before normal mapping) as RGB
        return vec4f(vertexNormal * 0.5 + 0.5, 1.0);
    }
    if (debugMode == 6) {
        // Mode 6: Tangent vector as RGB
        return vec4f(normalize(in.tangent_world.xyz) * 0.5 + 0.5, 1.0);
    }
    if (debugMode == 7) {
        // Mode 7: NdotL (sun) - bright = facing sun, dark = away
        let sunDir = normalize(sunLight.direction.xyz);
        let ndotl = max(dot(N, sunDir), 0.0);
        return vec4f(vec3f(ndotl), 1.0);
    }
    if (debugMode == 8) {
        // Mode 8: DDGI Probe Grid visualization
        // Show probe positions as colored dots overlaid on the scene
        let spacing = ddgiParams.grid_spacing.xyz;
        let gridMin = ddgiParams.grid_min.xyz;
        let relPos = (in.pos_world - gridMin) / spacing;
        let nearestProbe = round(relPos);
        let probePos = gridMin + nearestProbe * spacing;
        let distToProbe = length(in.pos_world - probePos);
        let probeRadius = min(min(spacing.x, spacing.y), spacing.z) * 0.08;
        if (distToProbe < probeRadius) {
            // Color by grid position
            let gridIdx = vec3i(nearestProbe);
            let col = vec3f(
                f32(gridIdx.x % 2),
                f32(gridIdx.y % 2),
                f32(gridIdx.z % 2)
            ) * 0.5 + 0.5;
            return vec4f(col, 1.0);
        }
        // Show faded version of normal scene + grid lines
        let gridFrac = fract(relPos);
        let gridLine = step(vec3f(0.95), gridFrac) + step(gridFrac, vec3f(0.05));
        let isGrid = max(max(gridLine.x, gridLine.y), gridLine.z);
        if (isGrid > 0.0) {
            return vec4f(0.0, 1.0, 1.0, 1.0); // Cyan grid lines
        }
        // Darken non-grid pixels slightly for contrast
        return vec4f(albedo * 0.3, 1.0);
    }

    // ---- NRC Debug Visualization ----
    if (nrcParams.scene_min.w > 0.5) {
        let nrcDebugMode = i32(nrcParams.scene_max.w);
        if (nrcDebugMode == 1) {
            // Mode 1: Raw Inference (amplified for visibility)
            let screenUV = in.fragcoord.xy / vec2f(nrcParams.screen_dims.x, nrcParams.screen_dims.y);
            let nrcTexSize = textureDimensions(nrcInferenceTex);
            let nrcCoord = vec2i(i32(screenUV.x * f32(nrcTexSize.x)), i32(screenUV.y * f32(nrcTexSize.y)));
            let nrcIrradiance = textureLoad(nrcInferenceTex, nrcCoord, 0).rgb;
            return vec4f(nrcIrradiance * 3.0, 1.0);
        }
        if (nrcDebugMode == 2) {
            // Mode 2: HDR Mapped
            let screenUV = in.fragcoord.xy / vec2f(nrcParams.screen_dims.x, nrcParams.screen_dims.y);
            let nrcTexSize = textureDimensions(nrcInferenceTex);
            let nrcCoord = vec2i(i32(screenUV.x * f32(nrcTexSize.x)), i32(screenUV.y * f32(nrcTexSize.y)));
            let nrcIrradiance = textureLoad(nrcInferenceTex, nrcCoord, 0).rgb;
            let mapped = nrcIrradiance / (nrcIrradiance + vec3f(1.0));
            return vec4f(pow(mapped, vec3f(1.0/2.2)), 1.0);
        }
    }

    // Combine: diffuse ambient + specular IBL
    // When DDGI is on, reduce specular IBL since the cubemap doesn't match interior lighting
    // DDGI only provides diffuse indirect; specular IBL from outdoor cubemap
    // creates unrealistic reflections on interior surfaces, so disable it with DDGI
    let specIBLScale = select(0.6, 0.0, ddgiParams.ddgi_enabled.x > 0.5);
    let ambient = (kD * diffuseAmbient + specularIBL * specIBLScale) * ao;

    let finalColor = ambient + Lo;

    // Tone mapping (Reinhard)
    let mapped = finalColor / (finalColor + vec3f(1.0));
    // Gamma correction
    let corrected = pow(mapped, vec3f(1.0/2.2));

    return vec4f(corrected, 1.0);
}

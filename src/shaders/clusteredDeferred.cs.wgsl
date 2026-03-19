@group(${bindGroup_scene}) @binding(0) var<uniform> camera: CameraUniforms;
@group(${bindGroup_scene}) @binding(1) var<storage, read> lightSet: LightSet;
@group(${bindGroup_scene}) @binding(2) var<storage, read> tileOffsets: array<TileMeta>;
@group(${bindGroup_scene}) @binding(3) var<storage, read> globalLightIndices: LightIndexListReadOnly;
@group(${bindGroup_scene}) @binding(4) var<uniform> clusterSet: ClusterSet;

@group(${bindGroup_scene}) @binding(5) var albedoTex: texture_2d<f32>;
@group(${bindGroup_scene}) @binding(6) var normalTex: texture_2d<f32>;
@group(${bindGroup_scene}) @binding(7) var positionTex: texture_2d<f32>;
@group(${bindGroup_scene}) @binding(8) var specularTex: texture_2d<f32>;
@group(${bindGroup_scene}) @binding(9) var depthTex: texture_depth_2d;
@group(${bindGroup_scene}) @binding(10) var outputTex: texture_storage_2d<rgba8unorm, write>;
@group(${bindGroup_scene}) @binding(11) var irradianceMap: texture_cube<f32>;
@group(${bindGroup_scene}) @binding(12) var prefilteredMap: texture_cube<f32>;
@group(${bindGroup_scene}) @binding(13) var brdfLutTex: texture_2d<f32>;
@group(${bindGroup_scene}) @binding(14) var iblSampler: sampler;
@group(${bindGroup_scene}) @binding(15) var ddgiIrradianceAtlas: texture_2d<f32>;
@group(${bindGroup_scene}) @binding(16) var ddgiVisibilityAtlas: texture_2d<f32>;
@group(${bindGroup_scene}) @binding(17) var<uniform> ddgiParams: DDGIUniforms;
@group(${bindGroup_scene}) @binding(18) var ddgiSampler: sampler;
@group(${bindGroup_scene}) @binding(19) var<uniform> sunLight: SunLight;
@group(${bindGroup_scene}) @binding(20) var shadowMap: texture_depth_2d;

@compute @workgroup_size(8, 8, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3u
) {
    let fragcoordi = vec2i(global_id.xy);
    let screen_pos_x = f32(global_id.x);
    let screen_pos_y = f32(global_id.y);

    let screen_dims = textureDimensions(albedoTex);
    if (global_id.x >= screen_dims.x || global_id.y >= screen_dims.y) {
        return;
    }

    let diffuseColor = textureLoad(albedoTex, fragcoordi, 0);
    if (diffuseColor.a < 0.5f) {
        textureStore(outputTex, fragcoordi, vec4f(0.0, 0.0, 0.0, 0.0));
        return;
    }

    let albedo = diffuseColor.rgb;
    let pos_world = textureLoad(positionTex, fragcoordi, 0).xyz;
    let nor_world = textureLoad(normalTex, fragcoordi, 0).xyz;
    let specularData = textureLoad(specularTex, fragcoordi, 0);
    
    let roughness = max(specularData.r, 0.04);
    let metallic = specularData.g;
    let ao = specularData.b; // ambient occlusion from G-buffer

    let N = normalize(nor_world);
    let V = normalize(camera.camera_pos.xyz - pos_world);

    // ---- Cluster lookup ----
    let screen_width = f32(clusterSet.screen_width);
    let screen_height = f32(clusterSet.screen_height);

    let num_clusters_X = clusterSet.num_clusters_X;
    let num_clusters_Y = clusterSet.num_clusters_Y;
    let num_clusters_Z = clusterSet.num_clusters_Z;

    let screen_size_cluster_x = screen_width / f32(num_clusters_X);
    let screen_size_cluster_y = screen_height / f32(num_clusters_Y);

    let clusterid_x = u32(screen_pos_x / screen_size_cluster_x);
    let clusterid_y_unflipped = u32(screen_pos_y / screen_size_cluster_y);
    let clusterid_y = clamp((num_clusters_Y - 1u) - clusterid_y_unflipped, 0u, num_clusters_Y - 1u);

    let pos_view = (camera.view_mat * vec4f(pos_world, 1.0)).xyz;
    let z_view = pos_view.z;

    let near = camera.near_plane;
    let far = camera.far_plane;
    let clamped_Z_positive = clamp(-z_view, near, far);

    let logFN = log(far / near);
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
        Lo += calculateLightContribPBR(light, pos_world, N, V, albedo, metallic, roughness);
    }

    // Sun/directional light with shadow (simple shadow for compute shader, no comparison sampler)
    let shadow = calculateShadowSimple(shadowMap, sunLight, pos_world, N);
    Lo += calculateSunLightPBR(sunLight, pos_world, N, V, albedo, metallic, roughness, shadow);

    // ---- IBL Ambient (split-sum approximation) ----
    let F0 = mix(vec3f(0.04), albedo, metallic);
    let NdotV = max(dot(N, V), 0.0);
    let F = fresnelSchlickRoughness(NdotV, F0, roughness);

    let kS = F;
    let kD = (vec3f(1.0) - kS) * (1.0 - metallic);

    // Diffuse IBL from preconvolved irradiance map
    let iblIrradiance = textureSampleLevel(irradianceMap, iblSampler, N, 0.0).rgb;

    // Specular IBL (split-sum)
    let R = reflect(-V, N);
    let maxLod = 4.0;
    let prefilteredColor = textureSampleLevel(prefilteredMap, iblSampler, R, roughness * maxLod).rgb;
    let brdfVal = textureSampleLevel(brdfLutTex, iblSampler, vec2f(NdotV, roughness), 0.0).rg;
    let specularIBL = prefilteredColor * (F * brdfVal.x + brdfVal.y);

    // Build ambient: scale IBL independently from DDGI
    var diffuseAmbient = vec3f(0.0);
    if (ddgiParams.ddgi_enabled.x > 0.5) {
        // Inline DDGI irradiance sampling (trilinear probe interpolation + Chebyshev visibility)
        let ddgi_spacing = ddgiParams.grid_spacing.xyz;
        let ddgi_gridMin = ddgiParams.grid_min.xyz;
        let ddgi_normalBias = ddgiParams.hysteresis.z;
        let ddgi_biasedPos = pos_world + N * ddgi_normalBias;
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

                    let p_dir = pos_world - p_pos;
                    let p_dist = length(p_dir);
                    let p_dirN = select(N, normalize(p_dir), p_dist > 0.001);

                    let p_wrap = (dot(p_dirN, N) + 1.0) * 0.5;
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
                    p_w = max(p_w, 0.0001);

                    let p_irrUV = ddgiIrradianceTexelCoord(p_idx, octEncode(N), ddgiParams);
                    let p_irr = textureSampleLevel(ddgiIrradianceAtlas, ddgiSampler, p_irrUV, 0.0).rgb;
                    ddgi_totalIrr += p_irr * p_w;
                    ddgi_totalW += p_w;
                }
            }
        }
        if (ddgi_totalW > 0.0) { ddgi_totalIrr /= ddgi_totalW; }

        // DDGI bounce at full strength + small IBL fill for areas probes don't reach
        let ddgiBounce = ddgi_totalIrr * albedo;
        let iblFill = iblIrradiance * albedo * 0.3;
        diffuseAmbient = ddgiBounce + iblFill;
    } else {
        // No DDGI: use IBL irradiance with moderate scaling
        diffuseAmbient = iblIrradiance * albedo * 0.7;
    }

    // Combine: DDGI diffuse (unscaled) + specular IBL (scaled down)
    let ambient = (kD * diffuseAmbient + specularIBL * 0.6) * ao;
    let finalColor = ambient + Lo;

    // Tone mapping (Reinhard)
    let mapped = finalColor / (finalColor + vec3f(1.0));
    // Gamma correction
    let corrected = pow(mapped, vec3f(1.0/2.2));

    textureStore(outputTex, fragcoordi, vec4f(corrected, 1.0));
}
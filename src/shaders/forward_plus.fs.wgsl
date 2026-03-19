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

@fragment
fn main(in: FragmentInput) -> @location(0) vec4f
{
    let diffuseColor = textureSample(diffuseTex, diffuseTexSampler, in.uv) * pbrParams.base_color_factor;
    if (diffuseColor.a < 0.5f) {
        discard;
    }

    let albedo = diffuseColor.rgb;

    // Per-pixel metallic/roughness from texture (glTF: G = roughness, B = metallic)
    var metallic = pbrParams.metallic;
    var roughness = pbrParams.roughness;
    if (pbrParams.has_mr_texture > 0.5) {
        let mrSample = textureSample(metallicRoughnessTex, metallicRoughnessTexSampler, in.uv);
        roughness = roughness * mrSample.g; // scalar * texture per glTF spec
        metallic = metallic * mrSample.b;
    }
    roughness = max(roughness, 0.04); // clamp to avoid singularity

    // Normal mapping: build TBN matrix and sample normal map
    var N = normalize(in.nor_world);
    if (pbrParams.has_normal_texture > 0.5) {
        let T = normalize(in.tangent_world.xyz);
        let B = cross(N, T) * in.tangent_world.w; // w = handedness
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

    // Sun/directional light
    Lo += calculateSunLightPBR(sunLight, in.pos_world, N, V, albedo, metallic, roughness);

    // ---- IBL Ambient (split-sum approximation) ----
    let F0 = mix(vec3f(0.04), albedo, metallic);
    let NdotV = max(dot(N, V), 0.0);
    let F = fresnelSchlickRoughness(NdotV, F0, roughness);

    let kS = F;
    let kD = (vec3f(1.0) - kS) * (1.0 - metallic);

    // Diffuse IBL — blend DDGI with IBL when enabled
    let iblIrradiance = textureSample(irradianceMap, iblSampler, N).rgb;
    var diffuseIBL = iblIrradiance * albedo;

    if (ddgiParams.ddgi_enabled.x > 0.5) {
        // Inline DDGI irradiance sampling (trilinear probe interpolation + Chebyshev visibility)
        let ddgi_spacing = ddgiParams.grid_spacing.xyz;
        let ddgi_gridMin = ddgiParams.grid_min.xyz;
        let ddgi_normalBias = ddgiParams.hysteresis.z;
        let ddgi_biasedPos = in.pos_world + N * ddgi_normalBias;
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

                    let p_dir = in.pos_world - p_pos;
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

        // Blend: IBL provides base ambient, DDGI adds indirect bounce light
        let ddgiBounce = ddgi_totalIrr * albedo;
        diffuseIBL = iblIrradiance * albedo * 0.3 + ddgiBounce * 0.7;
    }

    // Specular IBL
    let R = reflect(-V, N);
    let maxLod = 4.0; // PREFILTER_MIP_LEVELS - 1
    let prefilteredColor = textureSampleLevel(prefilteredMap, iblSampler, R, roughness * maxLod).rgb;
    let brdf = textureSample(brdfLut, iblSampler, vec2f(NdotV, roughness)).rg;
    let specularIBL = prefilteredColor * (F * brdf.x + brdf.y);

    let ambient = kD * diffuseIBL + specularIBL;

    let finalColor = ambient + Lo;

    // Tone mapping (Reinhard)
    let mapped = finalColor / (finalColor + vec3f(1.0));
    // Gamma correction
    let corrected = pow(mapped, vec3f(1.0/2.2));

    return vec4f(corrected, 1.0);
}

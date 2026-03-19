@group(${bindGroup_scene}) @binding(0) var<uniform> camera: CameraUniforms;
@group(${bindGroup_scene}) @binding(1) var<storage, read> lightSet: LightSet;
@group(${bindGroup_scene}) @binding(2) var<storage, read> tileOffsets: array<TileMeta>;
@group(${bindGroup_scene}) @binding(3) var<storage, read> globalLightIndices: LightIndexListReadOnly;
@group(${bindGroup_scene}) @binding(4) var<uniform> clusterSet: ClusterSet;
@group(${bindGroup_scene}) @binding(5) var irradianceMap: texture_cube<f32>;
@group(${bindGroup_scene}) @binding(6) var prefilteredMap: texture_cube<f32>;
@group(${bindGroup_scene}) @binding(7) var brdfLut: texture_2d<f32>;
@group(${bindGroup_scene}) @binding(8) var iblSampler: sampler;

@group(${bindGroup_material}) @binding(0) var diffuseTex: texture_2d<f32>;
@group(${bindGroup_material}) @binding(1) var diffuseTexSampler: sampler;

struct PBRParams {
    roughness: f32,
    metallic: f32,
    has_mr_texture: f32,
    _pad1: f32,
    base_color_factor: vec4f,
}
@group(${bindGroup_material}) @binding(2) var<uniform> pbrParams: PBRParams;
@group(${bindGroup_material}) @binding(3) var metallicRoughnessTex: texture_2d<f32>;
@group(${bindGroup_material}) @binding(4) var metallicRoughnessTexSampler: sampler;

struct FragmentInput
{
    @builtin(position) fragcoord: vec4f,
    @location(0) pos_world: vec3f,
    @location(1) nor_world: vec3f,
    @location(2) uv: vec2f
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

    let N = normalize(in.nor_world);
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

    // ---- IBL Ambient (split-sum approximation) ----
    let F0 = mix(vec3f(0.04), albedo, metallic);
    let NdotV = max(dot(N, V), 0.0);
    let F = fresnelSchlickRoughness(NdotV, F0, roughness);

    let kS = F;
    let kD = (vec3f(1.0) - kS) * (1.0 - metallic);

    // Diffuse IBL
    let irradiance = textureSample(irradianceMap, iblSampler, N).rgb;
    let diffuseIBL = irradiance * albedo;

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

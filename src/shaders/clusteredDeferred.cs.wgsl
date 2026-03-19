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

    // ---- IBL Ambient (split-sum approximation) ----
    let F0 = mix(vec3f(0.04), albedo, metallic);
    let NdotV = max(dot(N, V), 0.0);
    let F = fresnelSchlickRoughness(NdotV, F0, roughness);

    let kS = F;
    let kD = (vec3f(1.0) - kS) * (1.0 - metallic);

    // Diffuse IBL
    let irradiance = textureSampleLevel(irradianceMap, iblSampler, N, 0.0).rgb;
    let diffuseIBL = irradiance * albedo;

    // Specular IBL
    let R = reflect(-V, N);
    let maxLod = 4.0;
    let prefilteredColor = textureSampleLevel(prefilteredMap, iblSampler, R, roughness * maxLod).rgb;
    let brdfVal = textureSampleLevel(brdfLutTex, iblSampler, vec2f(NdotV, roughness), 0.0).rg;
    let specularIBL = prefilteredColor * (F * brdfVal.x + brdfVal.y);

    let ambient = kD * diffuseIBL + specularIBL;
    let finalColor = ambient + Lo;

    // Tone mapping (Reinhard)
    let mapped = finalColor / (finalColor + vec3f(1.0));
    // Gamma correction
    let corrected = pow(mapped, vec3f(1.0/2.2));

    textureStore(outputTex, fragcoordi, vec4f(corrected, 1.0));
}
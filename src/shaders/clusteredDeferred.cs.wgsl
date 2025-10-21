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
// TODO-2: implement the Forward+ fragment shader

// See naive.fs.wgsl for basic fragment shader setup; this shader should use light clusters instead of looping over all lights

// ------------------------------------
// Shading process:
// ------------------------------------
// Determine which cluster contains the current fragment.
// Retrieve the number of lights that affect the current fragment from the cluster’s data.
// Initialize a variable to accumulate the total light contribution for the fragment.
// For each light in the cluster:
//     Access the light's properties using its index.
//     Calculate the contribution of the light based on its position, the fragment’s position, and the surface normal.
//     Add the calculated contribution to the total light accumulation.
// Multiply the fragment’s diffuse color by the accumulated light contribution.
// Return the final color, ensuring that the alpha component is set appropriately (typically to 1).

struct FragmentInput
{
    @builtin(position) fragcoord: vec4f,
}

@compute @workgroup_size(8, 8, 1) // 假设 workgroup_size 为 8x8
fn main(
    @builtin(global_invocation_id) global_id: vec3u // 替换 @builtin(position)
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

    let pos_world = textureLoad(positionTex, fragcoordi, 0).xyz;
    let nor_world = textureLoad(normalTex, fragcoordi, 0).xyz;

    let screen_width = f32(clusterSet.screen_width);
    let screen_height = f32(clusterSet.screen_height);

    let num_clusters_X = clusterSet.num_clusters_X;
    let num_clusters_Y = clusterSet.num_clusters_Y;
    let num_clusters_Z = clusterSet.num_clusters_Z;
    let num_clusters = num_clusters_X * num_clusters_Y * num_clusters_Z;

    let num_lights = lightSet.numLights;

    let screen_size_cluster_x = f32(screen_width) / f32(num_clusters_X);
    let screen_size_cluster_y = f32(screen_height) / f32(num_clusters_Y);

    let clusterid_x = u32(screen_pos_x / screen_size_cluster_x);
    let clusterid_y_unflipped = u32(screen_pos_y / screen_size_cluster_y);
    let clusterid_y = clamp((num_clusters_Y - 1u) - clusterid_y_unflipped, 0u, num_clusters_Y - 1u);

    let pos_view = (camera.view_mat * vec4f(pos_world,1.0)).xyz;

    //z_view is negative in front of the camera
    let z_view = pos_view.z;

    let near = camera.near_plane; 
    let far = camera.far_plane;

    let clamped_Z_positive = clamp(- z_view, near, far);

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

    let normalized_normal = normalize(nor_world); // Normalize normal once

    var totalLightContrib = vec3f(0, 0, 0);
    for (var i = 0u; i < count; i += 1u) {
        // Get the actual light index from the global list
        let light_idx = globalLightIndices.indices[offset + i];
        
        // Get the light data
        let light = lightSet.lights[light_idx];

        // Calculate contribution (passing world space pos/normal and view matrix)
        totalLightContrib += calculateLightContrib(light, pos_world, normalized_normal);
    }

    let ambient = vec3f(${ambientR}, ${ambientG}, ${ambientB});
    let finalColor = diffuseColor.rgb * (totalLightContrib + ambient);
    textureStore(outputTex, fragcoordi, vec4f(finalColor, 1.0));
}
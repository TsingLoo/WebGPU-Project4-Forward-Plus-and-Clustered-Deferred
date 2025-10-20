@group(${bindGroup_scene}) @binding(0) var<uniform> camera: CameraUniforms;
@group(${bindGroup_scene}) @binding(1) var<storage, read> lightSet: LightSet;
@group(${bindGroup_scene}) @binding(2) var<storage, read> tileOffsets: array<TileMeta>;
@group(${bindGroup_scene}) @binding(3) var<storage, read> globalLightIndices: LightIndexListReadOnly;
@group(${bindGroup_scene}) @binding(4) var<uniform> clusterSet: ClusterSet;


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

@group(${bindGroup_material}) @binding(0) var diffuseTex: texture_2d<f32>;
@group(${bindGroup_material}) @binding(1) var diffuseTexSampler: sampler;

struct FragmentInput
{
    @builtin(position) clip_pos: vec4f,

    @location(0) pos_world: vec3f,
    @location(1) nor_world: vec3f,
    @location(2) uv: vec2f
}

@fragment
fn main(in: FragmentInput) -> @location(0) vec4f
{
    let diffuseColor = textureSample(diffuseTex, diffuseTexSampler, in.uv);
    if (diffuseColor.a < 0.5f) {
        discard;
    }

    let screen_pos_x = (in.clip_pos.x / in.clip_pos.w * 0.5 + 0.5) * f32(clusterSet.screen_width);
    let screen_pos_y = (in.clip_pos.y / in.clip_pos.w * -0.5 + 0.5) * f32(clusterSet.screen_height);

    let cluster_x = u32(screen_pos_x / (f32(clusterSet.screen_width) / f32(clusterSet.num_clusters_X)));
    let cluster_y = u32(screen_pos_y / (f32(clusterSet.screen_height) / f32(clusterSet.num_clusters_Y)));

    let pos_view = (camera.view_mat * vec4f(in.pos_world, 1.0)).xyz;
    let z_view = pos_view.z;

    let num_z_slices = f32(clusterSet.num_clusters_Z);

    let near = camera.near_plane; 
    let far = camera.far_plane;

    let clamped_z_view = clamp(z_view, -far, -near);


    let log_depth = log(-clamped_z_view / near); // Log of positive value
    let log_ratio = log(far / near);

    let safe_log_ratio = select(log_ratio, 1.0, log_ratio <= 0.0); 

    let z_slice_float = num_z_slices * log_depth / safe_log_ratio;

    let cluster_z = clamp(u32(floor(z_slice_float)), 0u, clusterSet.num_clusters_Z - 1u);

    let cluster_index = cluster_z * (clusterSet.num_clusters_X * clusterSet.num_clusters_Y) +
                          cluster_y * clusterSet.num_clusters_X +
                          cluster_x;

    let lightmeta = tileOffsets[cluster_index];
    let offset = lightmeta.offset;
    let count = lightmeta.count;

    let normalized_normal = normalize(in.nor_world); // Normalize normal once

    var totalLightContrib = vec3f(0, 0, 0);
    for (var i = 0u; i < count; i += 1u) {
        // Get the actual light index from the global list
        let light_idx = globalLightIndices.indices[offset + i];
        
        // Get the light data
        let light = lightSet.lights[light_idx];
        
        // Calculate contribution (passing world space pos/normal and view matrix)
        totalLightContrib += calculateLightContrib(light, in.pos_world, normalized_normal);
    }

    let ambient = vec3f(0.05, 0.05, 0.05); 
    let finalColor = diffuseColor.rgb * (totalLightContrib + ambient);
    
    return vec4(finalColor, 1);
}

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

@fragment
fn main(in: FragmentInput) -> @location(0) vec4f
{
    
    let screen_pos_x = in.fragcoord.x;
    let screen_pos_y = in.fragcoord.y;

    let fragcoordi = vec2i(in.fragcoord.xy);

    let diffuseColor = textureLoad(albedoTex, fragcoordi, 0);
    if (diffuseColor.a < 0.5f) {
        discard;
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


// // --- DEBUG VISUALIZATION (Unique Color Per Cluster) --- 🌈
//     // Calculate total clusters for normalization (used in color generation)
//     let total_clusters_f = f32(clusterSet.num_clusters_X * clusterSet.num_clusters_Y * clusterSet.num_clusters_Z);

//     // Generate a color based on the index.
//     // We can use a simple method mapping the index to hue.
//     let hue = f32(cluster_index) / total_clusters_f; // Hue ranges from 0.0 to approx 1.0

//     // Simple Hue to RGB conversion (approximated)
//     let R = abs(hue * 6.0 - 3.0) - 1.0;
//     let G = 2.0 - abs(hue * 6.0 - 2.0);
//     let B = 2.0 - abs(hue * 6.0 - 4.0);
//     let cluster_color = clamp(vec3f(R, G, B), vec3f(0.0), vec3f(1.0));

//     // Make clusters with index 0 black just to clearly identify it
//     // if (cluster_index == 0u) {
//     //     cluster_color = vec3f(0.0);
//     // }

//     // Output the generated unique color
//     return vec4f(cluster_color, 1.0);
// // --- DEBUG VISUALIZATION END (Unique Color Per Cluster) --- 🌈


    let lightmeta = tileOffsets[cluster_index];
    let offset = lightmeta.offset;
    let count = lightmeta.count;

    // --- DEBUG VISUALIZATION (Offset Heatmap) --- 💾
    // Define an estimated maximum possible offset. This depends on how large your
    // globalLightIndices buffer is and how many lights are typically assigned.
    // calculation from TypeScript (e.g., totalClusters * averageLightsPerTile)
    // Or you could pass this maximum value via a uniform buffer.
    // let max_possible_offset = f32(clusterSet.num_clusters_X * clusterSet.num_clusters_Y * clusterSet.num_clusters_Z * 64u); // Example: 3456 clusters * 64 lights/cluster

    // // Normalize the offset to the [0.0, 1.0] range
    // // Avoid division by zero if max_possible_offset could be 0
    // let normalized_offset = clamp(f32(offset) / max(1.0, max_possible_offset), 0.0, 1.0);

    // // Output as grayscale: Dark means small offset (early in the global list),
    // // Light means large offset (later in the global list)
    // return vec4f(normalized_offset, normalized_offset, normalized_offset, 1.0);
    
    // --- DEBUG END (Offset Heatmap) --- 💾


    // // --- DEBUG VISUALIZATION (Heatmap of Light Count) --- 🔥
    // // Define a maximum count for the visualization. Counts above this will be clamped to red.
    // // Adjust this value based on your scene and MAX_LIGHTS_PER_CLUSTER. 64 is a reasonable start.
    // let max_vis_count = 256.0; 

    // // Normalize the count to the [0.0, 1.0] range
    // let normalized_count = clamp(f32(count) / max_vis_count, 0.0, 1.0);

    // // Simple heatmap: Blue (low) -> Green (medium) -> Red (high)
    // var heatmap_color: vec3f;
    // if (normalized_count < 0.5) {
    //     // Interpolate Blue (0,0,1) to Green (0,1,0)
    //     heatmap_color = mix(vec3f(0.0, 0.0, 1.0), vec3f(0.0, 1.0, 0.0), normalized_count * 2.0);
    // } else {
    //     // Interpolate Green (0,1,0) to Red (1,0,0)
    //     heatmap_color = mix(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), (normalized_count - 0.5) * 2.0);
    // }

    // // Special case for zero lights (optional, e.g., show as black or gray)
    // if (count == 0u) {
    //     heatmap_color = vec3f(0.1, 0.1, 0.1); // Dark gray for empty clusters
    // }

    // // Return the heatmap color
    // return vec4f(heatmap_color, 1.0);
    // // --- DEBUG END (Heatmap of Light Count) --- 🔥

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

    let ambient = vec3f(${ambient[0]}, ${ambient[1]}, ${ambient[2]});
    // if(count > 0)
    // {
    //     ambient = vec3f(0.05, 0.05, 0.05); 
    // }else{
    //     ambient = vec3f(1.0, 1.0, 1.0);

    //     return vec4(ambient, 1);
    // }

    //let ambient = vec3f(0.05, 0.05, 0.05); 
    let finalColor = diffuseColor.rgb * (totalLightContrib + ambient);
    

    // totalLightContrib = vec3f(0, 0, 0);
    // for (var lightIdx = 0u; lightIdx < lightSet.numLights; lightIdx++) {
    //     let light = lightSet.lights[lightIdx];
    //     totalLightContrib += calculateLightContrib(light, in.pos_world, normalize(in.nor_world));
    // }

    // finalColor = diffuseColor.rgb * totalLightContrib;

    return vec4(finalColor, 1);
}

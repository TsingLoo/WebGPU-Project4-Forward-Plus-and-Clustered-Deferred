// No includes needed, TS handles it
@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<uniform> constants: SurfelGridConstants;
@group(0) @binding(2) var<storage, read> surfels: array<Surfel>;
@group(0) @binding(3) var<storage, read> gridOffsets: array<u32>;
@group(0) @binding(4) var<storage, read> gridItemList: array<u32>;

@group(0) @binding(5) var gbufferDepth: texture_depth_2d;
@group(0) @binding(6) var gbufferNormal: texture_2d<f32>;
@group(0) @binding(7) var gbufferPosition: texture_2d<f32>;
@group(0) @binding(8) var outputTex: texture_storage_2d<rgba16float, write>;

fn getCellIndex(worldPos: vec3f) -> i32 {
    let extents = constants.gridMax - constants.gridMin;
    let local = worldPos - constants.gridMin;
    if (any(local < vec3f(0.0)) || any(local >= extents)) { return -1; }
    
    let cx = clamp(u32((local.x / extents.x) * f32(constants.cellsX)), 0u, constants.cellsX - 1u);
    let cy = clamp(u32((local.y / extents.y) * f32(constants.cellsY)), 0u, constants.cellsY - 1u);
    let cz = clamp(u32((local.z / extents.z) * f32(constants.cellsZ)), 0u, constants.cellsZ - 1u);
    
    return i32(cz * constants.cellsX * constants.cellsY + cy * constants.cellsX + cx);
}

@compute @workgroup_size(8, 8, 1)
fn resolveMain(@builtin(global_invocation_id) global_id: vec3u) {
    let dim = textureDimensions(gbufferDepth);
    if (global_id.x >= dim.x || global_id.y >= dim.y) { return; }
    
    let fragcoordi = vec2i(global_id.xy);
    let pos_world = textureLoad(gbufferPosition, fragcoordi, 0).xyz;
    let nor_world = textureLoad(gbufferNormal, fragcoordi, 0).xyz;
    
    let cellObj = getCellIndex(pos_world);
    var resolvedColor = vec3f(0.0);
    
    if (cellObj >= 0) {
        let maxCells = constants.cellsX * constants.cellsY * constants.cellsZ;
        var totalWeight = 0.0;
        var totalIrradiance = vec3f(0.0);
        
        let cellCenterObj = (pos_world - constants.gridMin) / (constants.gridMax - constants.gridMin) * vec3f(f32(constants.cellsX), f32(constants.cellsY), f32(constants.cellsZ));
        
        for (var dz = -1; dz <= 1; dz++) {
            for (var dy = -1; dy <= 1; dy++) {
                for (var dx = -1; dx <= 1; dx++) {
                    let nx = i32(cellCenterObj.x) + dx;
                    let ny = i32(cellCenterObj.y) + dy;
                    let nz = i32(cellCenterObj.z) + dz;
                    
                    if (nx < 0 || ny < 0 || nz < 0 || nx >= i32(constants.cellsX) || ny >= i32(constants.cellsY) || nz >= i32(constants.cellsZ)) { continue; }
                    let ncidx = u32(nz * i32(constants.cellsX * constants.cellsY) + ny * i32(constants.cellsX) + nx);
                    
                    let offset = gridOffsets[ncidx];
                    let nextOffset = gridOffsets[ncidx + 1u];
                    var count = nextOffset - offset;
                    if (count > 8u) { count = 8u; } // Cap per cell limit (to maintain 60FPS)
                    
                    for (var i = 0u; i < count; i++) {
                        let surfelIdx = gridItemList[offset + i];
                        let surfel = surfels[surfelIdx];
                        
                        let dist = distance(surfel.position, pos_world);
                        let normalWeight = max(dot(surfel.normal, nor_world), 0.0);
                        let powerNormal = pow(normalWeight, 8.0); // Tighter normal falloff
                        
                        // Basic weighting: distance (gaussian-like) * orientation * age
                        if (dist < surfel.radius && normalWeight > 0.0) {
                            let distWeight = max(1.0 - (dist / surfel.radius), 0.0);
                            let ageWeight = clamp(surfel.age / 10.0, 0.01, 1.0); // fade in newborns
                            
                            let weight = distWeight * distWeight * powerNormal * ageWeight;
                            
                            if (weight > 0.001) {
                                totalIrradiance += surfel.irradiance * weight;
                                totalWeight += weight;
                            }
                        }
                    }
                }
            }
        }
        
        if (totalWeight > 0.0) {
            // Smooth fade out at the edges instead of hard cutoff
            // Use max(totalWeight, 1.0) to prevent multiplying back up to full intensity
            // when coverage is low
            resolvedColor = totalIrradiance / max(totalWeight, 1.0);
        }
    }
    
    textureStore(outputTex, fragcoordi, vec4f(resolvedColor, 1.0));
}

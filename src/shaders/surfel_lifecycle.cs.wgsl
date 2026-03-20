// No includes needed, TS handles it
@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<uniform> constants: SurfelGridConstants;
@group(0) @binding(2) var<storage, read_write> surfels: array<Surfel>;
@group(0) @binding(3) var gbufferDepth: texture_depth_2d;
@group(0) @binding(4) var gbufferNormal: texture_2d<f32>;

struct Allocator { count: atomic<u32> }
@group(0) @binding(5) var<storage, read_write> surfelAllocator: Allocator;
@group(0) @binding(6) var gbufferPosition: texture_2d<f32>;
@group(0) @binding(7) var<storage, read> gridOffsets: array<u32>;
struct Counter { count: atomic<u32> }
@group(0) @binding(8) var<storage, read_write> spawnCounters: array<Counter>;

fn getCellIndex(pos: vec3f) -> i32 {
    let bboxMin = constants.gridMin;
    let bboxMax = constants.gridMax;
    if (pos.x < bboxMin.x || pos.y < bboxMin.y || pos.z < bboxMin.z ||
        pos.x > bboxMax.x || pos.y > bboxMax.y || pos.z > bboxMax.z) {
        return -1;
    }
    let cellObj = (pos - bboxMin) / (bboxMax - bboxMin) * vec3f(f32(constants.cellsX), f32(constants.cellsY), f32(constants.cellsZ));
    let celli = vec3u(clamp(cellObj, vec3f(0.0), vec3f(f32(constants.cellsX - 1), f32(constants.cellsY - 1), f32(constants.cellsZ - 1))));
    return i32(celli.z * constants.cellsX * constants.cellsY + celli.y * constants.cellsX + celli.x);
}

// Pipeline 1: findMissing
@compute @workgroup_size(8, 8, 1)
fn findMissing(@builtin(global_invocation_id) global_id: vec3u) {
    let dim = textureDimensions(gbufferDepth);
    if (global_id.x >= dim.x || global_id.y >= dim.y) { return; }
    
    if ((global_id.x % 64u != 0u) || (global_id.y % 64u != 0u)) { return; } // Very sparse spawn
    
    let fragcoordi = vec2i(global_id.xy);
    let depth = textureLoad(gbufferDepth, fragcoordi, 0);
    if (depth >= 1.0) { return; } // sky
    
    let world_pos = textureLoad(gbufferPosition, fragcoordi, 0).xyz;
    let normal = textureLoad(gbufferNormal, fragcoordi, 0).xyz;
    if (length(normal) < 0.1) { return; }
    
    // Check if cell already has surfels
    let cidx = getCellIndex(world_pos);
    if (cidx >= 0) {
        let count = gridOffsets[u32(cidx) + 1u] - gridOffsets[u32(cidx)];
        let newly_spawned = atomicAdd(&spawnCounters[u32(cidx)].count, 1u);
        // If this cell already has 2 surfels (including just spawned by sibling threads), abort!
        if (count + newly_spawned >= 2u) { return; }
    }
    
    let s_idx = atomicAdd(&surfelAllocator.count, 1u) % constants.maxSurfels;
    surfels[s_idx].position = world_pos;
    surfels[s_idx].normal = normalize(normal);
    surfels[s_idx].radius = 2.0; 
    surfels[s_idx].age = 1.0;
    surfels[s_idx].irradiance = vec3f(0.0);
    surfels[s_idx].variance = 1.0;
}

// Pipeline 2: allocateCandidates (unused for MVP)
@compute @workgroup_size(1, 1, 1)
fn allocateCandidates(@builtin(global_invocation_id) global_id: vec3u) {
}

// Pipeline 3: ageSurfels
@compute @workgroup_size(64, 1, 1)
fn ageSurfels(@builtin(global_invocation_id) global_id: vec3u) {
    let idx = global_id.x;
    if (idx >= constants.maxSurfels) { return; }
    
    let age = surfels[idx].age;
    if (age > 0.0) {
        surfels[idx].age += 1.0;
    }
}

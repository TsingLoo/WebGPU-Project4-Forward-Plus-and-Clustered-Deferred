// No includes needed, TS handles it
@group(0) @binding(0) var<uniform> constants: SurfelGridConstants;
@group(0) @binding(1) var<storage, read> surfels: array<Surfel>;
struct Counter { count: atomic<u32> }
@group(0) @binding(2) var<storage, read_write> gridCounters: array<Counter>;
@group(0) @binding(3) var<storage, read_write> gridOffsets: array<u32>;
@group(0) @binding(4) var<storage, read_write> gridItemList: array<u32>;

fn getCellIndex(worldPos: vec3f) -> i32 {
    let extents = constants.gridMax - constants.gridMin;
    let local = worldPos - constants.gridMin;
    if (any(local < vec3f(0.0)) || any(local >= extents)) { return -1; }
    
    // Simplification for grid computation
    let cx = u32((local.x / extents.x) * f32(constants.cellsX));
    let cy = u32((local.y / extents.y) * f32(constants.cellsY));
    let cz = u32((local.z / extents.z) * f32(constants.cellsZ));
    
    return i32(cz * constants.cellsX * constants.cellsY + cy * constants.cellsX + cx);
}

// Pipeline 1: Clear Counters
@compute @workgroup_size(64, 1, 1)
fn clearCounters(@builtin(global_invocation_id) global_id: vec3u) {
    let cellIdx = global_id.x;
    let totalCells = constants.cellsX * constants.cellsY * constants.cellsZ;
    if (cellIdx < totalCells) {
        atomicStore(&gridCounters[cellIdx].count, 0u);
    }
}

// Pipeline 2: Count Surfels
@compute @workgroup_size(64, 1, 1)
fn countSurfels(@builtin(global_invocation_id) global_id: vec3u) {
    let idx = global_id.x;
    if (idx >= constants.maxSurfels) { return; }
    
    let surfel = surfels[idx];
    if (surfel.age == 0.0) { return; } // dead
    
    let cellObj = getCellIndex(surfel.position);
    if (cellObj >= 0) {
        // Here we could also add it to adjacent cells (hysteresis/radius).
        atomicAdd(&gridCounters[u32(cellObj)].count, 1u);
    }
}

// Pipeline 3: Exclusive prefix sum (scan) 
// For simplicity, doing a single pass scan if cell count is small, or multi-pass.
@compute @workgroup_size(1, 1, 1)
fn prefixSum() {
    // Note: extremely naive O(N) single-workgroup scan for prototyping.
    // Production would use tiered scan.
    let totalCells = constants.cellsX * constants.cellsY * constants.cellsZ;
    var sum = 0u;
    for (var i = 0u; i < totalCells; i++) {
        gridOffsets[i] = sum;
        sum += atomicLoad(&gridCounters[i].count);
    }
    gridOffsets[totalCells] = sum;
}

// Pipeline 4: Slot Surfels into List
@compute @workgroup_size(64, 1, 1)
fn slotSurfels(@builtin(global_invocation_id) global_id: vec3u) {
    let idx = global_id.x;
    if (idx >= constants.maxSurfels) { return; }
    
    let surfel = surfels[idx];
    if (surfel.age == 0.0) { return; } 
    
    let cellObj = getCellIndex(surfel.position);
    if (cellObj >= 0) {
        // Atomic decrement trick for writing into prefix-sumed contiguous ranges
        let offset = gridOffsets[u32(cellObj)];
        let countInCellThisThreadGot = atomicSub(&gridCounters[u32(cellObj)].count, 1u) - 1u;
        
        gridItemList[offset + countInCellThisThreadGot] = idx;
    }
}

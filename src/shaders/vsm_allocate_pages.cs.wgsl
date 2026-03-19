// VSM Page Allocation Compute Shader
// Scans marked pages and assigns physical pages from a free pool.
// Each thread handles one virtual page.

struct AllocationState {
    counter: atomic<u32>,  // global counter for physical page allocation
    total_requested: atomic<u32>,  // debug: how many pages requested
}

@group(0) @binding(0) var<storage, read> pageRequestFlags: array<u32>;
@group(0) @binding(1) var<storage, read_write> pageTable: array<u32>;
@group(0) @binding(2) var<storage, read_write> allocState: AllocationState;
@group(0) @binding(3) var<uniform> vsmUniforms: VSMUniforms;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let pagesPerAxis = vsmUniforms.pages_per_axis;
    let numLevels = vsmUniforms.clipmap_count;
    let totalVirtualPages = numLevels * pagesPerAxis * pagesPerAxis;
    let maxPhysicalPages = vsmUniforms.phys_pages_per_axis * vsmUniforms.phys_pages_per_axis;

    if (gid.x >= totalVirtualPages) {
        return;
    }

    let pageIdx = gid.x;

    if (pageRequestFlags[pageIdx] > 0u) {
        // Atomically claim a physical page
        let physIdx = atomicAdd(&allocState.counter, 1u);
        atomicAdd(&allocState.total_requested, 1u);

        if (physIdx < maxPhysicalPages) {
            // Map virtual page to physical page index
            pageTable[pageIdx] = physIdx;
        } else {
            // Ran out of physical pages — mark as invalid
            pageTable[pageIdx] = 0xFFFFFFFFu;
        }
    } else {
        pageTable[pageIdx] = 0xFFFFFFFFu;
    }
}

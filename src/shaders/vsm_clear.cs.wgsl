// VSM Clear Compute Shader
// Resets page request flags and allocation state each frame.

struct AllocationState {
    counter: atomic<u32>,
    total_requested: atomic<u32>,
}

@group(0) @binding(0) var<storage, read_write> pageRequestFlags: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> allocState: AllocationState;
@group(0) @binding(2) var<storage, read_write> pageTable: array<u32>;
@group(0) @binding(3) var<uniform> vsmUniforms: VSMUniforms;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let pagesPerAxis = vsmUniforms.pages_per_axis;
    let numLevels = vsmUniforms.clipmap_count;
    let totalVirtualPages = numLevels * pagesPerAxis * pagesPerAxis;

    if (gid.x >= totalVirtualPages) {
        return;
    }

    atomicStore(&pageRequestFlags[gid.x], 0u);
    pageTable[gid.x] = 0xFFFFFFFFu;

    // Reset allocation state (only thread 0)
    if (gid.x == 0u) {
        atomicStore(&allocState.counter, 0u);
        atomicStore(&allocState.total_requested, 0u);
    }
}

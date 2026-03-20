import { device } from '../renderer';
import * as shaders from '../shaders/shaders';
import { Camera } from './camera';
import { Scene } from './scene';

export class SurfelGI {
    enabled = false; // Disabled by default to avoid conflict with DDGI
    debugMode = false;
    
    // Limits
    static readonly MAX_SURFELS = 65536;
    static readonly GRID_CELLS_X = 16;
    static readonly GRID_CELLS_Y = 16;
    static readonly GRID_CELLS_Z = 16;
    static readonly TOTAL_CELLS = SurfelGI.GRID_CELLS_X * SurfelGI.GRID_CELLS_Y * SurfelGI.GRID_CELLS_Z;
    static readonly RAYS_PER_SURFEL = 8;

    gridMin: [number, number, number] = [-15, -4, -15];
    gridMax: [number, number, number] = [15, 16, 15];

    // Buffers
    surfelsBuffer: GPUBuffer;
    gridCountersBuffer: GPUBuffer;
    gridOffsetsBuffer: GPUBuffer;
    gridItemListBuffer: GPUBuffer;
    
    constantsBuffer: GPUBuffer;
    randomBuffer: GPUBuffer;
    surfelAllocatorBuffer: GPUBuffer;
    
    // Pipelines
    findMissingPipeline!: GPUComputePipeline;
    allocatePipeline!: GPUComputePipeline;
    agePipeline!: GPUComputePipeline;
    
    clearCountersPipeline!: GPUComputePipeline;
    countSurfelsPipeline!: GPUComputePipeline;
    prefixSumPipeline!: GPUComputePipeline;
    slotSurfelsPipeline!: GPUComputePipeline;
    
    integratorPipeline!: GPUComputePipeline;
    resolvePipeline!: GPUComputePipeline;

    // Layouts
    lifecycleLayout!: GPUBindGroupLayout;
    gridLayout!: GPUBindGroupLayout;
    integratorLayout!: GPUBindGroupLayout;
    resolveLayout!: GPUBindGroupLayout;
    sunLayout!: GPUBindGroupLayout;

    private camera: Camera;
    private frameCount: number = 0;

    constructor(camera: Camera) {
        this.camera = camera;

        // 56 bytes per surfel. Let's round to 64 bytes for WGSL strict alignment if needed.
        // float3 position, float radius (16)
        // float3 normal, float age (16)
        // float3 irradiance, float variance (16)
        // float3 shortMean, float pGuide (16) => Total 64 bytes
        this.surfelsBuffer = device.createBuffer({
            label: "Surfels Buffer",
            size: SurfelGI.MAX_SURFELS * 64, 
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        this.gridCountersBuffer = device.createBuffer({
            label: "Surfel Grid Counters",
            size: SurfelGI.TOTAL_CELLS * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        this.gridOffsetsBuffer = device.createBuffer({
            label: "Surfel Grid Offsets",
            size: (SurfelGI.TOTAL_CELLS + 1) * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        this.gridItemListBuffer = device.createBuffer({
            label: "Surfel Grid Item List",
            size: SurfelGI.MAX_SURFELS * 4, // Average 1 per surfel in grid
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        
        // SurfelGridConstants struct: 48 bytes 
        this.constantsBuffer = device.createBuffer({
            label: "Surfel Constants",
            size: 64, // Padded
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.randomBuffer = device.createBuffer({
            label: "Surfel Randoms",
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.surfelAllocatorBuffer = device.createBuffer({
            label: "Surfel Allocator",
            size: 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        this.initPipelines();
        console.log(`SurfelGI initialized: max ${SurfelGI.MAX_SURFELS} surfels`);
    }

    private createShader(device: GPUDevice, code: string, label: string) {
        const module = device.createShaderModule({ label, code });
        module.getCompilationInfo().then((info: GPUCompilationInfo) => {
            let hasError = false;
            for (let m of info.messages) {
                if (m.type === 'error') {
                    console.error(`[Shader Error] ${label} line ${m.lineNum}: ${m.message}`);
                    hasError = true;
                }
            }
            if (hasError) {
                console.error(`--- ${label} Source ---`);
                console.error(code.split('\\n').map((l,i) => `${i+1}: ${l}`).join('\\n'));
            }
        });
        return module;
    }

    private initPipelines() {
        // Lifecycle layout
        this.lifecycleLayout = device.createBindGroupLayout({
            label: "Surfel Lifecycle Layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'depth' } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 6, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
                { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            ]
        });

        // Grid layout
        this.gridLayout = device.createBindGroupLayout({
            label: "Surfel Grid Layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            ]
        });

        // Integrator layout (Group 0: common, Group 1: BVH, Group 2: Env, Group 3: Random)
        this.integratorLayout = device.createBindGroupLayout({
            label: "Surfel Integrator Layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'depth' } },
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            ]
        });

        // Group 1: BVH
        const bvhLayout = device.createBindGroupLayout({
            label: "BVH Layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // nodes
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // pos
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // indices
            ]
        });
        
        // Group 2: Env
        const envLayout = device.createBindGroupLayout({
            label: "Env Layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { viewDimension: 'cube' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, sampler: {} },
            ]
        });

        // Group 3: Random
        const randLayout = device.createBindGroupLayout({
            label: "Rand Layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            ]
        });

        // Group removed (merged to integratorLayout)
        
        this.integratorPipeline = device.createComputePipeline({
            label: "Surfel Integrator Pipeline",
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.integratorLayout, bvhLayout, envLayout, randLayout] }),
            compute: { module: this.createShader(device, shaders.surfelIntegratorSrc, "Integrator"), entryPoint: 'integratorMain' }
        });

        // Resolve layout
        this.resolveLayout = device.createBindGroupLayout({
            label: "Surfel Resolve Layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 5, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'depth' } }, // depth
                { binding: 6, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } }, // normal
                { binding: 7, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } }, // pos
                { binding: 8, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } }, // output
            ]
        });
        
        this.resolvePipeline = device.createComputePipeline({
            label: "Surfel Resolve Pipeline",
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.resolveLayout] }),
            compute: { module: this.createShader(device, shaders.surfelResolveSrc, "Resolve"), entryPoint: 'resolveMain' }
        });
        
        // Let's instantiate grid shaders lazily or directly
        this.clearCountersPipeline = device.createComputePipeline({
            label: "Surfel Grid Clear Pipeline",
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.gridLayout] }),
            compute: { module: this.createShader(device, shaders.surfelGridSrc, "Grid"), entryPoint: 'clearCounters' }
        });
        
        this.countSurfelsPipeline = device.createComputePipeline({
            label: "Surfel Grid Count Pipeline",
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.gridLayout] }),
            compute: { module: this.createShader(device, shaders.surfelGridSrc, "Grid"), entryPoint: 'countSurfels' }
        });
        
        this.prefixSumPipeline = device.createComputePipeline({
            label: "Surfel Grid Prefix Sum Pipeline",
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.gridLayout] }),
            compute: { module: this.createShader(device, shaders.surfelGridSrc, "Grid"), entryPoint: 'prefixSum' }
        });
        
        this.slotSurfelsPipeline = device.createComputePipeline({
            label: "Surfel Grid Slot Pipeline",
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.gridLayout] }),
            compute: { module: this.createShader(device, shaders.surfelGridSrc, "Grid"), entryPoint: 'slotSurfels' }
        });

        // Lifecycle pipelines
        this.findMissingPipeline = device.createComputePipeline({
            label: "Surfel Lifecycle Find Missing Pipeline",
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.lifecycleLayout] }),
            compute: { module: this.createShader(device, shaders.surfelLifecycleSrc, "Lifecycle"), entryPoint: 'findMissing' }
        });

        this.agePipeline = device.createComputePipeline({
            label: "Surfel Lifecycle Age Pipeline",
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.lifecycleLayout] }),
            compute: { module: this.createShader(device, shaders.surfelLifecycleSrc, "Lifecycle"), entryPoint: 'ageSurfels' }
        });
    }

    updateUniforms() {
        const data = new ArrayBuffer(64);
        const f32View = new Float32Array(data);
        const u32View = new Uint32Array(data);

        f32View[0] = this.gridMin[0]; f32View[1] = this.gridMin[1]; f32View[2] = this.gridMin[2]; 
        f32View[3] = 0; // pad0

        f32View[4] = this.gridMax[0]; f32View[5] = this.gridMax[1]; f32View[6] = this.gridMax[2]; 
        f32View[7] = 0; // pad1
        
        u32View[8]  = SurfelGI.GRID_CELLS_X;
        u32View[9]  = SurfelGI.GRID_CELLS_Y;
        u32View[10] = SurfelGI.GRID_CELLS_Z;
        u32View[11] = 64; // maxSurfelsPerCell (approx allowed)
        
        u32View[12] = SurfelGI.MAX_SURFELS;
        u32View[13] = 0; // allocatedCount
        u32View[14] = SurfelGI.RAYS_PER_SURFEL;
        f32View[15] = 0; // pad2

        device.queue.writeBuffer(this.constantsBuffer, 0, data);
        
        // Random buffer
        this.frameCount++;
        const randData = new Uint32Array(4);
        randData[0] = Math.floor(Math.random() * 0xffffffff);
        randData[1] = this.frameCount;
        device.queue.writeBuffer(this.randomBuffer, 0, randData);
    }

    update(encoder: GPUCommandEncoder, scene: Scene, envCubemapView: GPUTextureView, envSampler: GPUSampler, depthView: GPUTextureView, normalView: GPUTextureView, posView: GPUTextureView, outputView: GPUTextureView, sunBuffer: GPUBuffer, vsmAtlasView: GPUTextureView, vsmUniformBuffer: GPUBuffer) {
        if (!this.enabled || !scene.bvhData) return;
        this.updateUniforms();

        // 0. Lifecycle (Find missing and Age)
        const lifecycleGroup = device.createBindGroup({
            layout: this.lifecycleLayout,
            entries: [
                { binding: 0, resource: { buffer: this.camera.uniformsBuffer } },
                { binding: 1, resource: { buffer: this.constantsBuffer } },
                { binding: 2, resource: { buffer: this.surfelsBuffer } },
                { binding: 3, resource: depthView },
                { binding: 4, resource: normalView },
                { binding: 5, resource: { buffer: this.surfelAllocatorBuffer } },
                { binding: 6, resource: posView },
                { binding: 7, resource: { buffer: this.gridOffsetsBuffer } },
                { binding: 8, resource: { buffer: this.gridCountersBuffer } },
            ]
        });

        const lifecyclePass = encoder.beginComputePass({ label: "Surfel Lifecycle" });
        lifecyclePass.setBindGroup(0, lifecycleGroup);
        // Find Missing
        lifecyclePass.setPipeline(this.findMissingPipeline);
        lifecyclePass.dispatchWorkgroups(Math.ceil(2560 / 8), Math.ceil(1440 / 8));
        // Age Surfels
        lifecyclePass.setPipeline(this.agePipeline);
        lifecyclePass.dispatchWorkgroups(Math.ceil(SurfelGI.MAX_SURFELS / 64));
        lifecyclePass.end();

        // 1. Grid Build 
        const gridGroup = device.createBindGroup({
            layout: this.gridLayout,
            entries: [
                { binding: 0, resource: { buffer: this.constantsBuffer } },
                { binding: 1, resource: { buffer: this.surfelsBuffer } },
                { binding: 2, resource: { buffer: this.gridCountersBuffer } },
                { binding: 3, resource: { buffer: this.gridOffsetsBuffer } },
                { binding: 4, resource: { buffer: this.gridItemListBuffer } },
            ]
        });

        const gridPass = encoder.beginComputePass({ label: "Surfel Grid Build" });
        gridPass.setBindGroup(0, gridGroup);
        // Clear counters
        gridPass.setPipeline(this.clearCountersPipeline);
        gridPass.dispatchWorkgroups(Math.ceil(SurfelGI.TOTAL_CELLS / 64));
        
        // Count surfels
        gridPass.setPipeline(this.countSurfelsPipeline);
        gridPass.dispatchWorkgroups(Math.ceil(SurfelGI.MAX_SURFELS / 64));
        
        // Prefix sum
        gridPass.setPipeline(this.prefixSumPipeline);
        gridPass.dispatchWorkgroups(1);
        
        // Slot
        gridPass.setPipeline(this.slotSurfelsPipeline);
        gridPass.dispatchWorkgroups(Math.ceil(SurfelGI.MAX_SURFELS / 64));
        gridPass.end();

        // 2. Integrator (Ray Tracing)
        const commonGroup = device.createBindGroup({
            layout: this.integratorLayout,
            entries: [
                { binding: 0, resource: { buffer: this.camera.uniformsBuffer } },
                { binding: 1, resource: { buffer: this.constantsBuffer } },
                { binding: 2, resource: { buffer: this.surfelsBuffer } },
                { binding: 3, resource: { buffer: sunBuffer } },
                { binding: 4, resource: vsmAtlasView },
                { binding: 5, resource: { buffer: vsmUniformBuffer } }
            ]
        });
        const bvhGroup = device.createBindGroup({
            layout: this.integratorPipeline.getBindGroupLayout(1),
            entries: [
                { binding: 0, resource: { buffer: scene.bvhData.nodeBuffer } },
                { binding: 1, resource: { buffer: scene.bvhData.positionBuffer } },
                { binding: 2, resource: { buffer: scene.bvhData.indexBuffer } },
            ]
        });
        const envGroup = device.createBindGroup({
            layout: this.integratorPipeline.getBindGroupLayout(2),
            entries: [
                { binding: 0, resource: envCubemapView },
                { binding: 1, resource: envSampler },
            ]
        });
        const randGroup = device.createBindGroup({
            layout: this.integratorPipeline.getBindGroupLayout(3),
            entries: [
                { binding: 0, resource: { buffer: this.randomBuffer } },
            ]
        });

        // sunGroup merged

        const intPass = encoder.beginComputePass({ label: "Surfel Integrator" });
        intPass.setPipeline(this.integratorPipeline);
        intPass.setBindGroup(0, commonGroup);
        intPass.setBindGroup(1, bvhGroup);
        intPass.setBindGroup(2, envGroup);
        intPass.setBindGroup(3, randGroup);
        // intPass.setBindGroup(4, sunGroup); // Merged into 0
        intPass.dispatchWorkgroups(Math.ceil(SurfelGI.MAX_SURFELS / 64));
        intPass.end();

        // 3. Resolve to screen
        // (Assuming Screen dimensions from depthView size)
        // We'll read G-buffer and write to outputView (rgba16float).
        const resGroup = device.createBindGroup({
            layout: this.resolveLayout,
            entries: [
                { binding: 0, resource: { buffer: this.camera.uniformsBuffer } },
                { binding: 1, resource: { buffer: this.constantsBuffer } },
                { binding: 2, resource: { buffer: this.surfelsBuffer } },
                { binding: 3, resource: { buffer: this.gridOffsetsBuffer } },
                { binding: 4, resource: { buffer: this.gridItemListBuffer } },
                { binding: 5, resource: depthView },
                { binding: 6, resource: normalView },
                { binding: 7, resource: posView },
                { binding: 8, resource: outputView },
            ]
        });

        const resPass = encoder.beginComputePass({ label: "Surfel Resolve" });
        resPass.setPipeline(this.resolvePipeline);
        resPass.setBindGroup(0, resGroup);
        // Dispatch over screen dimensions manually from stage in real life, or approx: 
        // using constants.screen_width. For now assume fixed dimensions or derive:
        // We typically dispatch (width/8, height/8) 
        // But since we can't easily get texture dims here sync without storing them, 
        // we'll fetch them from the camera or pass them. 
        // Assuming 1920x1080 max for now:
        resPass.dispatchWorkgroups(Math.ceil(2560/8), Math.ceil(1440/8)); 
        resPass.end();
    }
}

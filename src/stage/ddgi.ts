import { device } from '../renderer';
import * as shaders from '../shaders/shaders';
import { Camera } from './camera';
import { Environment } from './environment';

/**
 * DDGI (Dynamic Diffuse Global Illumination) manager.
 * Manages probe grid, atlas textures, and compute pipelines for
 * screen-space probe-based irradiance and visibility updates.
 */
export class DDGI {
    // Grid configuration
    static readonly GRID_X = shaders.constants.ddgiProbeGridX;
    static readonly GRID_Y = shaders.constants.ddgiProbeGridY;
    static readonly GRID_Z = shaders.constants.ddgiProbeGridZ;
    static readonly TOTAL_PROBES = DDGI.GRID_X * DDGI.GRID_Y * DDGI.GRID_Z;
    static readonly RAYS_PER_PROBE = shaders.constants.ddgiRaysPerProbe;

    static readonly IRRADIANCE_TEXELS = shaders.constants.ddgiIrradianceTexels;   // 8
    static readonly VISIBILITY_TEXELS = shaders.constants.ddgiVisibilityTexels;  // 16
    static readonly IRRADIANCE_WITH_BORDER = DDGI.IRRADIANCE_TEXELS + 2;  // 10
    static readonly VISIBILITY_WITH_BORDER = DDGI.VISIBILITY_TEXELS + 2;  // 18

    // Atlas dimensions
    static readonly IRR_ATLAS_W = DDGI.GRID_X * DDGI.IRRADIANCE_WITH_BORDER;
    static readonly IRR_ATLAS_H = (DDGI.GRID_Y * DDGI.GRID_Z) * DDGI.IRRADIANCE_WITH_BORDER;
    static readonly VIS_ATLAS_W = DDGI.GRID_X * DDGI.VISIBILITY_WITH_BORDER;
    static readonly VIS_ATLAS_H = (DDGI.GRID_Y * DDGI.GRID_Z) * DDGI.VISIBILITY_WITH_BORDER;

    // World-space bounds (Sponza defaults)
    gridMin: [number, number, number] = [-14, 0, -7];
    gridMax: [number, number, number] = [14, 12, 7];

    // Hysteresis
    irradianceHysteresis = 0.95;
    visibilityHysteresis = 0.97;
    normalBias = 0.25;
    viewBias = 0.1;
    probeTraceAmbient = 0.3;

    enabled = true;
    debugMode = 0; // 0=off, 1=irradiance, 2=visibility

    // GPU resources
    irradianceAtlasA: GPUTexture;
    irradianceAtlasB: GPUTexture;
    visibilityAtlasA: GPUTexture;
    visibilityAtlasB: GPUTexture;

    irradianceAtlasAView: GPUTextureView;
    irradianceAtlasBView: GPUTextureView;
    visibilityAtlasAView: GPUTextureView;
    visibilityAtlasBView: GPUTextureView;

    // Current read/write targets (ping-pong)
    private pingPong = 0;

    rayDataBuffer: GPUBuffer;
    ddgiUniformBuffer: GPUBuffer;
    randomRotationBuffer: GPUBuffer;

    ddgiSampler: GPUSampler;

    // Pipelines
    probeTracePipeline: GPUComputePipeline;
    irradianceUpdatePipeline: GPUComputePipeline;
    visibilityUpdatePipeline: GPUComputePipeline;
    borderUpdateIrrPipeline: GPUComputePipeline;
    borderUpdateVisPipeline: GPUComputePipeline;

    // Bind group layouts
    probeTraceLayout: GPUBindGroupLayout;
    irradianceUpdateLayout: GPUBindGroupLayout;
    visibilityUpdateLayout: GPUBindGroupLayout;
    borderUpdateLayout: GPUBindGroupLayout;

    // Border params
    borderIrrParamsBuffer: GPUBuffer;
    borderVisParamsBuffer: GPUBuffer;

    private camera: Camera;
    private environment: Environment;

    constructor(camera: Camera, environment: Environment) {
        this.camera = camera;
        this.environment = environment;

        // Create irradiance atlas textures (ping-pong pair)
        const irrAtlasDesc: GPUTextureDescriptor = {
            label: "DDGI Irradiance Atlas",
            size: [DDGI.IRR_ATLAS_W, DDGI.IRR_ATLAS_H],
            format: 'rgba16float',
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
        };
        this.irradianceAtlasA = device.createTexture({ ...irrAtlasDesc, label: "DDGI Irradiance Atlas A" });
        this.irradianceAtlasB = device.createTexture({ ...irrAtlasDesc, label: "DDGI Irradiance Atlas B" });
        this.irradianceAtlasAView = this.irradianceAtlasA.createView();
        this.irradianceAtlasBView = this.irradianceAtlasB.createView();

        // Create visibility atlas textures (ping-pong pair)
        const visAtlasDesc: GPUTextureDescriptor = {
            label: "DDGI Visibility Atlas",
            size: [DDGI.VIS_ATLAS_W, DDGI.VIS_ATLAS_H],
            format: 'rgba16float',
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
        };
        this.visibilityAtlasA = device.createTexture({ ...visAtlasDesc, label: "DDGI Visibility Atlas A" });
        this.visibilityAtlasB = device.createTexture({ ...visAtlasDesc, label: "DDGI Visibility Atlas B" });
        this.visibilityAtlasAView = this.visibilityAtlasA.createView();
        this.visibilityAtlasBView = this.visibilityAtlasB.createView();

        // Ray data buffer: vec4f per ray per probe
        const rayDataSize = DDGI.TOTAL_PROBES * DDGI.RAYS_PER_PROBE * 16; // 4 floats * 4 bytes
        this.rayDataBuffer = device.createBuffer({
            label: "DDGI Ray Data",
            size: rayDataSize,
            usage: GPUBufferUsage.STORAGE,
        });

        // DDGI uniform buffer (DDGIUniforms struct: 8 vec4 = 128 bytes)
        this.ddgiUniformBuffer = device.createBuffer({
            label: "DDGI Uniforms",
            size: 128,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Random rotation matrix buffer (mat4x4f = 64 bytes)
        this.randomRotationBuffer = device.createBuffer({
            label: "DDGI Random Rotation",
            size: 64,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Sampler for atlas sampling
        this.ddgiSampler = device.createSampler({
            label: "DDGI Atlas Sampler",
            magFilter: 'linear',
            minFilter: 'linear',
            addressModeU: 'clamp-to-edge',
            addressModeV: 'clamp-to-edge',
        });

        // Border param buffers
        this.borderIrrParamsBuffer = device.createBuffer({
            label: "DDGI Border Irradiance Params",
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(this.borderIrrParamsBuffer, 0, new Uint32Array([
            DDGI.IRRADIANCE_TEXELS, DDGI.IRRADIANCE_WITH_BORDER, DDGI.GRID_X, DDGI.TOTAL_PROBES
        ]));

        this.borderVisParamsBuffer = device.createBuffer({
            label: "DDGI Border Visibility Params",
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(this.borderVisParamsBuffer, 0, new Uint32Array([
            DDGI.VISIBILITY_TEXELS, DDGI.VISIBILITY_WITH_BORDER, DDGI.GRID_X, DDGI.TOTAL_PROBES
        ]));

        this.updateUniforms();
        this.updateRandomRotation();  // Initialize with a valid rotation

        // Create pipelines
        this.probeTraceLayout = this.createProbeTraceLayout();
        this.irradianceUpdateLayout = this.createIrradianceUpdateLayout();
        this.visibilityUpdateLayout = this.createVisibilityUpdateLayout();
        this.borderUpdateLayout = this.createBorderUpdateLayout();

        this.probeTracePipeline = device.createComputePipeline({
            label: "DDGI Probe Trace Pipeline",
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.probeTraceLayout] }),
            compute: {
                module: device.createShaderModule({ label: "DDGI Probe Trace", code: shaders.ddgiProbeTraceSrc }),
                entryPoint: 'main'
            }
        });

        this.irradianceUpdatePipeline = device.createComputePipeline({
            label: "DDGI Irradiance Update Pipeline",
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.irradianceUpdateLayout] }),
            compute: {
                module: device.createShaderModule({ label: "DDGI Irradiance Update", code: shaders.ddgiIrradianceUpdateSrc }),
                entryPoint: 'main'
            }
        });

        this.visibilityUpdatePipeline = device.createComputePipeline({
            label: "DDGI Visibility Update Pipeline",
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.visibilityUpdateLayout] }),
            compute: {
                module: device.createShaderModule({ label: "DDGI Visibility Update", code: shaders.ddgiVisibilityUpdateSrc }),
                entryPoint: 'main'
            }
        });

        this.borderUpdateIrrPipeline = device.createComputePipeline({
            label: "DDGI Border Update Irradiance Pipeline",
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.borderUpdateLayout] }),
            compute: {
                module: device.createShaderModule({ label: "DDGI Border Update Irr", code: shaders.ddgiBorderUpdateSrc }),
                entryPoint: 'main'
            }
        });

        // For visibility border, we need a separate pipeline with rg16float format
        this.borderUpdateVisPipeline = this.borderUpdateIrrPipeline; // Reuse — format difference handled via bind group

        console.log(`DDGI initialized: ${DDGI.GRID_X}x${DDGI.GRID_Y}x${DDGI.GRID_Z} = ${DDGI.TOTAL_PROBES} probes`);
        console.log(`  Irradiance atlas: ${DDGI.IRR_ATLAS_W}x${DDGI.IRR_ATLAS_H}`);
        console.log(`  Visibility atlas: ${DDGI.VIS_ATLAS_W}x${DDGI.VIS_ATLAS_H}`);
    }

    updateUniforms() {
        const spacing = [
            (this.gridMax[0] - this.gridMin[0]) / (DDGI.GRID_X - 1),
            (this.gridMax[1] - this.gridMin[1]) / (DDGI.GRID_Y - 1),
            (this.gridMax[2] - this.gridMin[2]) / (DDGI.GRID_Z - 1),
        ];

        const data = new ArrayBuffer(128);
        const i32View = new Int32Array(data);
        const f32View = new Float32Array(data);

        // grid_count: vec4i (x, y, z, total)
        i32View[0] = DDGI.GRID_X;
        i32View[1] = DDGI.GRID_Y;
        i32View[2] = DDGI.GRID_Z;
        i32View[3] = DDGI.TOTAL_PROBES;

        // grid_min: vec4f
        f32View[4] = this.gridMin[0];
        f32View[5] = this.gridMin[1];
        f32View[6] = this.gridMin[2];
        f32View[7] = 0;

        // grid_max: vec4f
        f32View[8] = this.gridMax[0];
        f32View[9] = this.gridMax[1];
        f32View[10] = this.gridMax[2];
        f32View[11] = 0;

        // grid_spacing: vec4f (spacing.xyz, raysPerProbe)
        f32View[12] = spacing[0];
        f32View[13] = spacing[1];
        f32View[14] = spacing[2];
        f32View[15] = DDGI.RAYS_PER_PROBE;

        // irradiance_texel_size: vec4f
        f32View[16] = DDGI.IRRADIANCE_TEXELS;
        f32View[17] = DDGI.IRRADIANCE_WITH_BORDER;
        f32View[18] = DDGI.IRR_ATLAS_W;
        f32View[19] = DDGI.IRR_ATLAS_H;

        // visibility_texel_size: vec4f
        f32View[20] = DDGI.VISIBILITY_TEXELS;
        f32View[21] = DDGI.VISIBILITY_WITH_BORDER;
        f32View[22] = DDGI.VIS_ATLAS_W;
        f32View[23] = DDGI.VIS_ATLAS_H;

        // hysteresis: vec4f
        f32View[24] = this.irradianceHysteresis;
        f32View[25] = this.visibilityHysteresis;
        f32View[26] = this.normalBias;
        f32View[27] = this.viewBias;

        // ddgi_enabled: vec4f
        f32View[28] = this.enabled ? 1.0 : 0.0;
        f32View[29] = this.debugMode;
        f32View[30] = this.probeTraceAmbient;
        f32View[31] = 0;

        device.queue.writeBuffer(this.ddgiUniformBuffer, 0, data);
    }

    updateRandomRotation() {
        // Generate a random rotation matrix for temporal jitter
        const angle = Math.random() * Math.PI * 2;
        const axis = [
            Math.random() * 2 - 1,
            Math.random() * 2 - 1,
            Math.random() * 2 - 1
        ];
        const len = Math.sqrt(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]);
        axis[0] /= len; axis[1] /= len; axis[2] /= len;

        const c = Math.cos(angle);
        const s = Math.sin(angle);
        const t = 1 - c;
        const x = axis[0], y = axis[1], z = axis[2];

        // Column-major rotation matrix (mat4x4)
        const rotMat = new Float32Array([
            t * x * x + c,     t * x * y + s * z, t * x * z - s * y, 0,
            t * x * y - s * z, t * y * y + c,     t * y * z + s * x, 0,
            t * x * z + s * y, t * y * z - s * x, t * z * z + c,     0,
            0, 0, 0, 1
        ]);

        device.queue.writeBuffer(this.randomRotationBuffer, 0, rotMat);
    }

    private createProbeTraceLayout(): GPUBindGroupLayout {
        return device.createBindGroupLayout({
            label: "DDGI Probe Trace Layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },    // camera
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },    // ddgi uniforms
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },    // random rotation
                { binding: 3, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'depth' } }, // depth
                { binding: 4, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } }, // normal
                { binding: 5, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } }, // albedo
                { binding: 6, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } }, // position
                { binding: 7, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float', viewDimension: 'cube' } }, // env map
                { binding: 8, visibility: GPUShaderStage.COMPUTE, sampler: {} },                     // env sampler
                { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },     // ray data
                { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },    // sun light
                { binding: 11, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'depth' } }, // VSM physical atlas
                { binding: 12, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },     // VSM uniforms
            ]
        });
    }

    private createIrradianceUpdateLayout(): GPUBindGroupLayout {
        return device.createBindGroupLayout({
            label: "DDGI Irradiance Update Layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },     // ddgi
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },     // random rotation
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // ray data
                { binding: 3, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } }, // read atlas
                { binding: 4, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } }, // write atlas
            ]
        });
    }

    private createVisibilityUpdateLayout(): GPUBindGroupLayout {
        return device.createBindGroupLayout({
            label: "DDGI Visibility Update Layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } },
            ]
        });
    }

    private createBorderUpdateLayout(): GPUBindGroupLayout {
        return device.createBindGroupLayout({
            label: "DDGI Border Update Layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } }, // source
                { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } }, // dest
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }, // params
            ]
        });
    }

    /**
     * Dispatches all DDGI compute passes: probe trace, irradiance update, visibility update, border copy.
     */
    update(
        encoder: GPUCommandEncoder,
        gBuffer: {
            depth: GPUTextureView,
            normal: GPUTextureView,
            albedo: GPUTextureView,
            position: GPUTextureView,
        },
        sunLightBuffer: GPUBuffer,
        shadowMapView: GPUTextureView,
        vsmUniformBuffer: GPUBuffer,
    ) {
        if (!this.enabled) return;

        this.updateUniforms();
        this.updateRandomRotation();

        // Determine ping-pong targets
        const readIrr = this.pingPong === 0 ? this.irradianceAtlasAView : this.irradianceAtlasBView;
        const writeIrr = this.pingPong === 0 ? this.irradianceAtlasBView : this.irradianceAtlasAView;
        const readVis = this.pingPong === 0 ? this.visibilityAtlasAView : this.visibilityAtlasBView;
        const writeVis = this.pingPong === 0 ? this.visibilityAtlasBView : this.visibilityAtlasAView;

        // 1. Probe Ray Trace
        const traceBindGroup = device.createBindGroup({
            layout: this.probeTraceLayout,
            entries: [
                { binding: 0, resource: { buffer: this.camera.uniformsBuffer } },
                { binding: 1, resource: { buffer: this.ddgiUniformBuffer } },
                { binding: 2, resource: { buffer: this.randomRotationBuffer } },
                { binding: 3, resource: gBuffer.depth },
                { binding: 4, resource: gBuffer.normal },
                { binding: 5, resource: gBuffer.albedo },
                { binding: 6, resource: gBuffer.position },
                { binding: 7, resource: this.environment.envCubemapView },
                { binding: 8, resource: this.environment.envSampler },
                { binding: 9, resource: { buffer: this.rayDataBuffer } },
                { binding: 10, resource: { buffer: sunLightBuffer } },
                { binding: 11, resource: shadowMapView },
                { binding: 12, resource: { buffer: vsmUniformBuffer } },
            ]
        });

        const tracePass = encoder.beginComputePass({ label: "DDGI Probe Trace" });
        tracePass.setPipeline(this.probeTracePipeline);
        tracePass.setBindGroup(0, traceBindGroup);
        tracePass.dispatchWorkgroups(1, DDGI.TOTAL_PROBES, 1);
        tracePass.end();

        // 2. Irradiance Update
        const irrBindGroup = device.createBindGroup({
            layout: this.irradianceUpdateLayout,
            entries: [
                { binding: 0, resource: { buffer: this.ddgiUniformBuffer } },
                { binding: 1, resource: { buffer: this.randomRotationBuffer } },
                { binding: 2, resource: { buffer: this.rayDataBuffer } },
                { binding: 3, resource: readIrr },
                { binding: 4, resource: writeIrr },
            ]
        });

        const irrPass = encoder.beginComputePass({ label: "DDGI Irradiance Update" });
        irrPass.setPipeline(this.irradianceUpdatePipeline);
        irrPass.setBindGroup(0, irrBindGroup);
        irrPass.dispatchWorkgroups(1, 1, DDGI.TOTAL_PROBES);
        irrPass.end();

        // 3. Visibility Update
        const visBindGroup = device.createBindGroup({
            layout: this.visibilityUpdateLayout,
            entries: [
                { binding: 0, resource: { buffer: this.ddgiUniformBuffer } },
                { binding: 1, resource: { buffer: this.randomRotationBuffer } },
                { binding: 2, resource: { buffer: this.rayDataBuffer } },
                { binding: 3, resource: readVis },
                { binding: 4, resource: writeVis },
            ]
        });

        const visPass = encoder.beginComputePass({ label: "DDGI Visibility Update" });
        visPass.setPipeline(this.visibilityUpdatePipeline);
        visPass.setBindGroup(0, visBindGroup);
        visPass.dispatchWorkgroups(1, 1, DDGI.TOTAL_PROBES);
        visPass.end();

        // Border update skipped — WebGPU doesn't allow same texture as both read+write
        // in the same bind group. Instead, we inset sampling UVs in ddgiIrradianceTexelCoord
        // to avoid reading from uninitialized border texels.

        // Flip ping-pong
        this.pingPong = 1 - this.pingPong;
    }

    /**
     * Returns the current "read" atlas views (the one that was just written to).
     */
    getCurrentIrradianceView(): GPUTextureView {
        // After ping-pong flip, the "just written" is the opposite of current pingPong
        return this.pingPong === 0 ? this.irradianceAtlasBView : this.irradianceAtlasAView;
    }

    getCurrentVisibilityView(): GPUTextureView {
        return this.pingPong === 0 ? this.visibilityAtlasBView : this.visibilityAtlasAView;
    }
}

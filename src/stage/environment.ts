import { device } from '../renderer';
import * as shaders from '../shaders/shaders';

/**
 * Environment class manages IBL (Image-Based Lighting) resources:
 * - Procedurally generated sky cubemap
 * - Diffuse irradiance cubemap (convolved)
 * - Specular prefiltered environment cubemap (mip chain)
 * - BRDF integration LUT
 */
export class Environment {
    // Cubemap sizes
    static readonly ENV_SIZE = 256;
    static readonly IRRADIANCE_SIZE = 32;
    static readonly PREFILTER_SIZE = 128;
    static readonly PREFILTER_MIP_LEVELS = 5;
    static readonly BRDF_LUT_SIZE = 256;

    // GPU textures
    envCubemap: GPUTexture;
    irradianceMap: GPUTexture;
    prefilteredMap: GPUTexture;
    brdfLut: GPUTexture;

    // Views
    envCubemapView: GPUTextureView;
    irradianceMapView: GPUTextureView;
    prefilteredMapView: GPUTextureView;
    brdfLutView: GPUTextureView;

    // Sampler
    envSampler: GPUSampler;

    constructor() {
        // Create environment cubemap texture
        this.envCubemap = device.createTexture({
            label: "env cubemap",
            size: [Environment.ENV_SIZE, Environment.ENV_SIZE, 6],
            format: 'rgba16float',
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
            dimension: '2d',
        });
        this.envCubemapView = this.envCubemap.createView({
            dimension: 'cube',
            arrayLayerCount: 6,
        });

        // Create irradiance cubemap
        this.irradianceMap = device.createTexture({
            label: "irradiance cubemap",
            size: [Environment.IRRADIANCE_SIZE, Environment.IRRADIANCE_SIZE, 6],
            format: 'rgba16float',
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
            dimension: '2d',
        });
        this.irradianceMapView = this.irradianceMap.createView({
            dimension: 'cube',
            arrayLayerCount: 6,
        });

        // Create prefiltered specular cubemap with mip levels
        this.prefilteredMap = device.createTexture({
            label: "prefiltered specular cubemap",
            size: [Environment.PREFILTER_SIZE, Environment.PREFILTER_SIZE, 6],
            format: 'rgba16float',
            mipLevelCount: Environment.PREFILTER_MIP_LEVELS,
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
            dimension: '2d',
        });
        this.prefilteredMapView = this.prefilteredMap.createView({
            dimension: 'cube',
            arrayLayerCount: 6,
            baseMipLevel: 0,
            mipLevelCount: Environment.PREFILTER_MIP_LEVELS,
        });

        // Create BRDF LUT
        this.brdfLut = device.createTexture({
            label: "BRDF LUT",
            size: [Environment.BRDF_LUT_SIZE, Environment.BRDF_LUT_SIZE],
            format: 'rgba16float',
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.brdfLutView = this.brdfLut.createView();

        // Create sampler with linear filtering and mipmap support
        this.envSampler = device.createSampler({
            label: "env sampler",
            magFilter: 'linear',
            minFilter: 'linear',
            mipmapFilter: 'linear',
            addressModeU: 'clamp-to-edge',
            addressModeV: 'clamp-to-edge',
            addressModeW: 'clamp-to-edge',
        });

        // Run all precomputation
        this.generateCubemap();
        this.computeIrradiance();
        this.computePrefilteredMap();
        this.computeBrdfLut();

        console.log("IBL environment precomputed successfully");
    }

    private generateCubemap() {
        const module = device.createShaderModule({
            label: "generate cubemap shader",
            code: shaders.generateCubemapSrc,
        });

        const bindGroupLayout = device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float', viewDimension: '2d-array' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            ]
        });

        const paramsBuffer = device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        // face_size (u32), sun_dir (3 f32s)
        const paramsData = new ArrayBuffer(16);
        new Uint32Array(paramsData, 0, 1).set([Environment.ENV_SIZE]);
        new Float32Array(paramsData, 4, 3).set([0.5, 0.7, 0.3]); // sun direction
        device.queue.writeBuffer(paramsBuffer, 0, paramsData);

        const outputView = this.envCubemap.createView({
            dimension: '2d-array',
            arrayLayerCount: 6,
        });

        const bindGroup = device.createBindGroup({
            layout: bindGroupLayout,
            entries: [
                { binding: 0, resource: outputView },
                { binding: 1, resource: { buffer: paramsBuffer } },
            ]
        });

        const pipeline = device.createComputePipeline({
            layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
            compute: { module, entryPoint: 'main' }
        });

        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(
            Math.ceil(Environment.ENV_SIZE / 8),
            Math.ceil(Environment.ENV_SIZE / 8),
            6
        );
        pass.end();
        device.queue.submit([encoder.finish()]);
    }

    // Load an equirectangular HDRI map and convert it to the environment cubemap
    public async loadHDRI(hdrData: Float32Array, width: number, height: number) {
        // WebGPU requires bytesPerRow to be a multiple of 256
        const bytesPerPixel = 16; // rgba32float = 4 channels * 4 bytes
        const unalignedBytesPerRow = width * bytesPerPixel;
        const alignedBytesPerRow = Math.ceil(unalignedBytesPerRow / 256) * 256;

        // If alignment padding is needed, create a padded buffer
        let uploadData: Float32Array;
        if (alignedBytesPerRow !== unalignedBytesPerRow) {
            const paddedFloatsPerRow = alignedBytesPerRow / 4; // bytes to floats
            uploadData = new Float32Array(paddedFloatsPerRow * height);
            const srcFloatsPerRow = width * 4;
            for (let y = 0; y < height; y++) {
                uploadData.set(
                    hdrData.subarray(y * srcFloatsPerRow, (y + 1) * srcFloatsPerRow),
                    y * paddedFloatsPerRow
                );
            }
        } else {
            uploadData = hdrData;
        }

        // Create 2D equirectangular texture (rgba32float, unfilterable)
        const equiTex = device.createTexture({
            label: "equirectangular source HDRI",
            size: [width, height, 1],
            format: 'rgba32float',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });

        // Write data to texture
        device.queue.writeTexture(
            { texture: equiTex },
            uploadData.buffer,
            { offset: uploadData.byteOffset, bytesPerRow: alignedBytesPerRow, rowsPerImage: height },
            [width, height, 1]
        );

        console.log(`[HDRI Debug] Equirectangular texture: ${width}x${height}`);
        console.log(`[HDRI Debug] Output cubemap size: ${Environment.ENV_SIZE}x${Environment.ENV_SIZE}`);

        // Compute shader module
        const module = device.createShaderModule({
            label: "equirectangular to cubemap shader",
            code: shaders.equirectangularToCubemapSrc,
        });

        const bindGroupLayout = device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float', viewDimension: '2d-array' } },
            ]
        });

        const outputView = this.envCubemap.createView({
            dimension: '2d-array',
            arrayLayerCount: 6,
        });

        const bindGroup = device.createBindGroup({
            layout: bindGroupLayout,
            entries: [
                { binding: 0, resource: equiTex.createView() },
                { binding: 1, resource: outputView },
            ]
        });

        const pipeline = device.createComputePipeline({
            layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
            compute: { module, entryPoint: 'main' }
        });

        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        // The output cubemap faces are Environment.ENV_SIZE x Environment.ENV_SIZE
        pass.dispatchWorkgroups(
            Math.ceil(Environment.ENV_SIZE / 8),
            Math.ceil(Environment.ENV_SIZE / 8),
            6
        );
        pass.end();
        device.queue.submit([encoder.finish()]);

        // After the cubemap is populated, recompute irradiance and prefiltered maps
        // We don't need to recompute the BRDF LUT as it's independent of the environment map
        this.computeIrradiance();
        this.computePrefilteredMap();

        console.log("HDRI processed and IBL recomputed");
    }

    private computeIrradiance() {
        const module = device.createShaderModule({
            label: "irradiance convolution shader",
            code: shaders.irradianceConvolutionSrc,
        });

        const bindGroupLayout = device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float', viewDimension: 'cube' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, sampler: {} },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float', viewDimension: '2d-array' } },
            ]
        });

        const outputView = this.irradianceMap.createView({
            dimension: '2d-array',
            arrayLayerCount: 6,
        });

        const bindGroup = device.createBindGroup({
            layout: bindGroupLayout,
            entries: [
                { binding: 0, resource: this.envCubemapView },
                { binding: 1, resource: this.envSampler },
                { binding: 2, resource: outputView },
            ]
        });

        const pipeline = device.createComputePipeline({
            layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
            compute: { module, entryPoint: 'main' }
        });

        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(
            Math.ceil(Environment.IRRADIANCE_SIZE / 8),
            Math.ceil(Environment.IRRADIANCE_SIZE / 8),
            6
        );
        pass.end();
        device.queue.submit([encoder.finish()]);
    }

    private computePrefilteredMap() {
        const module = device.createShaderModule({
            label: "prefilter envmap shader",
            code: shaders.prefilterEnvmapSrc,
        });

        const bindGroupLayout = device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float', viewDimension: 'cube' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, sampler: {} },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float', viewDimension: '2d-array' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            ]
        });

        const pipeline = device.createComputePipeline({
            layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
            compute: { module, entryPoint: 'main' }
        });

        // Dispatch once per mip level
        for (let mip = 0; mip < Environment.PREFILTER_MIP_LEVELS; mip++) {
            const mipSize = Math.max(1, Environment.PREFILTER_SIZE >> mip);
            const roughness = mip / (Environment.PREFILTER_MIP_LEVELS - 1);
            const numSamples = 1024;

            const paramsBuffer = device.createBuffer({
                size: 16,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            });
            const paramsData = new ArrayBuffer(16);
            new Float32Array(paramsData, 0, 1).set([roughness]);
            new Uint32Array(paramsData, 4, 1).set([mipSize]);
            new Uint32Array(paramsData, 8, 1).set([numSamples]);
            new Uint32Array(paramsData, 12, 1).set([0]); // padding
            device.queue.writeBuffer(paramsBuffer, 0, paramsData);

            const mipView = this.prefilteredMap.createView({
                dimension: '2d-array',
                arrayLayerCount: 6,
                baseMipLevel: mip,
                mipLevelCount: 1,
            });

            const bindGroup = device.createBindGroup({
                layout: bindGroupLayout,
                entries: [
                    { binding: 0, resource: this.envCubemapView },
                    { binding: 1, resource: this.envSampler },
                    { binding: 2, resource: mipView },
                    { binding: 3, resource: { buffer: paramsBuffer } },
                ]
            });

            const encoder = device.createCommandEncoder();
            const pass = encoder.beginComputePass();
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, bindGroup);
            pass.dispatchWorkgroups(
                Math.ceil(mipSize / 8),
                Math.ceil(mipSize / 8),
                6
            );
            pass.end();
            device.queue.submit([encoder.finish()]);
        }
    }

    private computeBrdfLut() {
        const module = device.createShaderModule({
            label: "BRDF LUT shader",
            code: shaders.brdfLutSrc,
        });

        const bindGroupLayout = device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } },
            ]
        });

        const bindGroup = device.createBindGroup({
            layout: bindGroupLayout,
            entries: [
                { binding: 0, resource: this.brdfLutView },
            ]
        });

        const pipeline = device.createComputePipeline({
            layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
            compute: { module, entryPoint: 'main' }
        });

        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(
            Math.ceil(Environment.BRDF_LUT_SIZE / 8),
            Math.ceil(Environment.BRDF_LUT_SIZE / 8),
            1
        );
        pass.end();
        device.queue.submit([encoder.finish()]);
    }
}

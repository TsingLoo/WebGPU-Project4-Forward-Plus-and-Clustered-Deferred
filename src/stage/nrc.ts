import { device, canvas } from '../renderer';
import * as shaders from '../shaders/shaders';
import { Camera } from './camera';
import { Environment } from './environment';

/**
 * Neural Radiance Caching (NRC) manager.
 * Online-trained MLP that caches indirect radiance for real-time GI.
 * The MLP (15D input → 32 → 64 → 64 → 3 output) is trained each frame
 * from G-buffer derived training samples and queried per-pixel for indirect lighting.
 */
export class NRC {
    // MLP architecture constants (must match nrc_common.wgsl)
    static readonly TOTAL_PARAMS = 6980;
    static readonly MAX_TRAINING_SAMPLES = shaders.constants.nrcMaxTrainingSamples;
    static readonly SAMPLE_STRIDE = 20; // floats per sample

    // Configuration
    enabled = false;
    learningRate = 0.001;
    momentum = 0.9;
    debugMode = 0; // 0=off, 1=raw inference, 2=HDR mapped
    sampleStride = 4; // subsample every Nth pixel in x and y

    // Scene Bounds
    sceneMin = [-14.0, 0.0, -7.0];
    sceneMax = [14.0, 12.0, 7.0];

    // GPU resources
    weightsBuffer: GPUBuffer;
    gradAccumBuffer: GPUBuffer;
    momentumBuffer: GPUBuffer;
    trainingSamplesBuffer: GPUBuffer;
    sampleCounterBuffer: GPUBuffer;
    sampleCounterZeroBuffer: GPUBuffer;
    nrcUniformBuffer: GPUBuffer;

    // Inference output texture
    inferenceTexture: GPUTexture;
    inferenceTextureView: GPUTextureView;

    // Pipelines
    scatterPipeline: GPUComputePipeline;
    trainPipeline: GPUComputePipeline;
    inferencePipeline: GPUComputePipeline;

    // Bind group layouts
    scatterLayout: GPUBindGroupLayout;
    trainLayout: GPUBindGroupLayout;
    inferenceLayout: GPUBindGroupLayout;

    private camera: Camera;
    private environment: Environment;
    private frameCount = 0;

    constructor(camera: Camera, environment: Environment) {
        this.camera = camera;
        this.environment = environment;

        // ---- Create GPU Buffers ----

        // MLP weights buffer (initialized to small random values on CPU)
        const weightsData = new Float32Array(NRC.TOTAL_PARAMS);

        // Kaiming (He) Uniform initialization: U(-limit, limit) where limit = sqrt(6 / fan_in)
        let wOffset = 0;
        
        // Helper to init a layer
        const initLayer = (fanIn: number, fanOut: number, wSize: number, bSize: number) => {
            const limit = Math.sqrt(6.0 / fanIn);
            // Weights
            for (let i = 0; i < wSize; i++) {
                weightsData[wOffset++] = (Math.random() * 2 - 1) * limit;
            }
            // Biases
            for (let i = 0; i < bSize; i++) {
                weightsData[wOffset++] = 0.01; // Small positive bias to avoid dead ReLUs
            }
        };

        // Layer 0: 15 in -> 32 out
        initLayer(15, 32, 15 * 32, 32);
        // Layer 1: 32 in -> 64 out
        initLayer(32, 64, 32 * 64, 64);
        // Layer 2: 64 in -> 64 out
        initLayer(64, 64, 64 * 64, 64);
        // Layer 3: 64 in -> 3 out
        initLayer(64, 3, 64 * 3, 3);
        this.weightsBuffer = device.createBuffer({
            label: "NRC Weights",
            size: NRC.TOTAL_PARAMS * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true,
        });
        new Float32Array(this.weightsBuffer.getMappedRange()).set(weightsData);
        this.weightsBuffer.unmap();

        // Gradient accumulator (zeroed)
        this.gradAccumBuffer = device.createBuffer({
            label: "NRC Gradient Accumulator",
            size: NRC.TOTAL_PARAMS * 4,
            usage: GPUBufferUsage.STORAGE,
        });

        // Momentum buffer (zeroed)
        this.momentumBuffer = device.createBuffer({
            label: "NRC Momentum",
            size: NRC.TOTAL_PARAMS * 4,
            usage: GPUBufferUsage.STORAGE,
        });

        // Training samples buffer
        const maxSampleBytes = NRC.MAX_TRAINING_SAMPLES * NRC.SAMPLE_STRIDE * 4;
        this.trainingSamplesBuffer = device.createBuffer({
            label: "NRC Training Samples",
            size: maxSampleBytes,
            usage: GPUBufferUsage.STORAGE,
        });

        // Atomic counter for training samples (u32)
        this.sampleCounterBuffer = device.createBuffer({
            label: "NRC Sample Counter",
            size: 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        // Zero buffer for resetting counter
        this.sampleCounterZeroBuffer = device.createBuffer({
            label: "NRC Counter Zero",
            size: 4,
            usage: GPUBufferUsage.COPY_SRC,
            mappedAtCreation: true,
        });
        new Uint32Array(this.sampleCounterZeroBuffer.getMappedRange()).set([0]);
        this.sampleCounterZeroBuffer.unmap();

        // NRC uniform buffer (NRCUniforms: 4 × vec4f = 64 bytes)
        this.nrcUniformBuffer = device.createBuffer({
            label: "NRC Uniforms",
            size: 64,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Inference output texture (full screen resolution)
        this.inferenceTexture = device.createTexture({
            label: "NRC Inference Output",
            size: [canvas.width, canvas.height],
            format: 'rgba16float',
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.inferenceTextureView = this.inferenceTexture.createView();

        // ---- Create Bind Group Layouts ----
        this.scatterLayout = this.createScatterLayout();
        this.trainLayout = this.createTrainLayout();
        this.inferenceLayout = this.createInferenceLayout();

        // ---- Create Compute Pipelines ----
        this.scatterPipeline = device.createComputePipeline({
            label: "NRC Scatter Training Pipeline",
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.scatterLayout] }),
            compute: {
                module: device.createShaderModule({
                    label: "NRC Scatter Training",
                    code: shaders.nrcScatterTrainingSrc,
                }),
                entryPoint: 'main',
            },
        });

        this.trainPipeline = device.createComputePipeline({
            label: "NRC Train Pipeline",
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.trainLayout] }),
            compute: {
                module: device.createShaderModule({
                    label: "NRC Train",
                    code: shaders.nrcTrainSrc,
                }),
                entryPoint: 'main',
            },
        });

        this.inferencePipeline = device.createComputePipeline({
            label: "NRC Inference Pipeline",
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.inferenceLayout] }),
            compute: {
                module: device.createShaderModule({
                    label: "NRC Inference",
                    code: shaders.nrcInferenceSrc,
                }),
                entryPoint: 'main',
            },
        });

        this.updateUniforms();
        console.log(`NRC initialized: MLP ${NRC.TOTAL_PARAMS} params, max ${NRC.MAX_TRAINING_SAMPLES} training samples/frame`);
    }

    updateUniforms() {
        const data = new Float32Array(16); // 64 bytes = 16 floats

        // scene_min: vec4f (xyz = scene min, w = enabled)
        data[0] = this.sceneMin[0]; data[1] = this.sceneMin[1]; data[2] = this.sceneMin[2]; data[3] = this.enabled ? 1.0 : 0.0;

        // scene_max: vec4f (xyz = scene max, w = debug_mode)
        data[4] = this.sceneMax[0]; data[5] = this.sceneMax[1]; data[6] = this.sceneMax[2]; data[7] = this.debugMode;

        // params: vec4f (lr, num_samples, momentum, frame_count)
        data[8] = this.learningRate;
        data[9] = 0; // filled at dispatch time with actual count
        data[10] = this.momentum;
        data[11] = this.frameCount;

        // screen_dims: vec4f (w, h, stride_x, stride_y)
        data[12] = canvas.width;
        data[13] = canvas.height;
        data[14] = this.sampleStride;
        data[15] = this.sampleStride;

        device.queue.writeBuffer(this.nrcUniformBuffer, 0, data);
    }

    // Scene bounds can be updated from DDGI grid or custom settings
    setSceneBounds(min: [number, number, number], max: [number, number, number]) {
        this.sceneMin = [min[0], min[1], min[2]];
        this.sceneMax = [max[0], max[1], max[2]];
        this.updateUniforms();
    }

    private createScatterLayout(): GPUBindGroupLayout {
        return device.createBindGroupLayout({
            label: "NRC Scatter Layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },    // camera
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },    // nrc uniforms
                { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'depth' } }, // depth
                { binding: 3, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } }, // normal
                { binding: 4, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } }, // albedo
                { binding: 5, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } }, // position
                { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },    // sun light
                { binding: 7, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'depth' } }, // VSM atlas
                { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },    // VSM uniforms
                { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },     // training samples
                { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },    // sample counter
                { binding: 11, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float', viewDimension: 'cube' } }, // env map
                { binding: 12, visibility: GPUShaderStage.COMPUTE, sampler: {} },                     // env sampler
            ],
        });
    }

    private createTrainLayout(): GPUBindGroupLayout {
        return device.createBindGroupLayout({
            label: "NRC Train Layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },         // nrc uniforms
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },          // weights (read-write)
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // training samples
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },          // grad accum
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },          // momentum
            ],
        });
    }

    private createInferenceLayout(): GPUBindGroupLayout {
        return device.createBindGroupLayout({
            label: "NRC Inference Layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },    // camera
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },    // nrc uniforms
                { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'depth' } }, // depth
                { binding: 3, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } }, // normal
                { binding: 4, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } }, // position
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // weights (read)
                { binding: 6, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } }, // output
            ],
        });
    }

    /**
     * Dispatches all NRC compute passes: scatter training → train MLP → inference
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

        this.frameCount++;
        this.updateUniforms();

        // Reset sample counter
        encoder.copyBufferToBuffer(
            this.sampleCounterZeroBuffer, 0,
            this.sampleCounterBuffer, 0,
            4
        );

        // 1. Scatter Training Samples
        const scatterBindGroup = device.createBindGroup({
            layout: this.scatterLayout,
            entries: [
                { binding: 0, resource: { buffer: this.camera.uniformsBuffer } },
                { binding: 1, resource: { buffer: this.nrcUniformBuffer } },
                { binding: 2, resource: gBuffer.depth },
                { binding: 3, resource: gBuffer.normal },
                { binding: 4, resource: gBuffer.albedo },
                { binding: 5, resource: gBuffer.position },
                { binding: 6, resource: { buffer: sunLightBuffer } },
                { binding: 7, resource: shadowMapView },
                { binding: 8, resource: { buffer: vsmUniformBuffer } },
                { binding: 9, resource: { buffer: this.trainingSamplesBuffer } },
                { binding: 10, resource: { buffer: this.sampleCounterBuffer } },
                { binding: 11, resource: this.environment.envCubemapView },
                { binding: 12, resource: this.environment.envSampler },
            ],
        });

        const scatterPass = encoder.beginComputePass({ label: "NRC Scatter Training" });
        scatterPass.setPipeline(this.scatterPipeline);
        scatterPass.setBindGroup(0, scatterBindGroup);
        // Dispatch enough workgroups to cover subsampled screen
        const scatterWGX = Math.ceil(canvas.width / this.sampleStride / 8);
        const scatterWGY = Math.ceil(canvas.height / this.sampleStride / 8);
        scatterPass.dispatchWorkgroups(scatterWGX, scatterWGY, 1);
        scatterPass.end();

        // 2. Train MLP
        // Update uniform with estimated sample count
        const estimatedSamples = Math.min(
            Math.floor(canvas.width / this.sampleStride) * Math.floor(canvas.height / this.sampleStride),
            NRC.MAX_TRAINING_SAMPLES
        );
        // Update just the num_samples field
        const paramsUpdate = new Float32Array(4);
        paramsUpdate[0] = this.learningRate;
        paramsUpdate[1] = estimatedSamples;
        paramsUpdate[2] = this.momentum;
        paramsUpdate[3] = this.frameCount;
        device.queue.writeBuffer(this.nrcUniformBuffer, 32, paramsUpdate); // offset 32 = params vec4f

        const trainBindGroup = device.createBindGroup({
            layout: this.trainLayout,
            entries: [
                { binding: 0, resource: { buffer: this.nrcUniformBuffer } },
                { binding: 1, resource: { buffer: this.weightsBuffer } },
                { binding: 2, resource: { buffer: this.trainingSamplesBuffer } },
                { binding: 3, resource: { buffer: this.gradAccumBuffer } },
                { binding: 4, resource: { buffer: this.momentumBuffer } },
            ],
        });

        const trainPass = encoder.beginComputePass({ label: "NRC Train" });
        trainPass.setPipeline(this.trainPipeline);
        trainPass.setBindGroup(0, trainBindGroup);
        // Single workgroup iterates over all samples to avoid float data races
        trainPass.dispatchWorkgroups(1, 1, 1);
        trainPass.end();

        // 3. Inference (full-screen)
        const inferenceBindGroup = device.createBindGroup({
            layout: this.inferenceLayout,
            entries: [
                { binding: 0, resource: { buffer: this.camera.uniformsBuffer } },
                { binding: 1, resource: { buffer: this.nrcUniformBuffer } },
                { binding: 2, resource: gBuffer.depth },
                { binding: 3, resource: gBuffer.normal },
                { binding: 4, resource: gBuffer.position },
                { binding: 5, resource: { buffer: this.weightsBuffer } },
                { binding: 6, resource: this.inferenceTextureView },
            ],
        });

        const inferencePass = encoder.beginComputePass({ label: "NRC Inference" });
        inferencePass.setPipeline(this.inferencePipeline);
        inferencePass.setBindGroup(0, inferenceBindGroup);
        inferencePass.dispatchWorkgroups(
            Math.ceil(canvas.width / 8),
            Math.ceil(canvas.height / 8),
            1
        );
        inferencePass.end();
    }

    getInferenceView(): GPUTextureView {
        return this.inferenceTextureView;
    }
}

import * as renderer from '../renderer';
import * as shaders from '../shaders/shaders';
import { Stage } from '../stage/stage';
import { DDGI } from '../stage/ddgi';
import { NRC } from '../stage/nrc';
import { VSM } from '../stage/vsm';

export class ForwardPlusRenderer extends renderer.Renderer {
    depthTexture: GPUTexture;
    depthTextureView: GPUTextureView;

    tileOffsetsDeviceBuffer: GPUBuffer;
    globalLightIndicesDeviceBuffer: GPUBuffer;
    zeroDeviceBuffer: GPUBuffer;

    clusterSetDeviceBuffer: GPUBuffer;

    zPrepassPipeline: GPURenderPipeline;

    cullingBindGroupLayout: GPUBindGroupLayout;
    cullingBindGroup: GPUBindGroup;
    cullingPipeline: GPUComputePipeline;

    shadingBindGroupLayout: GPUBindGroupLayout; 
    shadingBindGroup!: GPUBindGroup;
    shadingPipeline: GPURenderPipeline;

    skyboxPipeline: GPURenderPipeline;
    skyboxBindGroupLayout: GPUBindGroupLayout;
    skyboxBindGroup: GPUBindGroup;

    // Volumetric Lighting
    volumetricTexture: GPUTexture;
    volumetricTextureView: GPUTextureView;
    volumetricPipeline: GPURenderPipeline;
    volumetricBindGroupLayout: GPUBindGroupLayout;
    volumetricBindGroup!: GPUBindGroup;
    
    volumetricCompositePipeline: GPURenderPipeline;
    volumetricCompositeBindGroupLayout: GPUBindGroupLayout;
    volumetricCompositeBindGroup!: GPUBindGroup;

    // G-buffer for DDGI probe tracing
    gBufferNormalTexture: GPUTexture;
    gBufferNormalTextureView: GPUTextureView;
    gBufferAlbedoTexture: GPUTexture;
    gBufferAlbedoTextureView: GPUTextureView;
    gBufferPositionTexture: GPUTexture;
    gBufferPositionTextureView: GPUTextureView;
    gBufferSpecularTexture: GPUTexture;
    gBufferSpecularTextureView: GPUTextureView;
    geometryBindGroupLayout: GPUBindGroupLayout;
    geometryBindGroup: GPUBindGroup;
    geometryPipeline: GPURenderPipeline;

    ddgi: DDGI;
    nrc: NRC;
    vsm: VSM;
    private stageEnv: import('../stage/environment').Environment;
    private stage: import('../stage/stage').Stage;

    constructor(stage: Stage) {
        super(stage);
        this.depthTexture = renderer.device.createTexture({
            size: [renderer.canvas.width, renderer.canvas.height],
            format: "depth24plus",
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
        });
        this.depthTextureView = this.depthTexture.createView();

        const volWidth = Math.max(1, Math.floor(renderer.canvas.width / 2));
        const volHeight = Math.max(1, Math.floor(renderer.canvas.height / 2));
        this.volumetricTexture = renderer.device.createTexture({
            label: "volumetric downsampled texture",
            size: [volWidth, volHeight],
            format: "rgba16float",
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
        });
        this.volumetricTextureView = this.volumetricTexture.createView();

        // G-buffer textures for DDGI probe tracing
        const gBufSize = [renderer.canvas.width, renderer.canvas.height];
        this.gBufferAlbedoTexture = renderer.device.createTexture({
            label: "Forward+ G-Buffer Albedo (DDGI)",
            size: gBufSize, format: 'rgba16float',
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.gBufferAlbedoTextureView = this.gBufferAlbedoTexture.createView();

        this.gBufferNormalTexture = renderer.device.createTexture({
            label: "Forward+ G-Buffer Normal (DDGI)",
            size: gBufSize, format: 'rgba16float',
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.gBufferNormalTextureView = this.gBufferNormalTexture.createView();

        this.gBufferPositionTexture = renderer.device.createTexture({
            label: "Forward+ G-Buffer Position (DDGI)",
            size: gBufSize, format: 'rgba16float',
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.gBufferPositionTextureView = this.gBufferPositionTexture.createView();

        this.gBufferSpecularTexture = renderer.device.createTexture({
            label: "Forward+ G-Buffer Specular (DDGI)",
            size: gBufSize, format: 'rgba8unorm',
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.gBufferSpecularTextureView = this.gBufferSpecularTexture.createView();

        this.tileOffsetsDeviceBuffer = renderer.device.createBuffer({
            size: shaders.constants.numTotalClustersConfig * 2 * 4,
            usage: GPUBufferUsage.STORAGE,
        })

        this.zeroDeviceBuffer = renderer.device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.COPY_SRC,
            mappedAtCreation: true
        });
        new Uint32Array(this.zeroDeviceBuffer.getMappedRange()).set([0]);
        this.zeroDeviceBuffer.unmap();

        this.clusterSetDeviceBuffer = renderer.device.createBuffer({
            size: 4 * 5,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true
        });
        const mappedRange = this.clusterSetDeviceBuffer.getMappedRange();
        const uintView = new Uint32Array(mappedRange);

        uintView[0] = renderer.canvas.width;
        uintView[1] = renderer.canvas.height;
        uintView[2] = shaders.constants.numClustersX;
        uintView[3] = shaders.constants.numClustersY;
        uintView[4] = shaders.constants.numClustersZ;
        this.clusterSetDeviceBuffer.unmap();

        const maxIndices = shaders.constants.numTotalClustersConfig * shaders.constants.averageLightsPerCluster;

        this.globalLightIndicesDeviceBuffer = renderer.device.createBuffer({
            size: 4 + maxIndices * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            label: "global light indices buffer"
        })

        const env = stage.environment;
        this.stageEnv = env;
        this.ddgi = stage.ddgi;
        this.nrc = stage.nrc;
        this.vsm = stage.vsm;
        this.stage = stage;

        // Geometry bind group (camera only) for G-buffer pass
        this.geometryBindGroupLayout = renderer.device.createBindGroupLayout({
            label: "fwd+ geometry bind group layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }
            ]
        });
        this.geometryBindGroup = renderer.device.createBindGroup({
            label: "fwd+ geometry bind group",
            layout: this.geometryBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.camera.uniformsBuffer } }
            ]
        });
        this.geometryPipeline = renderer.device.createRenderPipeline({
            label: "fwd+ geometry pipeline (DDGI)",
            layout: renderer.device.createPipelineLayout({
                bindGroupLayouts: [this.geometryBindGroupLayout, renderer.modelBindGroupLayout, renderer.materialBindGroupLayout]
            }),
            depthStencil: { format: 'depth24plus', depthWriteEnabled: false, depthCompare: 'equal' },
            vertex: {
                module: renderer.device.createShaderModule({ code: shaders.naiveVertSrc }),
                buffers: [renderer.vertexBufferLayout]
            },
            fragment: {
                module: renderer.device.createShaderModule({ code: shaders.geometryFragSrc }),
                entryPoint: 'main',
                targets: [
                    { format: 'rgba16float' },   // albedo
                    { format: 'rgba16float' },  // normal
                    { format: 'rgba16float' },  // position
                    { format: 'rgba8unorm' },   // specular
                ]
            },
            primitive: { topology: 'triangle-list', cullMode: 'back' }
        });

        // Shading bind group layout — now with VSM bindings
        this.shadingBindGroupLayout = renderer.device.createBindGroupLayout({
            label: "shading bind group layout",
            entries: [
                { // Camera Uniforms
                    binding: 0,
                    visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                    buffer: { type: "uniform" }
                },
                {   // Light Set
                    binding: 1,
                    visibility: GPUShaderStage.FRAGMENT,
                    buffer: { type: "read-only-storage" }
                },
                {   // Tile offsets
                    binding: 2,
                    visibility: GPUShaderStage.FRAGMENT,
                    buffer: { type: "read-only-storage" } 
                },
                {   // Global light indices
                    binding: 3,
                    visibility: GPUShaderStage.FRAGMENT,
                    buffer: { type: "read-only-storage" } 
                },
                {   // Cluster set
                    binding: 4,
                    visibility: GPUShaderStage.FRAGMENT,
                    buffer: { type: "uniform" }
                },
                {   // Irradiance cubemap
                    binding: 5,
                    visibility: GPUShaderStage.FRAGMENT,
                    texture: { sampleType: "float", viewDimension: "cube" }
                },
                {   // Prefiltered specular cubemap
                    binding: 6,
                    visibility: GPUShaderStage.FRAGMENT,
                    texture: { sampleType: "float", viewDimension: "cube" }
                },
                {   // BRDF LUT
                    binding: 7,
                    visibility: GPUShaderStage.FRAGMENT,
                    texture: { sampleType: "float" }
                },
                {   // IBL Sampler
                    binding: 8,
                    visibility: GPUShaderStage.FRAGMENT,
                    sampler: {}
                },
                {   // DDGI Irradiance Atlas
                    binding: 9,
                    visibility: GPUShaderStage.FRAGMENT,
                    texture: { sampleType: "float" }
                },
                {   // DDGI Visibility Atlas
                    binding: 10,
                    visibility: GPUShaderStage.FRAGMENT,
                    texture: { sampleType: "float" }
                },
                {   // DDGI Uniforms
                    binding: 11,
                    visibility: GPUShaderStage.FRAGMENT,
                    buffer: { type: "uniform" }
                },
                {   // DDGI Sampler
                    binding: 12,
                    visibility: GPUShaderStage.FRAGMENT,
                    sampler: {}
                },
                {   // Sun Light
                    binding: 13,
                    visibility: GPUShaderStage.FRAGMENT,
                    buffer: { type: "uniform" }
                },
                {   // VSM Physical Atlas (depth texture)
                    binding: 14,
                    visibility: GPUShaderStage.FRAGMENT,
                    texture: { sampleType: "depth" }
                },
                {   // VSM Comparison Sampler
                    binding: 15,
                    visibility: GPUShaderStage.FRAGMENT,
                    sampler: { type: "comparison" }
                },
                {   // VSM Page Table (read-only storage)
                    binding: 16,
                    visibility: GPUShaderStage.FRAGMENT,
                    buffer: { type: "read-only-storage" }
                },
                {   // VSM Uniforms
                    binding: 17,
                    visibility: GPUShaderStage.FRAGMENT,
                    buffer: { type: "uniform" }
                },
                {   // NRC Inference Texture
                    binding: 18,
                    visibility: GPUShaderStage.FRAGMENT,
                    texture: { sampleType: "float" }
                },
                {   // NRC Uniforms
                    binding: 19,
                    visibility: GPUShaderStage.FRAGMENT,
                    buffer: { type: "uniform" }
                },
                {   // GBuffer Position
                    binding: 20,
                    visibility: GPUShaderStage.FRAGMENT,
                    texture: { sampleType: "unfilterable-float" }
                },
                {   // GBuffer Normal
                    binding: 21,
                    visibility: GPUShaderStage.FRAGMENT,
                    texture: { sampleType: "unfilterable-float" }
                },
                {   // GBuffer Albedo
                    binding: 22,
                    visibility: GPUShaderStage.FRAGMENT,
                    texture: { sampleType: "unfilterable-float" }
                }
            ]
        });

        this.zPrepassPipeline = renderer.device.createRenderPipeline({
            label: "Z-Prepass pipeline",
            layout: renderer.device.createPipelineLayout({
                bindGroupLayouts: [
                    this.shadingBindGroupLayout,
                    renderer.modelBindGroupLayout,
                    renderer.materialBindGroupLayout
                ]
            }),
            depthStencil: {
                depthWriteEnabled: true,
                depthCompare: "less",
                format: "depth24plus"
            },
            vertex: {
                module: renderer.device.createShaderModule({
                    code: shaders.naiveVertSrc
                }),
                buffers: [ renderer.vertexBufferLayout ]
            },
            fragment: {
                module: renderer.device.createShaderModule({
                    code: shaders.zPrepassFragSrc
                }),
                entryPoint: "main",
                targets: [] 
            }
        });


        this.cullingBindGroupLayout = renderer.device.createBindGroupLayout({
        label: "culling bind group layout",
        entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "uniform" }
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "read-only-storage" }
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "storage" } 
                },
                {
                    binding: 3,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "storage" } 
                },
                {
                    binding: 4,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "uniform" }
                }
            ]
        })

        this.cullingBindGroup = renderer.device.createBindGroup({
            label: "culling bind group",
            layout: this.cullingBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.camera.uniformsBuffer }},
                { binding: 1, resource: { buffer: this.lights.lightSetStorageBuffer }},
                { binding: 2, resource: { buffer: this.tileOffsetsDeviceBuffer }},
                { binding: 3, resource: { buffer: this.globalLightIndicesDeviceBuffer }},
                { binding: 4, resource: { buffer: this.clusterSetDeviceBuffer }}
            ]
        });

        this.cullingPipeline = renderer.device.createComputePipeline({
            label: "culling compute pipeline",
            layout: renderer.device.createPipelineLayout({
                bindGroupLayouts: [this.cullingBindGroupLayout]
            }),
            compute: {
                module: renderer.device.createShaderModule({
                    code: shaders.clusteringComputeSrc
                }),
                entryPoint: "main" 
            }
        });

        // Initial shading bind group (will be recreated each frame for DDGI ping-pong)
        this.createShadingBindGroup();

        this.shadingPipeline = renderer.device.createRenderPipeline({
            label: "Forward+ Shading Pipeline",
            layout: renderer.device.createPipelineLayout({
                bindGroupLayouts: [
                    this.shadingBindGroupLayout,
                    renderer.modelBindGroupLayout,
                    renderer.materialBindGroupLayout
                ]
            }),
            depthStencil: {
                depthWriteEnabled: false,
                depthCompare: "equal",
                format: "depth24plus"
            },
            vertex: {
                module: renderer.device.createShaderModule({ code: shaders.naiveVertSrc }),
                buffers: [ renderer.vertexBufferLayout ]
            },
            fragment: {
                module: renderer.device.createShaderModule({
                    label: "Forward+ Shading Fragment",
                    code: shaders.forwardPlusFragSrc,
                }),
                targets: [ { format: renderer.canvasFormat }]
            }
        });

        // Skybox
        this.skyboxBindGroupLayout = renderer.device.createBindGroupLayout({
            label: "skybox bind group layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float", viewDimension: "cube" } },
                { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: {} }
            ]
        });

        this.skyboxBindGroup = renderer.device.createBindGroup({
            label: "skybox bind group",
            layout: this.skyboxBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.camera.uniformsBuffer } },
                { binding: 1, resource: env.envCubemapView },
                { binding: 2, resource: env.envSampler }
            ]
        });

        this.skyboxPipeline = renderer.device.createRenderPipeline({
            label: "skybox pipeline",
            layout: renderer.device.createPipelineLayout({
                bindGroupLayouts: [ this.skyboxBindGroupLayout ]
            }),
            depthStencil: {
                depthWriteEnabled: false,
                depthCompare: "less-equal",
                format: "depth24plus"
            },
            vertex: {
                module: renderer.device.createShaderModule({ code: shaders.skyboxVertSrc }),
                entryPoint: "main"
            },
            fragment: {
                module: renderer.device.createShaderModule({ code: shaders.skyboxFragSrc }),
                entryPoint: "main",
                targets: [ { format: renderer.canvasFormat } ]
            }
        });

        // Volumetric Lighting
        this.volumetricBindGroupLayout = renderer.device.createBindGroupLayout({
            label: "volumetric lighting bind group layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "depth" } },
                { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
                { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "depth" } },
                { binding: 4, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
                { binding: 5, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
            ]
        });

        this.volumetricPipeline = renderer.device.createRenderPipeline({
            label: "volumetric lighting pipeline",
            layout: renderer.device.createPipelineLayout({
                bindGroupLayouts: [this.volumetricBindGroupLayout]
            }),
            vertex: {
                module: renderer.device.createShaderModule({ code: shaders.volumetricLightingVertSrc }),
                entryPoint: "main"
            },
            fragment: {
                module: renderer.device.createShaderModule({ code: shaders.volumetricLightingFragSrc }),
                entryPoint: "main",
                targets: [ { 
                    format: "rgba16float" // Rendering to half-res HDR buffer
                } ]
            }
        });

        // Volumetric Bilateral Composite
        this.volumetricCompositeBindGroupLayout = renderer.device.createBindGroupLayout({
            label: "volumetric composite bind group layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "unfilterable-float" } },
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "depth" } },
                { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
            ]
        });

        this.volumetricCompositePipeline = renderer.device.createRenderPipeline({
            label: "volumetric composite pipeline",
            layout: renderer.device.createPipelineLayout({
                bindGroupLayouts: [this.volumetricCompositeBindGroupLayout]
            }),
            vertex: {
                module: renderer.device.createShaderModule({ code: shaders.volumetricLightingVertSrc }),
                entryPoint: "main"
            },
            fragment: {
                module: renderer.device.createShaderModule({ code: shaders.volumetricCompositeFragSrc }),
                entryPoint: "main",
                targets: [ { 
                    format: renderer.canvasFormat,
                    blend: {
                        color: { operation: 'add', srcFactor: 'one', dstFactor: 'one' },
                        alpha: { operation: 'add', srcFactor: 'zero', dstFactor: 'one' }
                    }
                } ]
            }
        });

        this.volumetricBindGroup = renderer.device.createBindGroup({
            label: "volumetric lighting bind group",
            layout: this.volumetricBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.camera.uniformsBuffer } },
                { binding: 1, resource: this.depthTextureView },
                { binding: 2, resource: { buffer: this.stage.sunLightBuffer } },
                { binding: 3, resource: this.vsm.physicalAtlasView },
                { binding: 4, resource: { buffer: this.vsm.pageTableBuffer } },
                { binding: 5, resource: { buffer: this.vsm.vsmUniformBuffer } },
            ]
        });

        this.volumetricCompositeBindGroup = renderer.device.createBindGroup({
            label: "volumetric composite bind group",
            layout: this.volumetricCompositeBindGroupLayout,
            entries: [
                { binding: 0, resource: this.volumetricTextureView },
                { binding: 1, resource: this.depthTextureView },
                { binding: 2, resource: { buffer: this.camera.uniformsBuffer } },
            ]
        });
    }

    private createShadingBindGroup() {
        this.shadingBindGroup = renderer.device.createBindGroup({
            label: "shading bind group",
            layout: this.shadingBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.camera.uniformsBuffer }},
                { binding: 1, resource: { buffer: this.lights.lightSetStorageBuffer }},
                { binding: 2, resource: { buffer: this.tileOffsetsDeviceBuffer }},
                { binding: 3, resource: { buffer: this.globalLightIndicesDeviceBuffer }},
                { binding: 4, resource: { buffer: this.clusterSetDeviceBuffer }},
                { binding: 5, resource: this.stageEnv.irradianceMapView },
                { binding: 6, resource: this.stageEnv.prefilteredMapView },
                { binding: 7, resource: this.stageEnv.brdfLutView },
                { binding: 8, resource: this.stageEnv.envSampler },
                { binding: 9, resource: this.ddgi.getCurrentIrradianceView() },
                { binding: 10, resource: this.ddgi.getCurrentVisibilityView() },
                { binding: 11, resource: { buffer: this.ddgi.ddgiUniformBuffer } },
                { binding: 12, resource: this.ddgi.ddgiSampler },
                { binding: 13, resource: { buffer: this.stage.sunLightBuffer } },
                // VSM bindings
                { binding: 14, resource: this.vsm.physicalAtlasView },
                { binding: 15, resource: this.vsm.shadowComparisonSampler },
                { binding: 16, resource: { buffer: this.vsm.pageTableBuffer } },
                { binding: 17, resource: { buffer: this.vsm.vsmUniformBuffer } },
                // NRC bindings
                { binding: 18, resource: this.nrc.getInferenceView() },
                { binding: 19, resource: { buffer: this.nrc.nrcUniformBuffer } },
                { binding: 20, resource: this.gBufferPositionTextureView },
                { binding: 21, resource: this.gBufferNormalTextureView },
                { binding: 22, resource: this.gBufferAlbedoTextureView },
            ]
        });
    }

    override draw() {
        const encoder = renderer.device.createCommandEncoder();
        const canvasTextureView = renderer.context.getCurrentTexture().createView();

        // Update sun light
        this.stage.updateSunLight();

        // Z-Prepass — fill depth buffer first (needed by VSM page marking)
        const zPrepass = encoder.beginRenderPass({
            label: "z prepass",
            colorAttachments: [],
            depthStencilAttachment: {
                view: this.depthTextureView,
                depthClearValue: 1.0,
                depthLoadOp: "clear",
                depthStoreOp: "store"
            }
        });
        zPrepass.setPipeline(this.zPrepassPipeline);
        zPrepass.setBindGroup(shaders.constants.bindGroup_scene, this.shadingBindGroup);
        this.scene.iterate(node => {
            zPrepass.setBindGroup(shaders.constants.bindGroup_model, node.modelBindGroup);
        }, material => {
            zPrepass.setBindGroup(shaders.constants.bindGroup_material, material.materialBindGroup);
        }, primitive => {
            zPrepass.setVertexBuffer(0, primitive.vertexBuffer);
            zPrepass.setIndexBuffer(primitive.indexBuffer, 'uint32');
            zPrepass.drawIndexed(primitive.numIndices);
        });
        zPrepass.end();

        // VSM Shadow Map Pass (uses depth buffer for page marking)
        this.stage.renderShadowMap(encoder, this.depthTextureView);

        // G-buffer pass (for DDGI probe tracing)
        const gBufferPass = encoder.beginRenderPass({
            label: "G-buffer pass (DDGI)",
            colorAttachments: [
                { view: this.gBufferAlbedoTextureView, loadOp: 'clear', clearValue: [0,0,0,0], storeOp: 'store' },
                { view: this.gBufferNormalTextureView, loadOp: 'clear', clearValue: [0,0,0,0], storeOp: 'store' },
                { view: this.gBufferPositionTextureView, loadOp: 'clear', clearValue: [0,0,0,0], storeOp: 'store' },
                { view: this.gBufferSpecularTextureView, loadOp: 'clear', clearValue: [0,0,0,0], storeOp: 'store' },
            ],
            depthStencilAttachment: {
                view: this.depthTextureView,
                depthReadOnly: true
            }
        });
        gBufferPass.setPipeline(this.geometryPipeline);
        gBufferPass.setBindGroup(shaders.constants.bindGroup_scene, this.geometryBindGroup);
        this.scene.iterate(node => {
            gBufferPass.setBindGroup(shaders.constants.bindGroup_model, node.modelBindGroup);
        }, material => {
            gBufferPass.setBindGroup(shaders.constants.bindGroup_material, material.materialBindGroup);
        }, primitive => {
            gBufferPass.setVertexBuffer(0, primitive.vertexBuffer);
            gBufferPass.setIndexBuffer(primitive.indexBuffer, 'uint32');
            gBufferPass.drawIndexed(primitive.numIndices);
        });
        gBufferPass.end();

        // Run DDGI update passes
        if (this.stage.ddgi.enabled) {
            this.stage.ddgi.update(
                encoder,
                this.stage.scene.voxelGridView,
                this.stage.sunLightBuffer,
                this.vsm.physicalAtlasView,
                this.vsm.vsmUniformBuffer
            );
        }
        // NRC update (uses same G-buffer data)
        this.nrc.update(encoder, {
            depth: this.depthTextureView,
            normal: this.gBufferNormalTextureView,
            albedo: this.gBufferAlbedoTextureView,
            position: this.gBufferPositionTextureView,
        }, this.stage.sunLightBuffer, this.vsm.physicalAtlasView, this.vsm.vsmUniformBuffer);

        // Recreate shading bind group each frame for DDGI ping-pong atlas views
        this.ddgi.updateUniforms();
        this.createShadingBindGroup();

        // Reset light indices counter
        encoder.copyBufferToBuffer(
            this.zeroDeviceBuffer, 0,
            this.globalLightIndicesDeviceBuffer, 0,
            4
        );

        const cullingComputePass = encoder.beginComputePass();
        cullingComputePass.setPipeline(this.cullingPipeline);
        cullingComputePass.setBindGroup(shaders.constants.bindGroup_scene, this.cullingBindGroup);
        cullingComputePass.dispatchWorkgroups(
            shaders.constants.numClustersX, 
            shaders.constants.numClustersY, 
            shaders.constants.numClustersZ
        );
        cullingComputePass.end();

        const shadingRenderPass = encoder.beginRenderPass({
            label: "Shading Pass",
            colorAttachments: [
                {
                    view: canvasTextureView,
                    clearValue: [0, 0, 0, 0],
                    loadOp: "clear",
                    storeOp: "store"
                }
            ],
            depthStencilAttachment: {
                view: this.depthTextureView,
                depthReadOnly: true
            }
        });
        shadingRenderPass.setPipeline(this.shadingPipeline);
        shadingRenderPass.setBindGroup(shaders.constants.bindGroup_scene, this.shadingBindGroup);

        this.scene.iterate(node => {
            shadingRenderPass.setBindGroup(shaders.constants.bindGroup_model, node.modelBindGroup);
        }, material => {
            shadingRenderPass.setBindGroup(shaders.constants.bindGroup_material, material.materialBindGroup);
        }, primitive => {
            shadingRenderPass.setVertexBuffer(0, primitive.vertexBuffer);
            shadingRenderPass.setIndexBuffer(primitive.indexBuffer, 'uint32');
            shadingRenderPass.drawIndexed(primitive.numIndices);
        });
        shadingRenderPass.end();

        // Skybox pass - draw behind all geometry
        const skyboxPass = encoder.beginRenderPass({
            label: "Skybox Pass",
            colorAttachments: [
                {
                    view: canvasTextureView,
                    loadOp: "load",
                    storeOp: "store"
                }
            ],
            depthStencilAttachment: {
                view: this.depthTextureView,
                depthLoadOp: "load",
                depthStoreOp: "store"
            }
        });
        skyboxPass.setPipeline(this.skyboxPipeline);
        skyboxPass.setBindGroup(0, this.skyboxBindGroup);
        skyboxPass.draw(3);
        skyboxPass.end();

        // Volumetric Lighting Generation Pass (Half-Res)
        if (this.stage.sunVolumetricEnabled) {
            const volumetricPass = encoder.beginRenderPass({
                label: "Volumetric Lighting Generator Pass",
                colorAttachments: [
                    {
                        view: this.volumetricTextureView,
                        loadOp: "clear",
                        clearValue: { r: 0, g: 0, b: 0, a: 0 },
                        storeOp: "store"
                    }
                ]
            });
            volumetricPass.setPipeline(this.volumetricPipeline);
            volumetricPass.setBindGroup(0, this.volumetricBindGroup);
            volumetricPass.draw(3);
            volumetricPass.end();

            // Volumetric Composite Pass (Full-Res Upsampling)
            const volumetricCompositePass = encoder.beginRenderPass({
                label: "Volumetric Composite Pass",
                colorAttachments: [
                    {
                        view: canvasTextureView,
                        loadOp: "load",
                        storeOp: "store"
                    }
                ]
            });
            volumetricCompositePass.setPipeline(this.volumetricCompositePipeline);
            volumetricCompositePass.setBindGroup(0, this.volumetricCompositeBindGroup);
            volumetricCompositePass.draw(3);
            volumetricCompositePass.end();
        }

        renderer.device.queue.submit([encoder.finish()]);
    }
}

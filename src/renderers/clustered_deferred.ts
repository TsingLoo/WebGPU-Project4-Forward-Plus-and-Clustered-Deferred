import * as renderer from '../renderer';
import * as shaders from '../shaders/shaders';
import { Stage } from '../stage/stage';
import { DDGI } from '../stage/ddgi';

export class ClusteredDeferredRenderer extends renderer.Renderer {
    depthTexture: GPUTexture;
    depthTextureView: GPUTextureView;

    geometryAlbedoDeviceTexture: GPUTexture;
    geometryAlbedoDeviceTextureView: GPUTextureView;

    geometryNormalDeviceTexture: GPUTexture;
    geometryNormalDeviceTextureView: GPUTextureView;

    geometryPositionDeviceTexture: GPUTexture;
    geometryPositionDeviceTextureView: GPUTextureView;

    geometrySpecularDeviceTexture: GPUTexture;
    geometrySpecularDeviceTextureView: GPUTextureView;

    shadingOutputDeviceTexture: GPUTexture;
    shadingOutputDeviceTextureView: GPUTextureView;

    tileOffsetsDeviceBuffer: GPUBuffer;
    globalLightIndicesDeviceBuffer: GPUBuffer;
    zeroDeviceBuffer: GPUBuffer;
    clusterSetDeviceBuffer: GPUBuffer;

    zPrepassPipeline: GPURenderPipeline;

    geometryBindGroupLayout: GPUBindGroupLayout;
    geometryBindGroup: GPUBindGroup;
    geometryPipeline: GPURenderPipeline;

    cullingBindGroupLayout: GPUBindGroupLayout;
    cullingBindGroup: GPUBindGroup;
    cullingPipeline: GPUComputePipeline;

    shadingBindGroupLayout: GPUBindGroupLayout; 
    shadingBindGroup: GPUBindGroup;
    shadingComputePipeline: GPUComputePipeline;

    blitSampler: GPUSampler;
    blitBindGroupLayout: GPUBindGroupLayout;
    blitBindGroup: GPUBindGroup;
    blitPipeline: GPURenderPipeline;

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

    ddgi: DDGI;
    private stageEnv: import('../stage/environment').Environment;
    private stage: import('../stage/stage').Stage;

    constructor(stage: Stage) {
        super(stage);

        let geometryDeviceTextureSize = [renderer.canvas.width, renderer.canvas.height];
        const env = stage.environment;
        this.stageEnv = env;
        this.stage = stage;
        this.ddgi = stage.ddgi;

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

        this.geometryAlbedoDeviceTexture = renderer.device.createTexture({
            label: "G-Buffer Albedo Texture",
            size: geometryDeviceTextureSize,
            format: "rgba16float",
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
        });
        this.geometryAlbedoDeviceTextureView = this.geometryAlbedoDeviceTexture.createView();

        this.geometryNormalDeviceTexture = renderer.device.createTexture({
            label: "G-Buffer Normal Texture",
            size: geometryDeviceTextureSize,
            format: "rgba16float",
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
        });
        this.geometryNormalDeviceTextureView = this.geometryNormalDeviceTexture.createView();

        this.geometryPositionDeviceTexture = renderer.device.createTexture({
            label: "geometry position Texture",
            size: geometryDeviceTextureSize,
            format: "rgba16float",
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
        });
        this.geometryPositionDeviceTextureView = this.geometryPositionDeviceTexture.createView();

        this.geometrySpecularDeviceTexture = renderer.device.createTexture({
            label: "geometry specular Texture",
            size: geometryDeviceTextureSize,
            format: "rgba8unorm",
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
        });
        this.geometrySpecularDeviceTextureView = this.geometrySpecularDeviceTexture.createView();

        this.shadingOutputDeviceTexture = renderer.device.createTexture({
            label: "shading output Texture",
            size: geometryDeviceTextureSize,
            format: "rgba8unorm",
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING
        });
        this.shadingOutputDeviceTextureView = this.shadingOutputDeviceTexture.createView();

        this.tileOffsetsDeviceBuffer = renderer.device.createBuffer({
            size: shaders.constants.numTotalClustersConfig * 2 * 4,
            usage: GPUBufferUsage.STORAGE,
        });

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

        const averageLightsPerTile = shaders.constants.averageLightsPerCluster; 
        const maxIndices = shaders.constants.numTotalClustersConfig * averageLightsPerTile;

        this.globalLightIndicesDeviceBuffer = renderer.device.createBuffer({
            size: 4 + maxIndices * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            label: "global light indices buffer"
        });

        this.geometryBindGroupLayout = renderer.device.createBindGroupLayout({
            label: "geometry bind group layout",
            entries:[
                {
                    binding: 0,
                    visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                    buffer: { type: "uniform" }
                }
            ]
        });

        this.geometryBindGroup = renderer.device.createBindGroup({
            label: "geometry bind group",
            layout: this.geometryBindGroupLayout,
            entries: [
                {binding: 0, resource: { buffer: this.camera.uniformsBuffer}}
            ]
        });

        this.geometryPipeline = renderer.device.createRenderPipeline({
            label: "geometry pipeline",
            layout: renderer.device.createPipelineLayout({
                bindGroupLayouts: [
                    this.geometryBindGroupLayout,
                    renderer.modelBindGroupLayout,
                    renderer.materialBindGroupLayout,
                ]
            }),
            depthStencil: { 
                format: "depth24plus",
                depthWriteEnabled: false,
                depthCompare: "equal"
            },
            vertex: {
                module: renderer.device.createShaderModule({
                    label: "geometry vertex shader",
                    code: shaders.naiveVertSrc
                }),
                entryPoint: "main",
                buffers: [ renderer.vertexBufferLayout ]
            },
            fragment: {
                module: renderer.device.createShaderModule({
                    label: "geometry fragment shader",
                    code: shaders.geometryFragSrc
                }),
                entryPoint: "main",
                targets: [
                    { format: "rgba16float" }, 
                    { format: "rgba16float" },
                    { format: "rgba16float" },
                    { format: "rgba8unorm" },
                ]
            },
            primitive: {
                topology: "triangle-list",
                cullMode: "back"
            }
        });

        this.shadingBindGroupLayout = renderer.device.createBindGroupLayout({
            label: "shading bind group layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
                { binding: 5, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float" } },
                { binding: 6, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float" } },
                { binding: 7, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float" } },
                { binding: 8, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float" } },
                { binding: 9, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "depth" } },
                { binding: 10, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rgba8unorm" } },
                // IBL textures
                { binding: 11, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float", viewDimension: "cube" } },
                { binding: 12, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float", viewDimension: "cube" } },
                { binding: 13, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float" } },
                { binding: 14, visibility: GPUShaderStage.COMPUTE, sampler: {} },
                // DDGI textures
                { binding: 15, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },  // irradiance atlas
                { binding: 16, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },  // visibility atlas
                { binding: 17, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },       // ddgi uniforms
                { binding: 18, visibility: GPUShaderStage.COMPUTE, sampler: {} },                       // ddgi sampler
                { binding: 19, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },       // sun light
                { binding: 20, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'depth' } },   // VSM physical atlas
                { binding: 21, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // VSM page table
                { binding: 22, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },         // VSM uniforms
                { binding: 23, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },    // NRC inference output
                { binding: 24, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },         // NRC uniforms
            ]
        });

        this.zPrepassPipeline = renderer.device.createRenderPipeline({
            label: "Z-Prepass pipeline",
            layout: renderer.device.createPipelineLayout({
                bindGroupLayouts: [
                    this.geometryBindGroupLayout,
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
            fragment:{
                module: renderer.device.createShaderModule({
                    code: shaders.zPrepassFragSrc
                }),
                targets: []
            }
        });

        this.cullingBindGroupLayout = renderer.device.createBindGroupLayout({
            label: "culling bind group layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } }
            ]
        });

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
                    label: "culling compute shader",
                    code: shaders.clusteringComputeSrc
                }),
                entryPoint: "main" 
            }
        });

        this.shadingBindGroup = renderer.device.createBindGroup({
            label: "shading bind group",
            layout: this.shadingBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.camera.uniformsBuffer }},
                { binding: 1, resource: { buffer: this.lights.lightSetStorageBuffer }},
                { binding: 2, resource: { buffer: this.tileOffsetsDeviceBuffer }},
                { binding: 3, resource: { buffer: this.globalLightIndicesDeviceBuffer }},
                { binding: 4, resource: { buffer: this.clusterSetDeviceBuffer }},
                { binding: 5, resource: this.geometryAlbedoDeviceTextureView },
                { binding: 6, resource: this.geometryNormalDeviceTextureView },
                { binding: 7, resource: this.geometryPositionDeviceTextureView },
                { binding: 8, resource: this.geometrySpecularDeviceTextureView },
                { binding: 9, resource: this.depthTextureView},
                { binding: 10, resource: this.shadingOutputDeviceTextureView },
                { binding: 11, resource: env.irradianceMapView },
                { binding: 12, resource: env.prefilteredMapView },
                { binding: 13, resource: env.brdfLutView },
                { binding: 14, resource: env.envSampler },
                // DDGI
                { binding: 15, resource: this.ddgi.getCurrentIrradianceView() },
                { binding: 16, resource: this.ddgi.getCurrentVisibilityView() },
                { binding: 17, resource: { buffer: this.ddgi.ddgiUniformBuffer } },
                { binding: 18, resource: this.ddgi.ddgiSampler },
                { binding: 19, resource: { buffer: this.stage.sunLightBuffer } },
                { binding: 20, resource: this.stage.vsm.physicalAtlasView },
                { binding: 21, resource: { buffer: this.stage.vsm.pageTableBuffer } },
                { binding: 22, resource: { buffer: this.stage.vsm.vsmUniformBuffer } },
                { binding: 23, resource: this.stage.nrc.getInferenceView() },
                { binding: 24, resource: { buffer: this.stage.nrc.nrcUniformBuffer } },
            ]
        });

        this.shadingComputePipeline = renderer.device.createComputePipeline({
            label: "Shading Compute Pipeline",
            layout: renderer.device.createPipelineLayout({
                bindGroupLayouts: [ this.shadingBindGroupLayout ]
            }),
            compute: {
                module: renderer.device.createShaderModule({
                    label: "Shading Compute Shader",
                    code: shaders.clusteredDeferredComputeSrc
                }),
                entryPoint: "main"
            }
        });

        this.blitSampler = renderer.device.createSampler({
            magFilter: 'linear',
            minFilter: 'linear',
        });

        this.blitBindGroupLayout = renderer.device.createBindGroupLayout({
            label: "Blit Bind Group Layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: {} },
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: {} }
            ]
        });

        this.blitBindGroup = renderer.device.createBindGroup({
            label: "Blit Bind Group",
            layout: this.blitBindGroupLayout,
            entries: [
                { binding: 0, resource: this.shadingOutputDeviceTextureView }, 
                { binding: 1, resource: this.blitSampler }
            ]
        });

        this.blitPipeline = renderer.device.createRenderPipeline({
            label: "Blit Pipeline",
            layout: renderer.device.createPipelineLayout({
                bindGroupLayouts: [this.blitBindGroupLayout]
            }),
            vertex: {
                module: renderer.device.createShaderModule({ 
                    code: shaders.clusteredDeferredFullscreenVertSrc
                }),
                entryPoint: "main"
            },
            fragment: {
                module: renderer.device.createShaderModule({ 
                    code: shaders.clusteredDeferredFullscreenFragSrc,
                }),
                entryPoint: "main",
                targets: [{ format: renderer.canvasFormat }]
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
                { binding: 3, resource: this.stage.vsm.physicalAtlasView },
                { binding: 4, resource: { buffer: this.stage.vsm.pageTableBuffer } },
                { binding: 5, resource: { buffer: this.stage.vsm.vsmUniformBuffer } },
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

    override draw() {
        const encoder = renderer.device.createCommandEncoder();
        const canvasTextureView = renderer.context.getCurrentTexture().createView();

        // Z-Prepass (must be before VSM shadow map for page marking)
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
        zPrepass.setBindGroup(shaders.constants.bindGroup_scene, this.geometryBindGroup);
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

        // VSM shadow map pass (after Z-prepass, needs depth for page marking)
        this.stage.renderShadowMap(encoder, this.depthTextureView);



        // Geometry pass
        const geometryRenderPass = encoder.beginRenderPass({
            label: "geometry render pass",
            colorAttachments: [
                { view: this.geometryAlbedoDeviceTextureView, loadOp: "clear", clearValue: [0, 0, 0, 0], storeOp: "store" },
                { view: this.geometryNormalDeviceTextureView, loadOp: "clear", clearValue: [0, 0, 0, 0], storeOp: "store" },
                { view: this.geometryPositionDeviceTextureView, loadOp: "clear", clearValue: [0, 0, 0, 0], storeOp: "store" },
                { view: this.geometrySpecularDeviceTextureView, loadOp: "clear", clearValue: [0, 0, 0, 0], storeOp: "store" }
            ],
            depthStencilAttachment: {
                view: this.depthTextureView,
                depthReadOnly: true
            }
        });
        geometryRenderPass.setPipeline(this.geometryPipeline);
        geometryRenderPass.setBindGroup(shaders.constants.bindGroup_scene, this.geometryBindGroup);
        this.scene.iterate(node => {
            geometryRenderPass.setBindGroup(shaders.constants.bindGroup_model, node.modelBindGroup);
        }, material => {
            geometryRenderPass.setBindGroup(shaders.constants.bindGroup_material, material.materialBindGroup);
        }, primitive => {
            geometryRenderPass.setVertexBuffer(0, primitive.vertexBuffer);
            geometryRenderPass.setIndexBuffer(primitive.indexBuffer, 'uint32');
            geometryRenderPass.drawIndexed(primitive.numIndices);
        });
        geometryRenderPass.end();

        // Reset light indices counter
        encoder.copyBufferToBuffer(
            this.zeroDeviceBuffer, 0,
            this.globalLightIndicesDeviceBuffer, 0,
            4
        );

        // Light clustering
        const cullingComputePass = encoder.beginComputePass();
        cullingComputePass.setPipeline(this.cullingPipeline);
        cullingComputePass.setBindGroup(shaders.constants.bindGroup_scene, this.cullingBindGroup);
        cullingComputePass.dispatchWorkgroups(
            shaders.constants.numClustersX, 
            shaders.constants.numClustersY, 
            shaders.constants.numClustersZ
        );
        cullingComputePass.end();

        // DDGI update (after geometry, before shading)
        this.ddgi.update(encoder, {
            depth: this.depthTextureView,
            normal: this.geometryNormalDeviceTextureView,
            albedo: this.geometryAlbedoDeviceTextureView,
            position: this.geometryPositionDeviceTextureView,
        }, this.stage.sunLightBuffer, this.stage.vsm.physicalAtlasView, this.stage.vsm.vsmUniformBuffer);

        // NRC update
        this.stage.nrc.update(encoder, {
            depth: this.depthTextureView,
            normal: this.geometryNormalDeviceTextureView,
            albedo: this.geometryAlbedoDeviceTextureView,
            position: this.geometryPositionDeviceTextureView,
        }, this.stage.sunLightBuffer, this.stage.vsm.physicalAtlasView, this.stage.vsm.vsmUniformBuffer);

        // Recreate shading bind group to pick up current DDGI atlas (ping-pong)
        this.shadingBindGroup = renderer.device.createBindGroup({
            label: "shading bind group",
            layout: this.shadingBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.camera.uniformsBuffer }},
                { binding: 1, resource: { buffer: this.lights.lightSetStorageBuffer }},
                { binding: 2, resource: { buffer: this.tileOffsetsDeviceBuffer }},
                { binding: 3, resource: { buffer: this.globalLightIndicesDeviceBuffer }},
                { binding: 4, resource: { buffer: this.clusterSetDeviceBuffer }},
                { binding: 5, resource: this.geometryAlbedoDeviceTextureView },
                { binding: 6, resource: this.geometryNormalDeviceTextureView },
                { binding: 7, resource: this.geometryPositionDeviceTextureView },
                { binding: 8, resource: this.geometrySpecularDeviceTextureView },
                { binding: 9, resource: this.depthTextureView},
                { binding: 10, resource: this.shadingOutputDeviceTextureView },
                { binding: 11, resource: this.stageEnv.irradianceMapView },
                { binding: 12, resource: this.stageEnv.prefilteredMapView },
                { binding: 13, resource: this.stageEnv.brdfLutView },
                { binding: 14, resource: this.stageEnv.envSampler },
                { binding: 15, resource: this.ddgi.getCurrentIrradianceView() },
                { binding: 16, resource: this.ddgi.getCurrentVisibilityView() },
                { binding: 17, resource: { buffer: this.ddgi.ddgiUniformBuffer } },
                { binding: 18, resource: this.ddgi.ddgiSampler },
                { binding: 19, resource: { buffer: this.stage.sunLightBuffer } },
                { binding: 20, resource: this.stage.vsm.physicalAtlasView },
                { binding: 21, resource: { buffer: this.stage.vsm.pageTableBuffer } },
                { binding: 22, resource: { buffer: this.stage.vsm.vsmUniformBuffer } },
                { binding: 23, resource: this.stage.nrc.getInferenceView() },
                { binding: 24, resource: { buffer: this.stage.nrc.nrcUniformBuffer } },
            ]
        });

        // Deferred shading compute pass
        const shadingComputePass = encoder.beginComputePass();
        shadingComputePass.setPipeline(this.shadingComputePipeline);
        shadingComputePass.setBindGroup(0, this.shadingBindGroup);
        const workgroupsX = Math.ceil(renderer.canvas.width / 8);
        const workgroupsY = Math.ceil(renderer.canvas.height / 8);
        shadingComputePass.dispatchWorkgroups(workgroupsX, workgroupsY, 1);
        shadingComputePass.end();

        // Blit pass (no depth needed — fullscreen quad)
        const blitPass = encoder.beginRenderPass({
            label: "Blit Pass",
            colorAttachments: [
                {
                    view: canvasTextureView,
                    clearValue: [0, 0, 0, 1],
                    loadOp: "clear",
                    storeOp: "store"
                }
            ]
        });
        blitPass.setPipeline(this.blitPipeline);
        blitPass.setBindGroup(0, this.blitBindGroup);
        blitPass.draw(3);
        blitPass.end();

        // Skybox pass
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

        renderer.device.queue.submit([encoder.finish()]);
    }
}

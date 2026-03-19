import * as renderer from '../renderer';
import * as shaders from '../shaders/shaders';
import { Camera } from '../stage/camera';
import { Stage } from '../stage/stage';

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
    shadingBindGroup: GPUBindGroup;
    shadingPipeline: GPURenderPipeline;

    skyboxPipeline: GPURenderPipeline;
    skyboxBindGroupLayout: GPUBindGroupLayout;
    skyboxBindGroup: GPUBindGroup;

    constructor(stage: Stage) {
        super(stage);
        this.depthTexture = renderer.device.createTexture({
            size: [renderer.canvas.width, renderer.canvas.height],
            format: "depth24plus",
            usage: GPUTextureUsage.RENDER_ATTACHMENT
        });
        this.depthTextureView = this.depthTexture.createView();

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

        this.shadingBindGroup = renderer.device.createBindGroup({
            label: "shading bind group",
            layout: this.shadingBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.camera.uniformsBuffer }},
                { binding: 1, resource: { buffer: this.lights.lightSetStorageBuffer }},
                { binding: 2, resource: { buffer: this.tileOffsetsDeviceBuffer }},
                { binding: 3, resource: { buffer: this.globalLightIndicesDeviceBuffer }},
                { binding: 4, resource: { buffer: this.clusterSetDeviceBuffer }},
                { binding: 5, resource: env.irradianceMapView },
                { binding: 6, resource: env.prefilteredMapView },
                { binding: 7, resource: env.brdfLutView },
                { binding: 8, resource: env.envSampler }
            ]
        });

        this.shadingPipeline = renderer.device.createRenderPipeline({
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
    }

    override draw() {
        const encoder = renderer.device.createCommandEncoder();
        const canvasTextureView = renderer.context.getCurrentTexture().createView();

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

        //reset the global light indices counter to zero
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

        renderer.device.queue.submit([encoder.finish()]);
    }
}

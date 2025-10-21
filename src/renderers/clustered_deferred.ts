import * as renderer from '../renderer';
import * as shaders from '../shaders/shaders';
import { Stage } from '../stage/stage';

export class ClusteredDeferredRenderer extends renderer.Renderer {
// TODO-2: add layouts, pipelines, textures, etc. needed for Forward+ here
    // you may need extra uniforms such as the camera view matrix and the canvas resolution
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

    //the index of the first light in each tile and the number of lights in each tile
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
    shadingPipeline: GPURenderPipeline;

    constructor(stage: Stage) {
        super(stage);

        let geometryDeviceTextureSize = [renderer.canvas.width, renderer.canvas.height];

        // TODO-2: initialize layouts, pipelines, textures, etc. needed for Forward+ here
        this.depthTexture = renderer.device.createTexture({
            size: [renderer.canvas.width, renderer.canvas.height],
            format: "depth24plus",
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
        });
        this.depthTextureView = this.depthTexture.createView();

        this.geometryAlbedoDeviceTexture = renderer.device.createTexture({
            label: "G-Buffer Albedo Texture",
            size: geometryDeviceTextureSize,
            format: "rgba8unorm",
            usage: GPUTextureUsage.RENDER_ATTACHMENT |
                    GPUTextureUsage.TEXTURE_BINDING
        });
        this.geometryAlbedoDeviceTextureView = this.geometryAlbedoDeviceTexture.createView();

        this.geometryNormalDeviceTexture = renderer.device.createTexture({
            label: "G-Buffer Normal Texture",
            size: geometryDeviceTextureSize,
            format: "rgba16float",
            usage: GPUTextureUsage.RENDER_ATTACHMENT |
                GPUTextureUsage.TEXTURE_BINDING
        });
        this.geometryNormalDeviceTextureView = this.geometryNormalDeviceTexture.createView();

        this.geometryPositionDeviceTexture = renderer.device.createTexture({
            label: "geometry position Texture",
            size: geometryDeviceTextureSize,
            format: "rgba16float",
            usage: GPUTextureUsage.RENDER_ATTACHMENT |
                GPUTextureUsage.TEXTURE_BINDING
        })
        this.geometryPositionDeviceTextureView = this.geometryPositionDeviceTexture.createView();

        this.geometrySpecularDeviceTexture = renderer.device.createTexture({
            label: "geometry specular Texture",
            size: geometryDeviceTextureSize,
            format: "rgba8unorm",
            usage: GPUTextureUsage.RENDER_ATTACHMENT |
                GPUTextureUsage.TEXTURE_BINDING
        })
        this.geometrySpecularDeviceTextureView = this.geometrySpecularDeviceTexture.createView();

        this.tileOffsetsDeviceBuffer = renderer.device.createBuffer({
            size: shaders.constants.numTotalClustersConfig * 2 * 4, // offset and count per tile
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
        //const floatView = new Float32Array(mappedRange);
        const uintView = new Uint32Array(mappedRange);

        uintView[0] = renderer.canvas.width;
        uintView[1] = renderer.canvas.height;
        uintView[2] = shaders.constants.numClustersX;
        uintView[3] = shaders.constants.numClustersY;
        uintView[4] = shaders.constants.numClustersZ;
        this.clusterSetDeviceBuffer.unmap();

        const averageLightsPerTile = 64; 
        const maxIndices = shaders.constants.numTotalClustersConfig * averageLightsPerTile;

        this.globalLightIndicesDeviceBuffer = renderer.device.createBuffer({
            size: 4 + maxIndices * 4, // one counter and maxLights indices
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            label: "global light indices buffer"
        })

        this.geometryBindGroupLayout = renderer.device.createBindGroupLayout({
            label: "geometry bind group layout",
            entries:[
                {
                    binding: 0,
                    visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                    buffer: { type: "uniform" }
                }
            ]
        })

        this.geometryBindGroup = renderer.device.createBindGroup({
            label: "geometry bind group",
            layout: this.geometryBindGroupLayout,
            entries: [
                {binding: 0, resource: { buffer: this.camera.uniformsBuffer}}
            ]
        })

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
                    { format: "rgba8unorm" }, 
                    { format: "rgba16float" },
                    { format: "rgba16float" },
                    { format: "rgba8unorm" },
                ]
            },
            primitive: {
                topology: "triangle-list",
                cullMode: "back"
            }
        })

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
                {
                    //Tile offsets
                    binding: 2,
                    visibility: GPUShaderStage.FRAGMENT,
                    buffer: { type: "read-only-storage" } 
                },
                {
                    //Global light indices
                    binding: 3,
                    visibility: GPUShaderStage.FRAGMENT,
                    buffer: { type: "read-only-storage" } 
                },
                {
                    //Cluster set
                    binding: 4,
                    visibility: GPUShaderStage.FRAGMENT,
                    buffer: { type: "uniform" }
                },
                {
                    // gbuffer albedo
                    binding: 5,
                    visibility: GPUShaderStage.FRAGMENT,
                    texture: { sampleType: "float" }
                },
                {
                    // gbuffer normal
                    binding: 6,
                    visibility: GPUShaderStage.FRAGMENT,
                    texture: { sampleType: "unfilterable-float" }
                },
                {
                    // gbuffer position
                    binding: 7,
                    visibility: GPUShaderStage.FRAGMENT,
                    texture: { sampleType: "unfilterable-float" }
                },
                {
                    // gbuffer specular
                    binding: 8,
                    visibility: GPUShaderStage.FRAGMENT,
                    texture: { sampleType: "float" }
                },
                {
                    // gbuffer depth
                    binding: 9,
                    visibility: GPUShaderStage.FRAGMENT,
                    texture: { sampleType: "depth" }
                }
            ]
        });

        this.zPrepassPipeline = renderer.device.createRenderPipeline({
            label: "Z-Prepass pipeline",
            layout: renderer.device.createPipelineLayout({
                bindGroupLayouts: [
                    this.geometryBindGroupLayout,
                    renderer.modelBindGroupLayout,
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
        });


        this.cullingBindGroupLayout = renderer.device.createBindGroupLayout({
        label: "culling bind group layout",
        entries: [
                {
                    //Camera uniforms
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "uniform" }
                },
                {
                    //Light set
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "read-only-storage" }
                },
                {
                    //Tile offsets
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "storage" } 
                },
                {
                    //Global light indices
                    binding: 3,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "storage" } 
                },
                {
                    //Cluster set
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
                { binding: 9, resource: this.depthTextureView}
            ]
        });

        this.shadingPipeline = renderer.device.createRenderPipeline({
            label: "shading pipeline",
            layout: renderer.device.createPipelineLayout({
                bindGroupLayouts: [
                    this.shadingBindGroupLayout
                ]
            }),
            vertex: {
                module: renderer.device.createShaderModule({ code: shaders.clusteredDeferredFullscreenVertSrc, label: "final vertex(triangle) shader",}),
                entryPoint: "main"
            },
            fragment: {
                module: renderer.device.createShaderModule({
                    label: "shading fragment shader",  
                    code: shaders.clusteredDeferredFragSrc,
                }),
                entryPoint: "main",
                targets: [ { format: renderer.canvasFormat }]
            }
        });
    }

    override draw() {
        const encoder = renderer.device.createCommandEncoder();
        const canvasTextureView = renderer.context.getCurrentTexture().createView();

        const zPrepass = encoder.beginRenderPass({
            label: "naive render pass",
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
            
        }, primitive => {
            zPrepass.setVertexBuffer(0, primitive.vertexBuffer);
            zPrepass.setIndexBuffer(primitive.indexBuffer, 'uint32');
            zPrepass.drawIndexed(primitive.numIndices);
        });
        zPrepass.end();

        const geometryRenderPass = encoder.beginRenderPass({
            label: "geometry render pass",
            colorAttachments: [
                {
                    view: this.geometryAlbedoDeviceTextureView, 
                    loadOp: "clear",
                    clearValue: [0, 0, 0, 0],
                    storeOp: "store"
                },
                {
                    view: this.geometryNormalDeviceTextureView, 
                    loadOp: "clear",
                    clearValue: [0, 0, 0, 0],
                    storeOp: "store"
                },
                {
                    view: this.geometryPositionDeviceTextureView, 
                    loadOp: "clear",
                    clearValue: [0, 0, 0, 0],
                    storeOp: "store"
                },
                {
                    view: this.geometrySpecularDeviceTextureView,
                    loadOp: "clear",
                    clearValue: [0, 0, 0, 0],
                    storeOp: "store"
                }
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
            ]
        });
        shadingRenderPass.setPipeline(this.shadingPipeline);
        shadingRenderPass.setBindGroup(shaders.constants.bindGroup_scene, this.shadingBindGroup);

        shadingRenderPass.draw(3, 1, 0, 0);

        shadingRenderPass.end();

        renderer.device.queue.submit([encoder.finish()]);
    }
}

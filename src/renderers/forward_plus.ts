import * as renderer from '../renderer';
import * as shaders from '../shaders/shaders';
import { Camera } from '../stage/camera';
import { Stage } from '../stage/stage';

export class ForwardPlusRenderer extends renderer.Renderer {
    // TODO-2: add layouts, pipelines, textures, etc. needed for Forward+ here
    // you may need extra uniforms such as the camera view matrix and the canvas resolution
    depthTexture: GPUTexture;
    depthTextureView: GPUTextureView;

    //the index of the first light in each tile and the number of lights in each tile
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

    constructor(stage: Stage) {
        super(stage);
        // TODO-2: initialize layouts, pipelines, textures, etc. needed for Forward+ here
        this.depthTexture = renderer.device.createTexture({
            size: [renderer.canvas.width, renderer.canvas.height],
            format: "depth24plus",
            usage: GPUTextureUsage.RENDER_ATTACHMENT
        });
        this.depthTextureView = this.depthTexture.createView();

        this.tileOffsetsDeviceBuffer = renderer.device.createBuffer({
            size: shaders.constants.totalTilesCount * 2 * 4, // offset and count per tile
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
            size: 4 * 7,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true
        });
        const mappedRange = this.clusterSetDeviceBuffer.getMappedRange();
        const floatView = new Float32Array(mappedRange);
        const uintView = new Uint32Array(mappedRange);

        floatView[0] = renderer.canvas.width;
        floatView[1] = renderer.canvas.height;
        floatView[5] = Camera.nearPlane;
        floatView[6] = Camera.farPlane;
        uintView[2] = shaders.constants.tilesizeX;
        uintView[3] = shaders.constants.tilesizeY;
        uintView[4] = shaders.constants.tilesizeZ;
        this.clusterSetDeviceBuffer.unmap();

        const averageLightsPerTile = 64; 
        const maxIndices = shaders.constants.totalTilesCount * averageLightsPerTile;

        this.globalLightIndicesDeviceBuffer = renderer.device.createBuffer({
            size: 4 + maxIndices * 4, // one counter and maxLights indices
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            label: "global light indices buffer"
        })

        this.shadingBindGroupLayout = renderer.device.createBindGroupLayout({
            label: "shading bind group layout",
            entries: [
                { // Camera Uniforms
                    binding: 0,
                    visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                    buffer: { type: "uniform" }
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.FRAGMENT,
                    buffer: { type: "read-only-storage" }
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.FRAGMENT,
                    buffer: { type: "read-only-storage" } 
                },
                {
                    binding: 3,
                    visibility: GPUShaderStage.FRAGMENT,
                    buffer: { type: "read-only-storage" } 
                }
            ]
        });

        this.zPrepassPipeline = renderer.device.createRenderPipeline({
            label: "Z-Prepass pipeline",
            layout: renderer.device.createPipelineLayout({
                bindGroupLayouts: [
                    this.shadingBindGroupLayout,
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
                { binding: 3, resource: { buffer: this.globalLightIndicesDeviceBuffer }}
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
        zPrepass.setBindGroup(shaders.constants.bindGroup_scene, this.shadingBindGroup);
        this.scene.iterate(node => {
            zPrepass.setBindGroup(shaders.constants.bindGroup_model, node.modelBindGroup);
        }, material => {
            
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

        const cullingPass = encoder.beginComputePass();
        cullingPass.setPipeline(this.cullingPipeline);
        cullingPass.setBindGroup(shaders.constants.bindGroup_scene, this.cullingBindGroup);
        cullingPass.dispatchWorkgroups(
            shaders.constants.tilesizeX, 
            shaders.constants.tilesizeY, 
            shaders.constants.tilesizeZ
        );
        cullingPass.end();

        const shadingPass = encoder.beginRenderPass({
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
                depthLoadOp: "load",
                depthStoreOp: "discard",
                depthReadOnly: true
            }
        });
        shadingPass.setPipeline(this.shadingPipeline);
        shadingPass.setBindGroup(shaders.constants.bindGroup_scene, this.shadingBindGroup);

        this.scene.iterate(node => {
            shadingPass.setBindGroup(shaders.constants.bindGroup_model, node.modelBindGroup);
        }, material => {
            shadingPass.setBindGroup(shaders.constants.bindGroup_material, material.materialBindGroup);
        }, primitive => {
            shadingPass.setVertexBuffer(0, primitive.vertexBuffer);
            shadingPass.setIndexBuffer(primitive.indexBuffer, 'uint32');
            shadingPass.drawIndexed(primitive.numIndices);
        });
        shadingPass.end();

        renderer.device.queue.submit([encoder.finish()]);
    }
}

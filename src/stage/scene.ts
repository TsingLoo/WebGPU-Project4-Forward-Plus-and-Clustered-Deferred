/*
Note that this glTF loader assumes a lot of things are always defined (textures, samplers, vertex/index info, etc.),
so you may run into issues loading files outside of the Sponza scene.

In particular, it is known to not work if there is a mesh with no material.
*/

import { registerLoaders, load, parse } from '@loaders.gl/core';
import { GLTFLoader, GLTFWithBuffers, GLTFMesh, GLTFMeshPrimitive, GLTFMaterial, GLTFSampler } from '@loaders.gl/gltf';
import { ImageLoader } from '@loaders.gl/images';
import { Mat4, mat4 } from 'wgpu-matrix';
import { device, materialBindGroupLayout, modelBindGroupLayout } from '../renderer';

export function setupLoaders() {
    registerLoaders([GLTFLoader, ImageLoader]);
}

function getFloatArray(gltfWithBuffers: GLTFWithBuffers, attribute: number) {
    const gltf = gltfWithBuffers.json;
    const accessor = gltf.accessors![attribute];
    const bufferView = gltf.bufferViews![accessor.bufferView!];
    const buffer = gltfWithBuffers.buffers[bufferView.buffer];
    const byteOffset = (accessor.byteOffset ?? 0) + (bufferView.byteOffset ?? 0) + buffer.byteOffset;
    return new Float32Array(buffer.arrayBuffer, byteOffset, bufferView.byteLength / 4);
}

class Texture {
    image: GPUTexture;
    sampler: GPUSampler;

    constructor(image: GPUTexture, sampler: GPUSampler) {
        this.image = image;
        this.sampler = sampler;
    }
}

export class Material {
    private static nextId = 0;
    readonly id: number;

    materialBindGroup: GPUBindGroup;

    constructor(gltfMaterial: GLTFMaterial, textures: Texture[], defaultTexture: Texture) {
        this.id = Material.nextId++;

        const texIndex = gltfMaterial.pbrMetallicRoughness?.baseColorTexture?.index;
        const diffuseTexture = (texIndex != null && texIndex < textures.length) ? textures[texIndex] : defaultTexture;

        // Metallic-roughness texture (glTF: green channel = roughness, blue channel = metallic)
        const mrTexIndex = gltfMaterial.pbrMetallicRoughness?.metallicRoughnessTexture?.index;
        const mrTexture = (mrTexIndex != null && mrTexIndex < textures.length) ? textures[mrTexIndex] : defaultTexture;

        // Extract PBR scalar factors from glTF
        // Note: glTF spec defaults are both 1.0, but Sponza materials are dielectric
        // so we default metallic to 0.0 when not specified
        const roughness = gltfMaterial.pbrMetallicRoughness?.roughnessFactor ?? 1.0;
        const metallic = gltfMaterial.pbrMetallicRoughness?.metallicFactor ?? 0.0;
        const baseColorFactor = gltfMaterial.pbrMetallicRoughness?.baseColorFactor ?? [1.0, 1.0, 1.0, 1.0];

        // Flag: does this material have a metallic-roughness texture?
        const hasMRTexture = (mrTexIndex != null && mrTexIndex < textures.length) ? 1.0 : 0.0;

        // PBR params uniform: roughness, metallic, hasMRTexture, pad, baseColorFactor (vec4f) = 32 bytes
        const pbrParamsBuffer = device.createBuffer({
            label: "PBR params uniform",
            size: 32,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(pbrParamsBuffer, 0, new Float32Array([
            roughness, metallic, hasMRTexture, 0.0,
            baseColorFactor[0], baseColorFactor[1], baseColorFactor[2], baseColorFactor[3]
        ]));

        this.materialBindGroup = device.createBindGroup({
            label: "material bind group",
            layout: materialBindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: diffuseTexture.image.createView()
                },
                {
                    binding: 1,
                    resource: diffuseTexture.sampler
                },
                {
                    binding: 2,
                    resource: { buffer: pbrParamsBuffer }
                },
                {
                    binding: 3,
                    resource: mrTexture.image.createView()
                },
                {
                    binding: 4,
                    resource: mrTexture.sampler
                }
            ]
        });
    }
}

export class Primitive {
    vertexBuffer: GPUBuffer;
    indexBuffer: GPUBuffer;
    numIndices = -1;

    material: Material;

    constructor(gltfPrim: GLTFMeshPrimitive, gltfWithBuffers: GLTFWithBuffers, material: Material) {
        this.material = material;

        const gltf = gltfWithBuffers.json;

        const indicesAccessor = gltf.accessors![gltfPrim.indices!];
        const indicesBufferView = gltf.bufferViews![indicesAccessor.bufferView!];
        const indicesDataType = indicesAccessor.componentType;
        const indicesBuffer = gltfWithBuffers.buffers[indicesBufferView.buffer];
        const indicesByteOffset = (indicesAccessor.byteOffset ?? 0)
            + (indicesBufferView.byteOffset ?? 0)
            + indicesBuffer.byteOffset;
        let indicesArray: Uint32Array;
        switch (indicesDataType) {
            case 0x1403: // UNSIGNED_SHORT
                indicesArray = Uint32Array.from(
                    new Uint16Array(indicesBuffer.arrayBuffer, indicesByteOffset, indicesAccessor.count));
                break;
            case 0x1405: // UNSIGNED_INT (untested)
                indicesArray = new Uint32Array(indicesBuffer.arrayBuffer, indicesByteOffset, indicesAccessor.count);
                break;
            default:
                throw new Error(`unsupported index buffer element component type: 0x${indicesDataType.toString(16)}`);
        }

        const positionsArray = getFloatArray(gltfWithBuffers, gltfPrim.attributes.POSITION);
        const normalsArray = getFloatArray(gltfWithBuffers, gltfPrim.attributes.NORMAL);
        const uvsArray = getFloatArray(gltfWithBuffers, gltfPrim.attributes.TEXCOORD_0);

        const numFloatsPerVert = 8;
        const numVerts = positionsArray.length / 3;
        const vertsArray = new Float32Array(numVerts * numFloatsPerVert);
        for (let vertIdx = 0; vertIdx < numVerts; ++vertIdx) {
            const vertStartIdx = vertIdx * numFloatsPerVert;
            vertsArray[vertStartIdx] = positionsArray[vertIdx * 3];
            vertsArray[vertStartIdx + 1] = positionsArray[vertIdx * 3 + 1];
            vertsArray[vertStartIdx + 2] = positionsArray[vertIdx * 3 + 2];
            vertsArray[vertStartIdx + 3] = normalsArray[vertIdx * 3];
            vertsArray[vertStartIdx + 4] = normalsArray[vertIdx * 3 + 1];
            vertsArray[vertStartIdx + 5] = normalsArray[vertIdx * 3 + 2];
            vertsArray[vertStartIdx + 6] = uvsArray[vertIdx * 2];
            vertsArray[vertStartIdx + 7] = uvsArray[vertIdx * 2 + 1];
        }

        this.indexBuffer = device.createBuffer({
            label: "index buffer",
            size: indicesArray.byteLength,
            usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(this.indexBuffer, 0, indicesArray);

        this.vertexBuffer = device.createBuffer({
            label: "vertex buffer",
            size: vertsArray.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(this.vertexBuffer, 0, vertsArray);

        this.numIndices = indicesArray.length;
    }
}

export class Mesh {
    primitives: Primitive[] = [];

    constructor(gltfMesh: GLTFMesh, gltfWithBuffers: GLTFWithBuffers, sceneMaterials: Material[]) {
        gltfMesh.primitives.forEach((gltfPrim: GLTFMeshPrimitive) => {
            const matIdx = gltfPrim.material ?? 0;
            if (matIdx < sceneMaterials.length) {
                this.primitives.push(new Primitive(gltfPrim, gltfWithBuffers, sceneMaterials[matIdx]));
            }
        });

        this.primitives.sort((primA: Primitive, primB: Primitive) => {
            return primA.material.id - primB.material.id;
        });
    }
}

export class Node {
    name: String = "node";

    parent: Node | undefined;
    children: Set<Node> = new Set<Node>();

    transform: Mat4 = mat4.identity();
    modelMatUniformBuffer!: GPUBuffer;
    modelBindGroup!: GPUBindGroup;
    mesh: Mesh | undefined;

    setName(newName: string) {
        this.name = newName;
    }

    setParent(newParent: Node) {
        if (this.parent != undefined) {
            this.parent.children.delete(this);
        }

        this.parent = newParent;
        newParent.children.add(this);
    }

    propagateTransformations() {
        if (this.parent != undefined) {
            this.transform = mat4.mul(this.parent.transform, this.transform);
        }

        if (this.mesh != undefined) {
            this.modelMatUniformBuffer = device.createBuffer({
                label: "model mat uniform",
                size: 16 * 4,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            });

            device.queue.writeBuffer(this.modelMatUniformBuffer, 0, this.transform);

            this.modelBindGroup = device.createBindGroup({
                label: "model bind group",
                layout: modelBindGroupLayout,
                entries: [
                    {
                        binding: 0,
                        resource: { buffer: this.modelMatUniformBuffer }
                    }
                ]
            });
        }

        for (let child of this.children) {
            child.propagateTransformations();
        }
    }
}

function createTexture(imageBitmap: ImageBitmap): GPUTexture {
    let texture = device.createTexture({
        size: [imageBitmap.width, imageBitmap.height],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT
    });

    device.queue.copyExternalImageToTexture(
        { source: imageBitmap },
        { texture: texture },
        { width: imageBitmap.width, height: imageBitmap.height }
    );

    return texture;
}

function convertWrapModeEnum(wrapMode: number | undefined): GPUAddressMode {
    switch (wrapMode) {
        case 0x2901: // REPEAT
            return 'repeat';
        case 0x812F: // CLAMP_TO_EDGE
            return 'clamp-to-edge';
        case 0x8370: // MIRRORED_REPEAT
            return 'mirror-repeat';
        default:
            return 'repeat'; // default fallback
    }
}

function createSampler(gltfSampler: GLTFSampler): GPUSampler {
    let samplerDescriptor: GPUSamplerDescriptor = {};

    switch (gltfSampler.magFilter) {
        case 0x2600: // NEAREST
            samplerDescriptor.magFilter = 'nearest';
            break;
        case 0x2601: // LINEAR
        default:
            samplerDescriptor.magFilter = 'linear';
            break;
    }

    switch (gltfSampler.minFilter) {
        case 0x2600: // NEAREST
            samplerDescriptor.minFilter = 'nearest';
            break;
        case 0x2700: // NEAREST_MIPMAP_NEAREST
            samplerDescriptor.minFilter = 'nearest';
            samplerDescriptor.mipmapFilter = 'nearest';
            break;
        case 0x2701: // LINEAR_MIPMAP_NEAREST
            samplerDescriptor.minFilter = 'linear';
            samplerDescriptor.mipmapFilter = 'nearest';
            break;
        case 0x2702: // NEAREST_MIPMAP_LINEAR
            samplerDescriptor.minFilter = 'nearest';
            samplerDescriptor.mipmapFilter = 'linear';
            break;
        case 0x2703: // LINEAR_MIPMAP_LINEAR
            samplerDescriptor.minFilter = 'linear';
            samplerDescriptor.mipmapFilter = 'linear';
            break;
        case 0x2601: // LINEAR
        default:
            samplerDescriptor.minFilter = 'linear';
            break;
    }

    samplerDescriptor.addressModeU = convertWrapModeEnum(gltfSampler.wrapS);
    samplerDescriptor.addressModeV = convertWrapModeEnum(gltfSampler.wrapT);

    return device.createSampler(samplerDescriptor);
}

export class Scene {
    private root: Node = new Node();

    constructor() {
        this.root.setName("root");
    }

    async loadGltfBuffer(buffer: ArrayBuffer) {
        const gltfWithBuffers = await parse(buffer, GLTFLoader) as unknown as GLTFWithBuffers;
        return this.processGltf(gltfWithBuffers);
    }

    async loadGltf(filePath: string) {
        const gltfWithBuffers = await load(filePath) as GLTFWithBuffers;
        return this.processGltf(gltfWithBuffers);
    }

    private processGltf(gltfWithBuffers: GLTFWithBuffers) {
        const gltf = gltfWithBuffers.json;

        // Create a default white 1x1 texture for materials without a baseColorTexture
        const defaultGpuTex = device.createTexture({
            size: [1, 1],
            format: 'rgba8unorm',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
        });
        device.queue.writeTexture(
            { texture: defaultGpuTex },
            new Uint8Array([255, 255, 255, 255]),
            { bytesPerRow: 4 },
            [1, 1]
        );
        const defaultSampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
        const defaultTexture = new Texture(defaultGpuTex, defaultSampler);

        let sceneTextures: Texture[] = [];
        {
            let sceneImages: GPUTexture[] = [];
            if (gltfWithBuffers.images) {
                for (let gltfImage of gltfWithBuffers.images) {
                    sceneImages.push(createTexture(gltfImage as ImageBitmap));
                }
            }

            let sceneSamplers: GPUSampler[] = [];
            if (gltf.samplers) {
                for (let gltfSampler of gltf.samplers) {
                    sceneSamplers.push(createSampler(gltfSampler));
                }
            }

            if (gltf.textures) {
                for (let gltfTexture of gltf.textures) {
                    const img = (gltfTexture.source != null) ? sceneImages[gltfTexture.source] : defaultGpuTex;
                    const smp = (gltfTexture.sampler != null && gltfTexture.sampler < sceneSamplers.length) ? sceneSamplers[gltfTexture.sampler] : defaultSampler;
                    sceneTextures.push(new Texture(img, smp));
                }
            }
        }

        let sceneMaterials: Material[] = [];
        if (gltf.materials) {
            for (let gltfMaterial of gltf.materials) {
                sceneMaterials.push(new Material(gltfMaterial, sceneTextures, defaultTexture));
            }
        }

        let sceneMeshes: Mesh[] = [];
        for (let gltfMesh of gltf.meshes!) {
            sceneMeshes.push(new Mesh(gltfMesh, gltfWithBuffers, sceneMaterials));
        }

        let sceneRoot: Node = new Node();
        sceneRoot.setName("scene root");
        sceneRoot.setParent(this.root);

        let sceneNodes: Node[] = [];
        for (let gltfNode of gltf.nodes!) {
            let newNode = new Node();
            newNode.setName(gltfNode.name);
            newNode.setParent(sceneRoot);

            if (gltfNode.mesh != undefined) {
                newNode.mesh = sceneMeshes[gltfNode.mesh];
            }

            if (gltfNode.matrix != undefined) {
                newNode.transform = new Float32Array(gltfNode.matrix);
            } else {
                if (gltfNode.translation != undefined) {
                    newNode.transform = mat4.mul(newNode.transform, mat4.translation(gltfNode.translation));
                }

                if (gltfNode.rotation != undefined) {
                    newNode.transform = mat4.mul(newNode.transform, mat4.fromQuat(gltfNode.rotation));
                }

                if (gltfNode.scale != undefined) {
                    newNode.transform = mat4.mul(newNode.transform, mat4.scaling(gltfNode.scale));
                }
            }

            sceneNodes.push(newNode);
        }

        for (let nodeIdx in gltf.nodes!) {
            const gltfNode = gltf.nodes[nodeIdx];

            if (gltfNode.children == undefined) {
                continue;
            }

            for (let childNodeIdx of gltfNode.children) {
                sceneNodes[childNodeIdx].setParent(sceneNodes[nodeIdx]);
            }
        }

        sceneRoot.propagateTransformations();
    }

    iterate(nodeFunction: (node: Node) => void, materialFunction: (material: Material) => void,
        primFunction: (primitive: Primitive) => void) {
        let nodes = [this.root];

        let lastMaterialId: number | undefined = undefined;

        while (nodes.length > 0) {
            let node = nodes.pop() as Node;
            if (node.mesh != undefined) {
                nodeFunction(node);

                for (let primitive of node.mesh.primitives) {
                    if (primitive.material.id != lastMaterialId) {
                        materialFunction(primitive.material);
                        lastMaterialId = primitive.material.id;
                    }

                    primFunction(primitive);
                }
            }

            for (let childNode of node.children) {
                nodes.push(childNode);
            }
        }
    }
}

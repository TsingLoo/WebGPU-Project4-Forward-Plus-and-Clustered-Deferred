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

    // Determine number of components per element from accessor type
    const typeToComponents: Record<string, number> = {
        'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4,
        'MAT2': 4, 'MAT3': 9, 'MAT4': 16
    };
    const numComponents = typeToComponents[accessor.type] || 1;
    const totalFloats = accessor.count * numComponents;

    // If buffer view has byte stride and it differs from tightly packed,
    // we need to extract data element-by-element
    const tightStride = numComponents * 4; // 4 bytes per float
    const byteStride = bufferView.byteStride ?? tightStride;

    if (byteStride === tightStride) {
        // Tightly packed — can create a direct view
        return new Float32Array(buffer.arrayBuffer, byteOffset, totalFloats);
    } else {
        // Strided — extract element by element
        const result = new Float32Array(totalFloats);
        const dataView = new DataView(buffer.arrayBuffer);
        for (let i = 0; i < accessor.count; i++) {
            const elemOffset = byteOffset + i * byteStride;
            for (let c = 0; c < numComponents; c++) {
                result[i * numComponents + c] = dataView.getFloat32(elemOffset + c * 4, true);
            }
        }
        return result;
    }
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

    constructor(gltfMaterial: GLTFMaterial, texturesSRGB: Texture[], texturesLinear: Texture[], defaultTextureSRGB: Texture, defaultTextureLinear: Texture) {
        this.id = Material.nextId++;

        // BaseColor texture uses sRGB (gamma-encoded color data)
        const texIndex = gltfMaterial.pbrMetallicRoughness?.baseColorTexture?.index;
        const diffuseTexture = (texIndex != null && texIndex < texturesSRGB.length) ? texturesSRGB[texIndex] : defaultTextureSRGB;

        // Metallic-roughness texture uses linear (non-color data: G = roughness, B = metallic)
        const mrTexIndex = gltfMaterial.pbrMetallicRoughness?.metallicRoughnessTexture?.index;
        const mrTexture = (mrTexIndex != null && mrTexIndex < texturesLinear.length) ? texturesLinear[mrTexIndex] : defaultTextureLinear;

        // Normal texture uses linear (non-color data)
        const normalTexIndex = (gltfMaterial as any).normalTexture?.index;
        const normalTexture = (normalTexIndex != null && normalTexIndex < texturesLinear.length) ? texturesLinear[normalTexIndex] : defaultTextureLinear;

        // Extract PBR scalar factors from glTF
        const roughness = gltfMaterial.pbrMetallicRoughness?.roughnessFactor ?? 1.0;
        const metallic = gltfMaterial.pbrMetallicRoughness?.metallicFactor ?? 0.0;
        const baseColorFactor = gltfMaterial.pbrMetallicRoughness?.baseColorFactor ?? [1.0, 1.0, 1.0, 1.0];

        // Flag: does this material have a metallic-roughness texture?
        const hasMRTexture = (mrTexIndex != null && mrTexIndex < texturesLinear.length) ? 1.0 : 0.0;
        const hasNormalTexture = (normalTexIndex != null && normalTexIndex < texturesLinear.length) ? 1.0 : 0.0;


        // PBR params uniform: 48 bytes (3 vec4f)
        // vec4f[0]: roughness, metallic, hasMRTexture, hasNormalTexture
        // vec4f[1]: baseColorFactor
        // vec4f[2]: reserved
        const pbrParamsBuffer = device.createBuffer({
            label: "PBR params uniform",
            size: 48,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(pbrParamsBuffer, 0, new Float32Array([
            roughness, metallic, hasMRTexture, hasNormalTexture,
            baseColorFactor[0], baseColorFactor[1], baseColorFactor[2], baseColorFactor[3],
            0.0, 0.0, 0.0, 0.0
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
                },
                {
                    binding: 5,
                    resource: normalTexture.image.createView()
                },
                {
                    binding: 6,
                    resource: normalTexture.sampler
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
    
    cpuPositions?: Float32Array;
    cpuIndices?: Uint32Array;

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

        // Load tangent data (vec4f: xyz = tangent direction, w = handedness +1/-1)
        let tangentsArray: Float32Array | null = null;
        if (gltfPrim.attributes.TANGENT != null) {
            tangentsArray = getFloatArray(gltfWithBuffers, gltfPrim.attributes.TANGENT);
        }

        const numFloatsPerVert = 12; // pos(3) + nor(3) + uv(2) + tangent(4)
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
            // Tangent (vec4f)
            if (tangentsArray) {
                vertsArray[vertStartIdx + 8] = tangentsArray[vertIdx * 4];
                vertsArray[vertStartIdx + 9] = tangentsArray[vertIdx * 4 + 1];
                vertsArray[vertStartIdx + 10] = tangentsArray[vertIdx * 4 + 2];
                vertsArray[vertStartIdx + 11] = tangentsArray[vertIdx * 4 + 3]; // handedness
            } else {
                // Default tangent along +X axis with positive handedness
                vertsArray[vertStartIdx + 8] = 1.0;
                vertsArray[vertStartIdx + 9] = 0.0;
                vertsArray[vertStartIdx + 10] = 0.0;
                vertsArray[vertStartIdx + 11] = 1.0;
            }
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
        
        // Save for CPU voxelization
        this.cpuPositions = positionsArray;
        this.cpuIndices = indicesArray;
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

function createTextureSRGB(imageBitmap: ImageBitmap): GPUTexture {
    let texture = device.createTexture({
        size: [imageBitmap.width, imageBitmap.height],
        format: 'rgba8unorm-srgb',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT
    });

    device.queue.copyExternalImageToTexture(
        { source: imageBitmap },
        { texture: texture },
        { width: imageBitmap.width, height: imageBitmap.height }
    );

    return texture;
}

function createTextureLinear(imageBitmap: ImageBitmap): GPUTexture {
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
    
    // Coarse Scene Voxel Grid for Ray Tracing
    public voxelGrid!: GPUTexture;
    public voxelGridView!: GPUTextureView;
    public readonly voxelGridSize = 128; // 128x128x128
    public readonly voxelBoundsMin = [-15, 0, -10];
    public readonly voxelBoundsMax = [15, 15, 10];

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

        // Create default white 1x1 textures (sRGB for baseColor, linear for MR)
        const defaultGpuTexSRGB = device.createTexture({
            size: [1, 1],
            format: 'rgba8unorm-srgb',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
        });
        device.queue.writeTexture(
            { texture: defaultGpuTexSRGB },
            new Uint8Array([255, 255, 255, 255]),
            { bytesPerRow: 4 },
            [1, 1]
        );
        const defaultGpuTexLinear = device.createTexture({
            size: [1, 1],
            format: 'rgba8unorm',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
        });
        device.queue.writeTexture(
            { texture: defaultGpuTexLinear },
            new Uint8Array([255, 255, 255, 255]),
            { bytesPerRow: 4 },
            [1, 1]
        );
        const defaultSampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
        const defaultTextureSRGB = new Texture(defaultGpuTexSRGB, defaultSampler);
        const defaultTextureLinear = new Texture(defaultGpuTexLinear, defaultSampler);

        // Build sRGB and linear image arrays from all images
        let sceneTexturesSRGB: Texture[] = [];  // for baseColor (sRGB encoded)
        let sceneTexturesLinear: Texture[] = []; // for metallic-roughness (linear data)
        {
            let sceneImagesSRGB: GPUTexture[] = [];
            let sceneImagesLinear: GPUTexture[] = [];
            if (gltfWithBuffers.images) {
                for (let gltfImage of gltfWithBuffers.images) {
                    sceneImagesSRGB.push(createTextureSRGB(gltfImage as ImageBitmap));
                    sceneImagesLinear.push(createTextureLinear(gltfImage as ImageBitmap));
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
                    const smp = (gltfTexture.sampler != null && gltfTexture.sampler < sceneSamplers.length) ? sceneSamplers[gltfTexture.sampler] : defaultSampler;
                    const imgSRGB = (gltfTexture.source != null) ? sceneImagesSRGB[gltfTexture.source] : defaultGpuTexSRGB;
                    const imgLinear = (gltfTexture.source != null) ? sceneImagesLinear[gltfTexture.source] : defaultGpuTexLinear;
                    sceneTexturesSRGB.push(new Texture(imgSRGB, smp));
                    sceneTexturesLinear.push(new Texture(imgLinear, smp));
                }
            }
        }

        let sceneMaterials: Material[] = [];
        if (gltf.materials) {
            for (let gltfMaterial of gltf.materials) {
                sceneMaterials.push(new Material(gltfMaterial, sceneTexturesSRGB, sceneTexturesLinear, defaultTextureSRGB, defaultTextureLinear));
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
        
        // Phase 2: Build World-Space Voxel Grid on CPU for Coarse Scene Tracing!
        this.buildVoxelGrid();
    }
    
    private buildVoxelGrid() {
        const size = this.voxelGridSize;
        const totalVoxels = size * size * size;
        const voxelData = new Uint8Array(totalVoxels * 4); // RGBA8Unorm
        voxelData.fill(0);
        
        const minX = this.voxelBoundsMin[0], minY = this.voxelBoundsMin[1], minZ = this.voxelBoundsMin[2];
        const maxX = this.voxelBoundsMax[0], maxY = this.voxelBoundsMax[1], maxZ = this.voxelBoundsMax[2];
        
        console.log(`Building Voxel Grid (${size}^3) on CPU...`);
        let nodes = [this.root];
        while (nodes.length > 0) {
            let node = nodes.pop()!;
            if (node.mesh) {
                for (let prim of node.mesh.primitives) {
                    if (!prim.cpuPositions || !prim.cpuIndices) continue;
                    
                    const pos = prim.cpuPositions;
                    const ind = prim.cpuIndices;
                    const mat = node.transform;
                    
                    for (let i = 0; i < prim.numIndices; i += 3) {
                        const i0 = ind[i] * 3, i1 = ind[i+1] * 3, i2 = ind[i+2] * 3;
                        
                        const v0x = pos[i0]*mat[0] + pos[i0+1]*mat[4] + pos[i0+2]*mat[8] + mat[12];
                        const v0y = pos[i0]*mat[1] + pos[i0+1]*mat[5] + pos[i0+2]*mat[9] + mat[13];
                        const v0z = pos[i0]*mat[2] + pos[i0+1]*mat[6] + pos[i0+2]*mat[10] + mat[14];
                        
                        const v1x = pos[i1]*mat[0] + pos[i1+1]*mat[4] + pos[i1+2]*mat[8] + mat[12];
                        const v1y = pos[i1]*mat[1] + pos[i1+1]*mat[5] + pos[i1+2]*mat[9] + mat[13];
                        const v1z = pos[i1]*mat[2] + pos[i1+1]*mat[6] + pos[i1+2]*mat[10] + mat[14];
                        
                        const v2x = pos[i2]*mat[0] + pos[i2+1]*mat[4] + pos[i2+2]*mat[8] + mat[12];
                        const v2y = pos[i2]*mat[1] + pos[i2+1]*mat[5] + pos[i2+2]*mat[9] + mat[13];
                        const v2z = pos[i2]*mat[2] + pos[i2+1]*mat[6] + pos[i2+2]*mat[10] + mat[14];
                        
                        // Compute geometric normal
                        let nx = (v1y - v0y)*(v2z - v0z) - (v1z - v0z)*(v2y - v0y);
                        let ny = (v1z - v0z)*(v2x - v0x) - (v1x - v0x)*(v2z - v0z);
                        let nz = (v1x - v0x)*(v2y - v0y) - (v1y - v0y)*(v2x - v0x);
                        const nLen = Math.sqrt(nx*nx + ny*ny + nz*nz);
                        if (nLen > 0) { nx /= nLen; ny /= nLen; nz /= nLen; }
                        else { nx = 0; ny = 1; nz = 0; }
                        
                        const r = Math.floor((nx * 0.5 + 0.5) * 255);
                        const g = Math.floor((ny * 0.5 + 0.5) * 255);
                        const b = Math.floor((nz * 0.5 + 0.5) * 255);
                        const a = 255;
                        
                        const dx = (maxX - minX) / size;
                        const dy = (maxY - minY) / size;
                        const dz = (maxZ - minZ) / size;
                        const voxelDiag = Math.sqrt(dx*dx + dy*dy + dz*dz);
                        
                        const len1 = Math.sqrt((v1x-v0x)**2 + (v1y-v0y)**2 + (v1z-v0z)**2);
                        const len2 = Math.sqrt((v2x-v0x)**2 + (v2y-v0y)**2 + (v2z-v0z)**2);
                        const len3 = Math.sqrt((v2x-v1x)**2 + (v2y-v1y)**2 + (v2z-v1z)**2);
                        const maxEdge = Math.max(len1, len2, len3);
                        
                        const steps = Math.max(1, Math.ceil(maxEdge / (voxelDiag * 0.5)));
                        
                        for (let s1 = 0; s1 <= steps; s1++) {
                            const u = s1 / steps;
                            for (let s2 = 0; s2 <= steps - s1; s2++) {
                                const v = s2 / steps;
                                const w = Math.max(0, 1.0 - u - v);
                                
                                const px = v0x*u + v1x*v + v2x*w;
                                const py = v0y*u + v1y*v + v2y*w;
                                const pz = v0z*u + v1z*v + v2z*w;
                                
                                const vx = Math.floor((px - minX) / dx);
                                const vy = Math.floor((py - minY) / dy);
                                const vz = Math.floor((pz - minZ) / dz);
                                
                                if (vx >= 0 && vx < size && vy >= 0 && vy < size && vz >= 0 && vz < size) {
                                    const idx = (vz * size * size + vy * size + vx) * 4;
                                    voxelData[idx] = r;
                                    voxelData[idx+1] = g;
                                    voxelData[idx+2] = b;
                                    voxelData[idx+3] = a;
                                }
                            }
                        }
                    }
                    
                    // Free CPU memory
                    prim.cpuPositions = undefined;
                    prim.cpuIndices = undefined;
                }
            }
            for (let child of node.children) {
                nodes.push(child);
            }
        }
        
        console.log("CPU Voxelization Complete.");
        
        this.voxelGrid = device.createTexture({
            label: "World Space Voxel Grid",
            size: [size, size, size],
            format: 'rgba8unorm',
            dimension: '3d',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });
        
        device.queue.writeTexture(
            { texture: this.voxelGrid },
            voxelData,
            { bytesPerRow: size * 4, rowsPerImage: size },
            [size, size, size]
        );
        
        this.voxelGridView = this.voxelGrid.createView({ dimension: '3d' });
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

// CHECKITOUT: this file loads all the shaders and preprocesses them with some common code

import { Camera } from '../stage/camera';

import commonRaw from './common.wgsl?raw';

import naiveVertRaw from './naive.vs.wgsl?raw';
import naiveFragRaw from './naive.fs.wgsl?raw';

import geometryFragRaw from './geometry.fs.wgsl?raw';

import forwardPlusFragRaw from './forward_plus.fs.wgsl?raw';

import clusteredDeferredFragRaw from './clustered_deferred.fs.wgsl?raw';
import clusteredDeferredFullscreenVertRaw from './clustered_deferred_fullscreen.vs.wgsl?raw';
import clusteredDeferredFullscreenFragRaw from './clustered_deferred_fullscreen.fs.wgsl?raw';

import clusteredDeferredComputeSrcRaw from './clusteredDeferred.cs.wgsl?raw';

import moveLightsComputeRaw from './move_lights.cs.wgsl?raw';
import clusteringComputeRaw from './clustering.cs.wgsl?raw';

import zPrepassFragRaw from './zPrepass.fs.wgsl?raw';

// CONSTANTS (for use in shaders)
// =================================

// CHECKITOUT: feel free to add more constants here and to refer to them in your shader code

// Note that these are declared in a somewhat roundabout way because otherwise minification will drop variables
// that are unused in host side code.

const numClustersXConfig = 16;
const numClustersYConfig = 16;
const numClusterZConfig = 16;
const numTotalClustersConfig = numClustersXConfig * numClustersYConfig * numClusterZConfig;

export const constants = {
    numClustersX: numClustersXConfig,
    numClustersY: numClustersYConfig,
    numClustersZ: numClusterZConfig,
    numTotalClustersConfig: numTotalClustersConfig,

    
    averageLightsPerCluster: 1024,
    maxLightsPerCluster: 1024,

    ambientR: 0.05,
    ambientG: 0.05,
    ambientB: 0.05,

    bindGroup_scene: 0,
    bindGroup_model: 1,
    bindGroup_material: 2,

    moveLightsWorkgroupSize: 128,

    lightRadius: 2
};

// =================================

function evalShaderRaw(raw: string) {
    return raw
    .replace(/\$\{bindGroup_scene\}/g, constants.bindGroup_scene.toString())
    .replace(/\$\{bindGroup_model\}/g, constants.bindGroup_model.toString())
    .replace(/\$\{bindGroup_material\}/g, constants.bindGroup_material.toString())
    .replace(/\$\{moveLightsWorkgroupSize\}/g, constants.moveLightsWorkgroupSize.toString())

    .replace(/\$\{averageLightsPerCluster\}/g, constants.averageLightsPerCluster.toString())
    .replace(/\$\{maxLightsPerCluster\}/g, constants.maxLightsPerCluster.toString())
    
    .replace(/\$\{ambientR\}/g, constants.ambientR.toString())
    .replace(/\$\{ambientG\}/g, constants.ambientG.toString())
    .replace(/\$\{ambientB\}/g, constants.ambientB.toString())

    .replace(/\$\{lightRadius\}/g, constants.lightRadius.toString());
}

const commonSrc: string = evalShaderRaw(commonRaw);

function processShaderRaw(raw: string) {
    return commonSrc + evalShaderRaw(raw);
}

export const naiveVertSrc: string = processShaderRaw(naiveVertRaw);
export const naiveFragSrc: string = processShaderRaw(naiveFragRaw);

export const geometryFragSrc: string = processShaderRaw(geometryFragRaw);

export const forwardPlusFragSrc: string = processShaderRaw(forwardPlusFragRaw);

export const clusteredDeferredFragSrc: string = processShaderRaw(clusteredDeferredFragRaw);
export const clusteredDeferredFullscreenVertSrc: string = processShaderRaw(clusteredDeferredFullscreenVertRaw);
export const clusteredDeferredFullscreenFragSrc: string = processShaderRaw(clusteredDeferredFullscreenFragRaw);

export const clusteredDeferredComputeSrc: string = processShaderRaw(clusteredDeferredComputeSrcRaw);

export const moveLightsComputeSrc: string = processShaderRaw(moveLightsComputeRaw);
export const clusteringComputeSrc: string = processShaderRaw(clusteringComputeRaw);

export const zPrepassFragSrc: string = processShaderRaw(zPrepassFragRaw);
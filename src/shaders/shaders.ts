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

// IBL shaders
import generateCubemapRaw from './generate_cubemap.cs.wgsl?raw';
import irradianceConvolutionRaw from './irradiance_convolution.cs.wgsl?raw';
import prefilterEnvmapRaw from './prefilter_envmap.cs.wgsl?raw';
import brdfLutRaw from './brdf_lut.cs.wgsl?raw';
import equirectangularToCubemapRaw from './equirectangular_to_cubemap.cs.wgsl?raw';

// DDGI shaders
import ddgiProbeTraceRaw from './ddgi_probe_trace.cs.wgsl?raw';
import ddgiIrradianceUpdateRaw from './ddgi_irradiance_update.cs.wgsl?raw';
import ddgiVisibilityUpdateRaw from './ddgi_visibility_update.cs.wgsl?raw';
import ddgiBorderUpdateRaw from './ddgi_border_update.cs.wgsl?raw';

// NRC shaders
import nrcCommonRaw from './nrc_common.wgsl?raw';
import nrcScatterTrainingRaw from './nrc_scatter_training.cs.wgsl?raw';
import nrcTrainRaw from './nrc_train.cs.wgsl?raw';
import nrcInferenceRaw from './nrc_inference.cs.wgsl?raw';

// Shadow shaders
import shadowVertRaw from './shadow.vs.wgsl?raw';
import shadowFragRaw from './shadow.fs.wgsl?raw';

// VSM shaders
import vsmClearRaw from './vsm_clear.cs.wgsl?raw';
import vsmMarkPagesRaw from './vsm_mark_pages.cs.wgsl?raw';
import vsmAllocatePagesRaw from './vsm_allocate_pages.cs.wgsl?raw';

// Skybox shaders
import skyboxVertRaw from './skybox.vs.wgsl?raw';
import skyboxFragRaw from './skybox.fs.wgsl?raw';

// Volumetric Lighting shaders
import volumetricLightingVertRaw from './volumetric_lighting.vs.wgsl?raw';
import volumetricLightingFragRaw from './volumetric_lighting.fs.wgsl?raw';
import volumetricCompositeFragRaw from './volumetric_composite.fs.wgsl?raw';

// CONSTANTS (for use in shaders)
// =================================

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

    lightRadius: 2,

    // DDGI
    ddgiProbeGridX: 8,
    ddgiProbeGridY: 8,
    ddgiProbeGridZ: 8,
    ddgiRaysPerProbe: 64,
    ddgiIrradianceTexels: 8,
    ddgiVisibilityTexels: 16,

    // VSM
    vsmPageSize: 128,
    vsmPhysAtlasSize: 4096,
    vsmPhysPagesPerAxis: 32,
    vsmNumClipmapLevels: 6,
    vsmPagesPerLevelAxis: 128,

    // NRC
    nrcMaxTrainingSamples: 4096,
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

    .replace(/\$\{lightRadius\}/g, constants.lightRadius.toString())

    .replace(/\$\{ddgiRaysPerProbe\}/g, constants.ddgiRaysPerProbe.toString())
    .replace(/\$\{ddgiIrradianceTexels\}/g, constants.ddgiIrradianceTexels.toString())
    .replace(/\$\{ddgiVisibilityTexels\}/g, constants.ddgiVisibilityTexels.toString())
    .replace(/\$\{nrcMaxTrainingSamples\}/g, constants.nrcMaxTrainingSamples.toString());
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

// IBL shaders (standalone, not prepended with common)
export const generateCubemapSrc = generateCubemapRaw;
export const irradianceConvolutionSrc = irradianceConvolutionRaw;
export const prefilterEnvmapSrc = prefilterEnvmapRaw;
export const brdfLutSrc = brdfLutRaw;
export const equirectangularToCubemapSrc = equirectangularToCubemapRaw;

// Skybox shaders (need common for CameraUniforms)
export const skyboxVertSrc: string = processShaderRaw(skyboxVertRaw);
export const skyboxFragSrc: string = processShaderRaw(skyboxFragRaw);

// Volumetric shaders (need common)
export const volumetricLightingVertSrc: string = processShaderRaw(volumetricLightingVertRaw);
export const volumetricLightingFragSrc: string = processShaderRaw(volumetricLightingFragRaw);
export const volumetricCompositeFragSrc: string = processShaderRaw(volumetricCompositeFragRaw);

// DDGI shaders (need common for structs/utilities)
export const ddgiProbeTraceSrc: string = processShaderRaw(ddgiProbeTraceRaw);
export const ddgiIrradianceUpdateSrc: string = processShaderRaw(ddgiIrradianceUpdateRaw);
export const ddgiVisibilityUpdateSrc: string = processShaderRaw(ddgiVisibilityUpdateRaw);
export const ddgiBorderUpdateSrc: string = ddgiBorderUpdateRaw; // standalone, no common

// Shadow shaders (standalone)
export const shadowVertSrc: string = shadowVertRaw;
export const shadowFragSrc: string = shadowFragRaw;

// VSM shaders (need common for VSMUniforms struct)
export const vsmClearSrc: string = processShaderRaw(vsmClearRaw);
export const vsmMarkPagesSrc: string = processShaderRaw(vsmMarkPagesRaw);
export const vsmAllocatePagesSrc: string = processShaderRaw(vsmAllocatePagesRaw);

// NRC shaders (need common + nrc_common for structs/utilities)
const nrcCommonSrc: string = evalShaderRaw(nrcCommonRaw);
function processNrcShaderRaw(raw: string) {
    return commonSrc + nrcCommonSrc + evalShaderRaw(raw);
}
export const nrcScatterTrainingSrc: string = processNrcShaderRaw(nrcScatterTrainingRaw);
export const nrcTrainSrc: string = processNrcShaderRaw(nrcTrainRaw);
export const nrcInferenceSrc: string = processNrcShaderRaw(nrcInferenceRaw);
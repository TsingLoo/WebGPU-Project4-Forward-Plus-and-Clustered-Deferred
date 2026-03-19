const isMobileDevice = /Mobi|Android|iPhone|iPad|iPod/i.test(navigator.userAgent);

import Stats from 'stats.js';
import { GUI } from 'dat.gui';

import { initWebGPU, Renderer } from './renderer';
import { NaiveRenderer } from './renderers/naive';
import { ForwardPlusRenderer } from './renderers/forward_plus';
import { ClusteredDeferredRenderer } from './renderers/clustered_deferred';

// @ts-ignore
import parseHdr from 'parse-hdr';
// @ts-ignore
import parseExr from 'parse-exr';

const canvas = document.getElementById('webgpu-canvas') as HTMLCanvasElement;
import { setupLoaders, Scene } from './stage/scene';
import { Lights } from './stage/lights';
import { Camera } from './stage/camera';
import { Stage } from './stage/stage';
import { Environment } from './stage/environment';

const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));
const monitorTime = 8.0;
const restTime = 1000;

await initWebGPU();
setupLoaders();

let scene = new Scene();
await scene.loadGltf('./scenes/sponza/Sponza.gltf');

const camera = new Camera();
const lights = new Lights(camera);
const environment = new Environment();

const stats = new Stats();

const originalStatsBegin = stats.begin.bind(stats);

stats.begin = () => {
    originalStatsBegin();

    const now = performance.now();

    if (avgStats.collecting) {
        const elapsedTime = (now - avgStats.startTime) / 1000; 
        const frameTime = now - avgStats.lastFrameTime;

        if (elapsedTime < monitorTime) {
            avgStats.frameCount++;
        } else {
            avgStats.collecting = false;
            if (avgStats.frameCount > 0) {
                const avg = avgStats.frameCount / monitorTime;
                avgStats.avgFPS_20s = avg.toFixed(2);
            }else{
                avgStats.avgFPS_20s = 'N/A';
            }
        }
    }
    
    avgStats.lastFrameTime = now;
};

stats.showPanel(0);
document.body.appendChild(stats.dom);

const resultsElement = document.createElement('div');
resultsElement.style.cssText = `
    position: absolute;
    bottom: 10px;
    left: 10px;
    padding: 8px;
    background-color: rgba(0, 0, 0, 0.75);
    color: #00FF00;
    font-family: monospace;
    font-size: 14px;
    z-index: 100;
    max-height: 40vh;
    width: calc(100vw - 40px);
    overflow-y: auto;
    white-space: pre-wrap;
    display: none;
    box-sizing: border-box;
`;
document.body.appendChild(resultsElement);

const avgStats = {
    startTime: performance.now(),
    lastFrameTime: performance.now(),
    frameCount: 0,
    collecting: false,
    avgFPS_20s: 'Idle',

    reset: () => {
        avgStats.startTime = performance.now();
        avgStats.lastFrameTime = avgStats.startTime;
        avgStats.frameCount = 0;
        avgStats.collecting = true;
        avgStats.avgFPS_20s = 'Calculating...';
    }
};

const gui = new GUI();
gui.add(avgStats, 'avgFPS_20s').name('Avg FPS (20s)').listen();

const desiredMobileOptions = [5, 10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1250, 1450, 1500];

const desiredPCOptions = [1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000, 3200, 3400, 3600, 3800, 4000, 4200, 4400, 4600, 4800, 5000, 6000, 7000, 8000];

let desiredOptions = isMobileDevice? desiredMobileOptions : desiredPCOptions;

const safeOptions = desiredOptions.filter(
    count => count <= Lights.maxNumLights
);


safeOptions.push(desiredOptions[3]);
safeOptions.sort((a, b) => a - b); 


const lightNumSlider = gui.add(lights, 'numLights').min(1).max(Lights.maxNumLights).step(1).onChange(() => {
    lights.updateLightSetUniformNumLights();
});


const lightCountController = gui.add(lights, 'numLights', safeOptions).onChange(() => {
    lights.updateLightSetUniformNumLights();
});

const benchmarkController = {
    runBenchmark: async () => {

        resultsElement.innerHTML = '--- Benchmark Begin ---<br>';
        resultsElement.style.display = 'block';

        console.log("--- Benchmark Begin ---");
        const allResults: string[] = [];
        
        for (const lightCount of safeOptions) {
            
            lights.numLights = lightCount;
            lights.updateLightSetUniformNumLights();
            lightCountController.updateDisplay();
            lightNumSlider.updateDisplay();

            avgStats.avgFPS_20s = `Idling (${lightCount} lights)...`;
            await sleep(restTime);
            
            avgStats.reset();
            
            while (avgStats.collecting) {
                await sleep(200);
            }
            
            const resultString = `${lightCount} lights: ${avgStats.avgFPS_20s} FPS`;
            allResults.push(resultString);
            console.log(resultString);
            
            resultsElement.innerHTML += resultString + '<br>';
            resultsElement.scrollTop = resultsElement.scrollHeight;

            await sleep(500);
        }
        
        avgStats.avgFPS_20s = "Finished!";
        console.log("--- Benchmark End ---");
        console.log(allResults.join('\n'));

        resultsElement.innerHTML += '--- Benchmark End ---<br>';
        resultsElement.scrollTop = resultsElement.scrollHeight;
    }
};

gui.add(benchmarkController, 'runBenchmark').name('Run Full Benchmark');

const stage = new Stage(scene, lights, camera, stats, environment);

var renderer: Renderer | undefined;

function setRenderer(mode: string) {
    renderer?.stop();

    switch (mode) {
        case renderModes.naive:
            renderer = new NaiveRenderer(stage);
            break;
        case renderModes.forwardPlus:
            renderer = new ForwardPlusRenderer(stage);
            break;
        case renderModes.clusteredDeferred:
            renderer = new ClusteredDeferredRenderer(stage);
            break;
    }
}

// Helper: parse HDR file and return RGBA Float32Array + dimensions
function parseHdrFile(buffer: ArrayBuffer): { rgbaData: Float32Array, width: number, height: number } {
    const parsedLayout = parseHdr(buffer);

    let width: number, height: number;
    let data: Float32Array;

    if (parsedLayout.shape && parsedLayout.data) {
        width = parsedLayout.shape[0];
        height = parsedLayout.shape[1];
        data = parsedLayout.data;
    } else {
        throw new Error("Invalid HDR file format");
    }

    // parse-hdr returns RGBA (4 channels) already
    const rgbaData = new Float32Array(width * height * 4);
    if (data.length === width * height * 3) {
        for (let i = 0; i < width * height; i++) {
            rgbaData[i * 4 + 0] = data[i * 3 + 0];
            rgbaData[i * 4 + 1] = data[i * 3 + 1];
            rgbaData[i * 4 + 2] = data[i * 3 + 2];
            rgbaData[i * 4 + 3] = 1.0;
        }
    } else if (data.length === width * height * 4) {
        rgbaData.set(data);
    } else {
        throw new Error(`HDRI data length ${data.length} does not match dimensions ${width}x${height}`);
    }

    return { rgbaData, width, height };
}

// Helper: parse EXR file and return RGBA Float32Array + dimensions
function parseExrFile(buffer: ArrayBuffer): { rgbaData: Float32Array, width: number, height: number } {
    const FloatType = 1015;
    const RGBAFormat = 1023;
    const parsed = parseExr(buffer, FloatType);

    const { data, width, height, format } = parsed;
    const numPixels = width * height;

    let rgbaData: Float32Array;

    if (format === RGBAFormat) {
        // Data is RGBA, 4 floats per pixel
        if (data.length === numPixels * 4) {
            rgbaData = data as Float32Array;
        } else {
            // Data might be RGB (3 floats per pixel) even with RGBA format flag
            const channels = data.length / numPixels;
            rgbaData = new Float32Array(numPixels * 4);
            if (channels === 3) {
                for (let i = 0; i < numPixels; i++) {
                    rgbaData[i * 4 + 0] = data[i * 3 + 0];
                    rgbaData[i * 4 + 1] = data[i * 3 + 1];
                    rgbaData[i * 4 + 2] = data[i * 3 + 2];
                    rgbaData[i * 4 + 3] = 1.0;
                }
            } else {
                throw new Error(`Unexpected EXR channel count: ${channels}`);
            }
        }
    } else {
        // Single channel or other format
        throw new Error(`Unsupported EXR format code: ${format}. Expected RGBA (1023).`);
    }

    // parse-exr outputs rows bottom-to-top (OpenGL convention)
    // Flip to top-to-bottom to match .hdr convention and our shader's V-flip
    const floatsPerRow = width * 4;
    const tempRow = new Float32Array(floatsPerRow);
    for (let y = 0; y < Math.floor(height / 2); y++) {
        const topOffset = y * floatsPerRow;
        const bottomOffset = (height - 1 - y) * floatsPerRow;
        // Swap rows
        tempRow.set(rgbaData.subarray(topOffset, topOffset + floatsPerRow));
        rgbaData.set(rgbaData.subarray(bottomOffset, bottomOffset + floatsPerRow), topOffset);
        rgbaData.set(tempRow, bottomOffset);
    }

    return { rgbaData, width, height };
}

const fileInput = document.createElement('input');
fileInput.type = 'file';
fileInput.accept = '.hdr,.exr';
fileInput.style.display = 'none';
document.body.appendChild(fileInput);

fileInput.addEventListener('change', async (event) => {
    const file = (event.target as HTMLInputElement).files?.[0];
    if (!file) return;

    try {
        const fileName = file.name.toLowerCase();
        const buffer = await file.arrayBuffer();

        let rgbaData: Float32Array;
        let width: number;
        let height: number;

        if (fileName.endsWith('.hdr')) {
            ({ rgbaData, width, height } = parseHdrFile(buffer));
        } else if (fileName.endsWith('.exr')) {
            ({ rgbaData, width, height } = parseExrFile(buffer));
        } else {
            alert('Unsupported file format. Please upload .hdr or .exr');
            return;
        }

        console.log(`[HDRI] Loaded ${file.name}: ${width}x${height}, data length: ${rgbaData.length}`);

        await stage.environment.loadHDRI(rgbaData, width, height);
    } catch (e) {
        console.error("Failed to load HDRI:", e);
        alert("Failed to load HDRI: " + String(e));
    }
});

const uploadController = {
    uploadHDRI: () => {
        fileInput.click();
    },
    resetHDRI: () => {
        stage.environment.clearHDRI();
    }
};

gui.add(uploadController, 'uploadHDRI').name('Upload HDRI (.hdr/.exr)');
gui.add(uploadController, 'resetHDRI').name('Clear HDRI');

// --- Model upload (.gltf / .glb) ---
const modelFileInput = document.createElement('input');
modelFileInput.type = 'file';
modelFileInput.accept = '.gltf,.glb';
modelFileInput.style.display = 'none';
document.body.appendChild(modelFileInput);

modelFileInput.addEventListener('change', async (event) => {
    const file = (event.target as HTMLInputElement).files?.[0];
    if (!file) return;

    try {
        const fileName = file.name.toLowerCase();
        if (!fileName.endsWith('.gltf') && !fileName.endsWith('.glb')) {
            alert('Unsupported format. Please upload .gltf or .glb');
            return;
        }

        console.log(`[Model] Loading ${file.name}...`);

        // Read file as ArrayBuffer and parse directly
        const buffer = await file.arrayBuffer();

        // Replace the current scene
        const newScene = new Scene();
        await newScene.loadGltfBuffer(buffer);

        // Update stage with new scene
        stage.scene = newScene;

        // Disable random point lights (designed for Sponza) to avoid color artifacts
        lights.numLights = 0;
        lights.updateLightSetUniformNumLights();
        lightNumSlider.updateDisplay();
        lightCountController.updateDisplay();

        // Re-create renderer to pick up the new scene
        if (renderModeController) {
            setRenderer(renderModeController.getValue());
        }

        console.log(`[Model] Successfully loaded ${file.name}`);
    } catch (e) {
        console.error("Failed to load model:", e);
        alert("Failed to load model: " + String(e));
    }

    // Reset the input so the same file can be re-selected
    modelFileInput.value = '';
});

const modelUploadController = {
    loadModel: () => {
        modelFileInput.click();
    }
};

gui.add(modelUploadController, 'loadModel').name('Load Model (.gltf/.glb)');

const renderModes = { naive: 'naive', forwardPlus: 'forward+', clusteredDeferred: 'clustered deferred' };
let renderModeController = gui.add({ mode: renderModes.forwardPlus }, 'mode', renderModes);
renderModeController.onChange(setRenderer);

// DDGI controls
const ddgiFolder = gui.addFolder('DDGI');
ddgiFolder.add(stage.ddgi, 'enabled').name('DDGI Enabled').onChange(() => {
    stage.ddgi.updateUniforms();
});
ddgiFolder.add(stage.ddgi, 'irradianceHysteresis', 0.8, 0.999).step(0.001).name('Irr Hysteresis').onChange(() => {
    stage.ddgi.updateUniforms();
});
ddgiFolder.add(stage.ddgi, 'normalBias', 0.0, 1.0).step(0.01).name('Normal Bias').onChange(() => {
    stage.ddgi.updateUniforms();
});
ddgiFolder.add(stage.ddgi, 'debugMode', { 'Off': 0, 'Raw Atlas': 1, 'Decoded Irr': 2, 'IBL Only': 3, 'Mapped Normal': 4, 'Vertex Normal': 5, 'Tangent': 6, 'NdotL': 7 }).name('Debug View').onChange(() => {
    stage.ddgi.updateUniforms();
});
ddgiFolder.open();

// Sun Light controls
const sunFolder = gui.addFolder('Sun Light');
sunFolder.add(stage, 'sunEnabled').name('Enabled').onChange(() => {
    stage.updateSunLight();
});
sunFolder.add(stage, 'sunIntensity', 0.0, 20.0).step(0.1).name('Intensity').onChange(() => {
    stage.updateSunLight();
});
const sunDirProxy = { x: stage.sunDirection[0], y: stage.sunDirection[1], z: stage.sunDirection[2] };
sunFolder.add(sunDirProxy, 'x', -1, 1).step(0.01).name('Dir X').onChange(() => {
    stage.sunDirection = [sunDirProxy.x, sunDirProxy.y, sunDirProxy.z];
    stage.updateSunLight();
});
sunFolder.add(sunDirProxy, 'y', -1, 1).step(0.01).name('Dir Y').onChange(() => {
    stage.sunDirection = [sunDirProxy.x, sunDirProxy.y, sunDirProxy.z];
    stage.updateSunLight();
});
sunFolder.add(sunDirProxy, 'z', -1, 1).step(0.01).name('Dir Z').onChange(() => {
    stage.sunDirection = [sunDirProxy.x, sunDirProxy.y, sunDirProxy.z];
    stage.updateSunLight();
});
sunFolder.open();

// VSM Shadow controls
const vsmFolder = gui.addFolder('Shadow (VSM)');
const vsmProxy = {
    physAtlasSize: stage.vsm.physAtlasSize,
    pageSize: stage.vsm.pageSize,
    numClipmapLevels: stage.vsm.numClipmapLevels,
    pagesPerLevelAxis: stage.vsm.pagesPerLevelAxis,
    // Display-only derived values
    get virtualSize() { return stage.vsm.virtualSize; },
    get maxPhysPages() { return stage.vsm.maxPhysPages; },
};

vsmFolder.add(vsmProxy, 'physAtlasSize', [1024, 2048, 4096, 8192]).name('Atlas Size').onChange((v: number) => {
    stage.vsm.physAtlasSize = v;
    stage.vsm.recreate();
    stage.updateSunLight();
});
vsmFolder.add(vsmProxy, 'pageSize', [64, 128, 256]).name('Page Size').onChange((v: number) => {
    stage.vsm.pageSize = v;
    stage.vsm.recreate();
    stage.updateSunLight();
});
vsmFolder.add(vsmProxy, 'numClipmapLevels', 1, 8).step(1).name('Clipmap Levels').onChange((v: number) => {
    stage.vsm.numClipmapLevels = v;
    stage.vsm.recreate();
    stage.updateSunLight();
});
vsmFolder.add(vsmProxy, 'pagesPerLevelAxis', [32, 64, 128, 256]).name('Pages/Level Axis').onChange((v: number) => {
    stage.vsm.pagesPerLevelAxis = v;
    stage.vsm.recreate();
    stage.updateSunLight();
});
vsmFolder.add(vsmProxy, 'virtualSize').name('Virtual Size (px)').listen();
vsmFolder.add(vsmProxy, 'maxPhysPages').name('Max Phys Pages').listen();
vsmFolder.open();

setRenderer(renderModeController.getValue());


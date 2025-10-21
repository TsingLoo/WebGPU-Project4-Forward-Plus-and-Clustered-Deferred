import Stats from 'stats.js';
import { GUI } from 'dat.gui';

import { initWebGPU, Renderer } from './renderer';
import { NaiveRenderer } from './renderers/naive';
import { ForwardPlusRenderer } from './renderers/forward_plus';
import { ClusteredDeferredRenderer } from './renderers/clustered_deferred';

import { setupLoaders, Scene } from './stage/scene';
import { Lights } from './stage/lights';
import { Camera } from './stage/camera';
import { Stage } from './stage/stage';

const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

await initWebGPU();
setupLoaders();

let scene = new Scene();
await scene.loadGltf('./scenes/sponza/Sponza.gltf');

const camera = new Camera();
const lights = new Lights(camera);

const stats = new Stats();

const originalStatsBegin = stats.begin.bind(stats);

stats.begin = () => {
    originalStatsBegin();

    const now = performance.now();

    if (avgStats.collecting) {
        const elapsedTime = (now - avgStats.startTime) / 1000; 
        const frameTime = now - avgStats.lastFrameTime;

        if (elapsedTime < 20.0) {
            if (frameTime > 0) {
                const currentFPS = 1000.0 / frameTime;
                avgStats.frames.push(currentFPS);
            }
        } else {
            avgStats.collecting = false;
            
            if (avgStats.frames.length > 0) {
                const sum = avgStats.frames.reduce((a, b) => a + b, 0);
                const avg = sum / avgStats.frames.length;
                avgStats.avgFPS_20s = avg.toFixed(2);
            } else {
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
    frames: [] as number[],
    collecting: false,
    avgFPS_20s: 'Idle',

    reset: () => {
        avgStats.startTime = performance.now();
        avgStats.lastFrameTime = avgStats.startTime;
        avgStats.frames = [];
        avgStats.collecting = true;
        avgStats.avgFPS_20s = 'Calculating...';
    }
};

const gui = new GUI();
gui.add(avgStats, 'avgFPS_20s').name('Avg FPS (20s)').listen();

const desiredOptions = [5, 10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1250, 1500, 2000, 2500, 3000, 3800, 5000];

const safeOptions = desiredOptions.filter(
    count => count <= Lights.maxNumLights
);

if (!safeOptions.includes(Lights.maxNumLights)) {
    safeOptions.push(Lights.maxNumLights);
    safeOptions.sort((a, b) => a - b); 
}

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
            await sleep(3000);
            
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

const stage = new Stage(scene, lights, camera, stats);

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

const renderModes = { naive: 'naive', forwardPlus: 'forward+', clusteredDeferred: 'clustered deferred' };
let renderModeController = gui.add({ mode: renderModes.forwardPlus }, 'mode', renderModes);
renderModeController.onChange(setRenderer);

setRenderer(renderModeController.getValue());

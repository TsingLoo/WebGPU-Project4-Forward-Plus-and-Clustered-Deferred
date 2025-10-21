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

await initWebGPU();
setupLoaders();

let scene = new Scene();
await scene.loadGltf('./scenes/sponza/Sponza.gltf');

const camera = new Camera();
const lights = new Lights(camera);

const stats = new Stats();
stats.showPanel(0);
document.body.appendChild(stats.dom);

const gui = new GUI();
gui.add(lights, 'numLights').min(1).max(Lights.maxNumLights).step(1).onChange(() => {
    lights.updateLightSetUniformNumLights();
});

const desiredOptions = [5, 10, 50, 100, 200, 500, 600, 700, 800, 1000, 1500, 2000, 2500, 3000, 3800, 5000, 6000];

const safeOptions = desiredOptions.filter(
    count => count <= Lights.maxNumLights
);

if (!safeOptions.includes(Lights.maxNumLights)) {
    safeOptions.push(Lights.maxNumLights);
    safeOptions.sort((a, b) => a - b); 
}

gui.add(lights, 'numLights', safeOptions).onChange(() => {
    lights.updateLightSetUniformNumLights();
});

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

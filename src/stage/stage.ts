import { device } from "../renderer";
import { Camera } from "./camera";
import { DDGI } from "./ddgi";
import { Environment } from "./environment";
import { Lights } from "./lights";
import { Scene } from "./scene";
import { VSM } from "./vsm";

export class Stage {
    scene: Scene;
    lights: Lights;
    camera: Camera;
    stats: Stats;
    environment: Environment;
    ddgi: DDGI;
    vsm: VSM;

    // Sun light
    sunLightBuffer: GPUBuffer;
    sunDirection: [number, number, number] = [0.5, 0.8, 0.3]; // direction TO light
    sunColor: [number, number, number] = [1.0, 0.95, 0.85];   // warm white
    sunIntensity: number = 3.0;
    sunEnabled: boolean = true;

    constructor(scene: Scene, lights: Lights, camera: Camera, stats: Stats, environment: Environment) {
        this.scene = scene;
        this.lights = lights;
        this.camera = camera;
        this.stats = stats;
        this.environment = environment;
        this.ddgi = new DDGI(camera, environment);
        this.vsm = new VSM(camera);

        // Sync sun direction into VSM
        this.vsm.sunDirection = this.sunDirection;

        // SunLight struct: direction(vec4f) + color(vec4f) + light_vp(mat4x4f) + shadow_params(vec4f) = 112 bytes
        this.sunLightBuffer = device.createBuffer({
            label: "Sun Light Uniform",
            size: 112,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.updateSunLight();
    }

    updateSunLight() {
        // Sync sun direction to VSM
        this.vsm.sunDirection = this.sunDirection;

        const d = this.sunDirection;
        const len = Math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);

        // Write to GPU buffer: direction(16) + color(16) + light_vp(64) + shadow_params(16) = 112 bytes
        const data = new Float32Array(28); // 112 / 4
        // direction.xyz, w=intensity
        data[0] = d[0] / len; data[1] = d[1] / len; data[2] = d[2] / len; data[3] = this.sunIntensity;
        // color.rgb, a=enabled
        data[4] = this.sunColor[0]; data[5] = this.sunColor[1]; data[6] = this.sunColor[2];
        data[7] = this.sunEnabled ? 1.0 : 0.0;
        // light_vp matrix (16 floats) — placeholder identity, VSM uses its own clipmap VPs
        data[8] = 1; data[13] = 1; data[18] = 1; data[23] = 1;
        // shadow_params: x = texel size, y = bias
        data[24] = 1.0 / this.vsm.physAtlasSize;
        data[25] = 0.05; // normal bias
        data[26] = 0;
        data[27] = 0;

        device.queue.writeBuffer(this.sunLightBuffer, 0, data.buffer);
    }

    /**
     * Runs the full VSM shadow pipeline: clear → mark → allocate → render.
     * Call after the Z-prepass so the depth buffer is available.
     */
    renderShadowMap(encoder: GPUCommandEncoder, depthTextureView: GPUTextureView) {
        if (!this.sunEnabled) return;

        this.vsm.update(
            encoder,
            depthTextureView,
            this.scene,
            Array.from(this.camera.cameraPos).slice(0, 3) as [number, number, number],
        );
    }
}

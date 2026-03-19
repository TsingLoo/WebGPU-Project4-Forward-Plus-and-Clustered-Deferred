import { device } from "../renderer";
import { Camera } from "./camera";
import { DDGI } from "./ddgi";
import { Environment } from "./environment";
import { Lights } from "./lights";
import { Scene } from "./scene";

export class Stage {
    scene: Scene;
    lights: Lights;
    camera: Camera;
    stats: Stats;
    environment: Environment;
    ddgi: DDGI;

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

        this.sunLightBuffer = device.createBuffer({
            label: "Sun Light Uniform",
            size: 32, // 2 x vec4f
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.updateSunLight();
    }

    updateSunLight() {
        // Normalize direction
        const d = this.sunDirection;
        const len = Math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
        const data = new Float32Array([
            d[0] / len, d[1] / len, d[2] / len, this.sunIntensity,  // direction.xyz, w=intensity
            this.sunColor[0], this.sunColor[1], this.sunColor[2], this.sunEnabled ? 1.0 : 0.0, // color.rgb, a=enabled
        ]);
        device.queue.writeBuffer(this.sunLightBuffer, 0, data);
    }
}

import { Camera } from "./camera";
import { Environment } from "./environment";
import { Lights } from "./lights";
import { Scene } from "./scene";

export class Stage {
    scene: Scene;
    lights: Lights;
    camera: Camera;
    stats: Stats;
    environment: Environment;

    constructor(scene: Scene, lights: Lights, camera: Camera, stats: Stats, environment: Environment) {
        this.scene = scene;
        this.lights = lights;
        this.camera = camera;
        this.stats = stats;
        this.environment = environment;
    }
}

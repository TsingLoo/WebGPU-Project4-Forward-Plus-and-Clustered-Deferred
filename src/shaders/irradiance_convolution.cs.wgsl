// Diffuse irradiance convolution compute shader
// Input: environment cubemap (sampled), Output: irradiance cubemap (texture array)

@group(0) @binding(0) var envMap: texture_cube<f32>;
@group(0) @binding(1) var envSampler: sampler;
@group(0) @binding(2) var outputTex: texture_storage_2d_array<rgba16float, write>;

const PI: f32 = 3.14159265359;

fn getCubeDir(face: u32, uv: vec2f) -> vec3f {
    let u = uv.x * 2.0 - 1.0;
    let v = uv.y * 2.0 - 1.0;

    switch (face) {
        case 0u: { return normalize(vec3f( 1.0, -v,  -u)); }
        case 1u: { return normalize(vec3f(-1.0, -v,   u)); }
        case 2u: { return normalize(vec3f(  u,  1.0,  v)); }
        case 3u: { return normalize(vec3f(  u, -1.0, -v)); }
        case 4u: { return normalize(vec3f(  u,  -v, 1.0)); }
        case 5u: { return normalize(vec3f( -u,  -v,-1.0)); }
        default: { return vec3f(0.0); }
    }
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let face = gid.z;
    let dims = textureDimensions(outputTex);
    let size = dims.x;
    if (gid.x >= size || gid.y >= size || face >= 6u) { return; }

    let uv = (vec2f(f32(gid.x), f32(gid.y)) + 0.5) / f32(size);
    let N = getCubeDir(face, uv);

    // Build TBN from N
    var up = vec3f(0.0, 1.0, 0.0);
    if (abs(N.y) > 0.99) { up = vec3f(1.0, 0.0, 0.0); }
    let tangent = normalize(cross(up, N));
    let bitangent = cross(N, tangent);

    // Hemisphere sampling for diffuse irradiance
    var irradiance = vec3f(0.0);
    let sampleDelta: f32 = 0.05;
    var numSamples: f32 = 0.0;

    var phi: f32 = 0.0;
    loop {
        if (phi >= 2.0 * PI) { break; }
        var theta: f32 = 0.0;
        loop {
            if (theta >= 0.5 * PI) { break; }

            // Spherical to cartesian (tangent space)
            let tanVec = vec3f(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));

            // Tangent space to world
            let sampleDir = tanVec.x * tangent + tanVec.y * bitangent + tanVec.z * N;

            let sampleColor = textureSampleLevel(envMap, envSampler, sampleDir, 0.0).rgb;
            // Clamp extreme HDR values to prevent artifacts from sun/bright spots
            let maxLuminance = 100.0;
            let luminance = dot(sampleColor, vec3f(0.2126, 0.7152, 0.0722));
            var clampedColor = sampleColor;
            if (luminance > maxLuminance) {
                clampedColor = sampleColor * (maxLuminance / luminance);
            }
            irradiance += clampedColor * cos(theta) * sin(theta);
            numSamples += 1.0;

            theta += sampleDelta;
        }
        phi += sampleDelta;
    }

    irradiance = PI * irradiance / numSamples;
    textureStore(outputTex, vec2i(gid.xy), i32(face), vec4f(irradiance, 1.0));
}

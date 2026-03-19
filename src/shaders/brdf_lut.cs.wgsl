// BRDF Integration LUT compute shader
// Generates a 2D LUT where x = NdotV, y = roughness
// Output: RGBA16Float texture with (scale, bias, 0, 1) for the split-sum approximation

@group(0) @binding(0) var outputTex: texture_storage_2d<rgba16float, write>;

const PI: f32 = 3.14159265359;

fn radicalInverseVdC(bits_in: u32) -> f32 {
    var bits = bits_in;
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return f32(bits) * 2.3283064365386963e-10;
}

fn hammersley(i: u32, N: u32) -> vec2f {
    return vec2f(f32(i) / f32(N), radicalInverseVdC(i));
}

fn importanceSampleGGX(Xi: vec2f, N: vec3f, roughness: f32) -> vec3f {
    let a = roughness * roughness;

    let phi = 2.0 * PI * Xi.x;
    let cosTheta = sqrt((1.0 - Xi.y) / (1.0 + (a * a - 1.0) * Xi.y));
    let sinTheta = sqrt(1.0 - cosTheta * cosTheta);

    let H = vec3f(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);

    var up = vec3f(0.0, 1.0, 0.0);
    if (abs(N.y) > 0.99) { up = vec3f(1.0, 0.0, 0.0); }
    let tangent = normalize(cross(up, N));
    let bitangent = cross(N, tangent);

    return normalize(tangent * H.x + bitangent * H.y + N * H.z);
}

fn geometrySchlickGGX(NdotV: f32, roughness: f32) -> f32 {
    let a = roughness;
    let k = (a * a) / 2.0; // IBL uses k = a^2 / 2
    return NdotV / (NdotV * (1.0 - k) + k);
}

fn geometrySmith(N: vec3f, V: vec3f, L: vec3f, roughness: f32) -> f32 {
    let NdotV = max(dot(N, V), 0.0);
    let NdotL = max(dot(N, L), 0.0);
    return geometrySchlickGGX(NdotV, roughness) * geometrySchlickGGX(NdotL, roughness);
}

fn integrateBRDF(NdotV_in: f32, roughness: f32) -> vec2f {
    let NdotV = max(NdotV_in, 0.001);
    let V = vec3f(sqrt(1.0 - NdotV * NdotV), 0.0, NdotV);

    var A: f32 = 0.0;
    var B: f32 = 0.0;

    let N = vec3f(0.0, 0.0, 1.0);
    let SAMPLE_COUNT = 1024u;

    for (var i = 0u; i < SAMPLE_COUNT; i += 1u) {
        let Xi = hammersley(i, SAMPLE_COUNT);
        let H = importanceSampleGGX(Xi, N, roughness);
        let L = normalize(2.0 * dot(V, H) * H - V);

        let NdotL = max(L.z, 0.0);
        let NdotH = max(H.z, 0.0);
        let VdotH = max(dot(V, H), 0.0);

        if (NdotL > 0.0) {
            let G = geometrySmith(N, V, L, roughness);
            let G_Vis = (G * VdotH) / (NdotH * NdotV);
            let Fc = pow(1.0 - VdotH, 5.0);

            A += (1.0 - Fc) * G_Vis;
            B += Fc * G_Vis;
        }
    }

    A /= f32(SAMPLE_COUNT);
    B /= f32(SAMPLE_COUNT);
    return vec2f(A, B);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let dims = textureDimensions(outputTex);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }

    let NdotV = (f32(gid.x) + 0.5) / f32(dims.x);
    let roughness = (f32(gid.y) + 0.5) / f32(dims.y);

    let result = integrateBRDF(NdotV, roughness);
    textureStore(outputTex, vec2i(gid.xy), vec4f(result.x, result.y, 0.0, 1.0));
}

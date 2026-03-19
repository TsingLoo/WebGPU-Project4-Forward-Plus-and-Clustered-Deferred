// CHECKITOUT: code that you add here will be prepended to all shaders

const PI = 3.14159265359;

struct Light {
    pos: vec3f,
    color: vec3f
}

struct LightSet {
    numLights: u32,
    lights: array<Light>
}

struct TileMeta {
    offset: u32,
    count: u32,
};

struct LightIndexList {
    counter: atomic<u32>,
    indices: array<u32>,
};

struct LightIndexListReadOnly {
    counter: u32,
    indices: array<u32>,
};

struct CameraUniforms {
    view_proj_mat: mat4x4f,
    inv_proj_mat: mat4x4f,
    proj_mat: mat4x4f,
    view_mat: mat4x4f,
    near_plane: f32,
    far_plane: f32,
    _pad0: f32,
    _pad1: f32,
    camera_pos: vec4f,
}

struct ClusterSet {
    screen_width: u32,
    screen_height: u32,
    num_clusters_X: u32,
    num_clusters_Y: u32,
    num_clusters_Z: u32
}

// ============================
// Attenuation
// ============================
fn rangeAttenuation(distance: f32) -> f32 {
    return clamp(1.f - pow(distance / ${lightRadius}, 4.f), 0.f, 1.f) / (distance * distance);
}

// ============================
// Cook-Torrance PBR BRDF
// ============================

// Trowbridge-Reitz GGX Normal Distribution Function
fn distributionGGX(N: vec3f, H: vec3f, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let NdotH = max(dot(N, H), 0.0);
    let NdotH2 = NdotH * NdotH;

    let denom = NdotH2 * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

// Schlick-GGX Geometry function (single direction)
fn geometrySchlickGGX(NdotV: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

// Smith's method combining geometry for both view and light directions
fn geometrySmith(N: vec3f, V: vec3f, L: vec3f, roughness: f32) -> f32 {
    let NdotV = max(dot(N, V), 0.0);
    let NdotL = max(dot(N, L), 0.0);
    return geometrySchlickGGX(NdotV, roughness) * geometrySchlickGGX(NdotL, roughness);
}

// Fresnel-Schlick approximation
fn fresnelSchlick(cosTheta: f32, F0: vec3f) -> vec3f {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// Fresnel-Schlick with roughness (for IBL ambient specular)
fn fresnelSchlickRoughness(cosTheta: f32, F0: vec3f, roughness: f32) -> vec3f {
    return F0 + (max(vec3f(1.0 - roughness), F0) - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// ============================
// PBR Point Light Contribution (Cook-Torrance)
// ============================
fn calculateLightContribPBR(light: Light, posWorld: vec3f, N: vec3f, V: vec3f, albedo: vec3f, metallic: f32, roughness: f32) -> vec3f {
    let vecToLight = light.pos - posWorld;
    let distToLight = length(vecToLight);
    let L = normalize(vecToLight);
    let H = normalize(V + L);

    let attenuation = rangeAttenuation(distToLight);
    let radiance = light.color * attenuation;

    // F0 for dielectrics is 0.04, for metals it's the albedo
    let F0 = mix(vec3f(0.04), albedo, metallic);

    // Cook-Torrance specular BRDF
    let NDF = distributionGGX(N, H, roughness);
    let G = geometrySmith(N, V, L, roughness);
    let F = fresnelSchlick(max(dot(H, V), 0.0), F0);

    let numerator = NDF * G * F;
    let denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
    let specular = numerator / denominator;

    // Energy conservation: kS is what's reflected, kD is what's refracted (diffuse)
    let kS = F;
    // Metals have no diffuse
    let kD = (vec3f(1.0) - kS) * (1.0 - metallic);

    let NdotL = max(dot(N, L), 0.0);
    return (kD * albedo / PI + specular) * radiance * NdotL;
}

// Simple Lambert fallback (for backward compatibility)
fn calculateLightContrib(light: Light, posWorld: vec3f, nor: vec3f) -> vec3f {
    let vecToLight = light.pos - posWorld;
    let distToLight = length(vecToLight);
    let lambert = max(dot(nor, normalize(vecToLight)), 0.f);
    return light.color * lambert * rangeAttenuation(distToLight);
}

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

// ============================
// Directional Sun Light
// ============================
struct SunLight {
    direction: vec4f,   // xyz = direction TO light (normalized), w = intensity
    color: vec4f,       // rgb = color, a = enabled (0 or 1)
}

fn calculateSunLightPBR(
    sun: SunLight,
    posWorld: vec3f,
    N: vec3f,
    V: vec3f,
    albedo: vec3f,
    metallic: f32,
    roughness: f32
) -> vec3f {
    if (sun.color.a < 0.5) { return vec3f(0.0); }

    let L = normalize(sun.direction.xyz);
    let H = normalize(V + L);
    let intensity = sun.direction.w;
    let radiance = sun.color.rgb * intensity;

    let F0 = mix(vec3f(0.04), albedo, metallic);
    let NdotV = max(dot(N, V), 0.0);

    let NDF = distributionGGX(N, H, roughness);
    let G = geometrySmith(N, V, L, roughness);
    let F = fresnelSchlick(max(dot(H, V), 0.0), F0);

    let numerator = NDF * G * F;
    let denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
    let specular = numerator / denominator;

    let kS = F;
    let kD = (vec3f(1.0) - kS) * (1.0 - metallic);

    let NdotL = max(dot(N, L), 0.0);
    return (kD * albedo / PI + specular) * radiance * NdotL;
}

// ============================
// DDGI
// ============================

struct DDGIUniforms {
    grid_count: vec4i,       // x, y, z, total
    grid_min: vec4f,         // world-space min corner
    grid_max: vec4f,         // world-space max corner
    grid_spacing: vec4f,     // spacing per axis, w = rays per probe
    irradiance_texel_size: vec4f, // texel_dim, texel_dim_with_border, atlas_width, atlas_height
    visibility_texel_size: vec4f, // texel_dim, texel_dim_with_border, atlas_width, atlas_height
    hysteresis: vec4f,       // irradiance_hysteresis, visibility_hysteresis, normal_bias, view_bias
    ddgi_enabled: vec4f,     // x = enabled (0 or 1), y = debug_mode (0=off,1=irr,2=vis)
}

// Octahedral encoding: map direction to [0,1]^2
fn octEncode(n: vec3f) -> vec2f {
    let sum = abs(n.x) + abs(n.y) + abs(n.z);
    var oct = vec2f(n.x, n.y) / sum;
    if (n.z < 0.0) {
        let signs = vec2f(
            select(-1.0, 1.0, oct.x >= 0.0),
            select(-1.0, 1.0, oct.y >= 0.0)
        );
        oct = (1.0 - abs(vec2f(oct.y, oct.x))) * signs;
    }
    return oct * 0.5 + 0.5;
}

// Octahedral decoding: map [0,1]^2 to direction
fn octDecode(uv: vec2f) -> vec3f {
    var f = uv * 2.0 - 1.0;
    var n = vec3f(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
    if (n.z < 0.0) {
        let signs = vec2f(
            select(-1.0, 1.0, n.x >= 0.0),
            select(-1.0, 1.0, n.y >= 0.0)
        );
        let xy = (1.0 - abs(vec2f(n.y, n.x))) * signs;
        n = vec3f(xy.x, xy.y, n.z);
    }
    return normalize(n);
}

// Get world-space position of a probe given its 3D grid index
fn ddgiProbePosition(gridIdx: vec3i, ddgi: DDGIUniforms) -> vec3f {
    return ddgi.grid_min.xyz + vec3f(gridIdx) * ddgi.grid_spacing.xyz;
}

// Get the texel coordinate in the irradiance atlas for a probe index and octahedral UV
fn ddgiIrradianceTexelCoord(probeIdx: i32, octUV: vec2f, ddgi: DDGIUniforms) -> vec2f {
    let texelDim = i32(ddgi.irradiance_texel_size.x);      // 8
    let texelDimBorder = i32(ddgi.irradiance_texel_size.y); // 10 (8+2)
    let atlasWidth = ddgi.irradiance_texel_size.z;
    let atlasHeight = ddgi.irradiance_texel_size.w;

    let probesPerRow = i32(ddgi.grid_count.x);
    let probeRow = probeIdx / probesPerRow;
    let probeCol = probeIdx % probesPerRow;

    let cornerX = f32(probeCol * texelDimBorder + 1); // +1 for border
    let cornerY = f32(probeRow * texelDimBorder + 1);

    // UV within probe texel region
    let texelX = cornerX + octUV.x * f32(texelDim - 1);
    let texelY = cornerY + octUV.y * f32(texelDim - 1);

    return vec2f(texelX / atlasWidth, texelY / atlasHeight);
}

// Get the texel coordinate in the visibility atlas for a probe index and octahedral UV
fn ddgiVisibilityTexelCoord(probeIdx: i32, octUV: vec2f, ddgi: DDGIUniforms) -> vec2f {
    let texelDim = i32(ddgi.visibility_texel_size.x);       // 16
    let texelDimBorder = i32(ddgi.visibility_texel_size.y);  // 18 (16+2)
    let atlasWidth = ddgi.visibility_texel_size.z;
    let atlasHeight = ddgi.visibility_texel_size.w;

    let probesPerRow = i32(ddgi.grid_count.x);
    let probeRow = probeIdx / probesPerRow;
    let probeCol = probeIdx % probesPerRow;

    let cornerX = f32(probeCol * texelDimBorder + 1);
    let cornerY = f32(probeRow * texelDimBorder + 1);

    let texelX = cornerX + octUV.x * f32(texelDim - 1);
    let texelY = cornerY + octUV.y * f32(texelDim - 1);

    return vec2f(texelX / atlasWidth, texelY / atlasHeight);
}

// Flatten 3D grid index to linear probe index
fn ddgiProbeLinearIndex(gridIdx: vec3i, ddgi: DDGIUniforms) -> i32 {
    return gridIdx.z * ddgi.grid_count.x * ddgi.grid_count.y
         + gridIdx.y * ddgi.grid_count.x
         + gridIdx.x;
}


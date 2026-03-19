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
    direction: vec4f,       // xyz = direction TO light (normalized), w = intensity
    color: vec4f,           // rgb = color, a = enabled (0 or 1)
    light_vp: mat4x4f,      // light-space view-projection matrix (unused now, kept for layout compat)
    shadow_params: vec4f,    // x = 1/shadow_map_size, y = bias, z = 0, w = 0
}

fn calculateSunLightPBR(
    sun: SunLight,
    posWorld: vec3f,
    N: vec3f,
    V: vec3f,
    albedo: vec3f,
    metallic: f32,
    roughness: f32,
    shadow: f32
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
    return (kD * albedo / PI + specular) * radiance * NdotL * shadow;
}

// ============================
// Virtual Shadow Map (VSM)
// ============================
struct VSMUniforms {
    clipmap_vp: array<mat4x4f, 6>,   // VP matrix per clipmap level
    inv_view_proj: mat4x4f,          // camera inverse view-projection (for mark pass)
    clipmap_count: u32,
    pages_per_axis: u32,             // 128
    phys_atlas_size: u32,            // 4096
    phys_pages_per_axis: u32,        // 32
}

// Select best clipmap level based on world position → light NDC coverage
// Uses an inset margin to prevent level oscillation at boundaries
fn vsmSelectClipmapLevel(
    vsm: VSMUniforms,
    posWorld: vec3f,
) -> u32 {
    for (var level = 0u; level < vsm.clipmap_count; level++) {
        let lightClip = vsm.clipmap_vp[level] * vec4f(posWorld, 1.0);
        let lightNDC = lightClip.xyz / lightClip.w;

        // Inset margin prevents flickering at level boundaries
        let margin = 0.9;
        if (lightNDC.x >= -margin && lightNDC.x <= margin &&
            lightNDC.y >= -margin && lightNDC.y <= margin &&
            lightNDC.z >= 0.0    && lightNDC.z <= 1.0) {
            return level;
        }
    }
    return vsm.clipmap_count; // No valid level
}

// Compute atlas tile offset and size for a clipmap level (square grid layout)
fn vsmTileInfo(vsm: VSMUniforms, level: u32) -> vec3u {
    // Returns (xOffset, yOffset, tileSize)
    let gridCols = u32(ceil(sqrt(f32(vsm.clipmap_count))));
    let tileSize = vsm.phys_atlas_size / gridCols;
    let col = level % gridCols;
    let row = level / gridCols;
    return vec3u(col * tileSize, row * tileSize, tileSize);
}

// Calculate shadow using VSM (Virtual Shadow Map) with clipmap atlas
// Uses textureLoad with bilinear PCF for smooth shadow edges
fn calculateShadowVSM(
    physAtlas: texture_depth_2d,
    shadowSampler: sampler_comparison, // kept for API compat, unused
    vsm: VSMUniforms,
    sun: SunLight,
    posWorld: vec3f,
    N: vec3f,
) -> f32 {
    // Normal bias to avoid shadow acne
    let bias = sun.shadow_params.y;
    let biasedPos = posWorld + N * bias;

    // Select finest valid clipmap level
    let level = vsmSelectClipmapLevel(vsm, biasedPos);

    // Compute UV and depth for sampling (use level 0 as safe fallback)
    let safeLevel = min(level, vsm.clipmap_count - 1u);
    let lightClip = vsm.clipmap_vp[safeLevel] * vec4f(biasedPos, 1.0);
    let lightNDC = lightClip.xyz / lightClip.w;

    let uv = vec2f(lightNDC.x * 0.5 + 0.5, -lightNDC.y * 0.5 + 0.5);
    let depth = lightNDC.z;

    // Square grid layout: each level gets a square tile
    let tile = vsmTileInfo(vsm, safeLevel);
    let tileX = f32(tile.x);
    let tileY = f32(tile.y);
    let tileSz = f32(tile.z);

    // Sub-texel coordinates for bilinear weighting
    let texCoordX = tileX + uv.x * tileSz - 0.5;
    let texCoordY = tileY + uv.y * tileSz - 0.5;
    let baseX = i32(floor(texCoordX));
    let baseY = i32(floor(texCoordY));
    let fracX = texCoordX - floor(texCoordX);
    let fracY = texCoordY - floor(texCoordY);

    let safeDepth = clamp(depth, 0.0, 1.0);

    // Tile boundary clamps to prevent PCF bleeding into adjacent tiles
    let tileMinX = i32(tile.x);
    let tileMinY = i32(tile.y);
    let tileMaxX = i32(tile.x + tile.z) - 1;
    let tileMaxY = i32(tile.y + tile.z) - 1;

    // Bilinear-interpolated PCF: 5×5 kernel with Gaussian weighting
    var shadow = 0.0;
    var totalWeight = 0.0;

    for (var ky = -2; ky <= 2; ky++) {
        for (var kx = -2; kx <= 2; kx++) {
            let dist = f32(kx * kx + ky * ky);
            let w = exp(-dist * 0.3);

            let ox = baseX + kx;
            let oy = baseY + ky;

            // Clamp to tile boundaries (not whole atlas) to prevent cross-level bleeding
            let s00 = textureLoad(physAtlas, vec2i(clamp(ox,     tileMinX, tileMaxX), clamp(oy,     tileMinY, tileMaxY)), 0);
            let s10 = textureLoad(physAtlas, vec2i(clamp(ox + 1, tileMinX, tileMaxX), clamp(oy,     tileMinY, tileMaxY)), 0);
            let s01 = textureLoad(physAtlas, vec2i(clamp(ox,     tileMinX, tileMaxX), clamp(oy + 1, tileMinY, tileMaxY)), 0);
            let s11 = textureLoad(physAtlas, vec2i(clamp(ox + 1, tileMinX, tileMaxX), clamp(oy + 1, tileMinY, tileMaxY)), 0);

            let c00 = select(0.0, 1.0, safeDepth <= s00);
            let c10 = select(0.0, 1.0, safeDepth <= s10);
            let c01 = select(0.0, 1.0, safeDepth <= s01);
            let c11 = select(0.0, 1.0, safeDepth <= s11);

            let bilinear = mix(mix(c00, c10, fracX), mix(c01, c11, fracX), fracY);

            shadow += bilinear * w;
            totalWeight += w;
        }
    }

    shadow /= totalWeight;

    // If sun is disabled or position is outside all clipmap levels, return fully lit
    let valid = select(0.0, 1.0, sun.color.a >= 0.5 && level < vsm.clipmap_count);
    return mix(1.0, shadow, valid);
}

// Simple VSM shadow for compute shaders (DDGI probes) — no comparison sampler
fn calculateShadowVSMSimple(
    physAtlas: texture_depth_2d,
    vsm: VSMUniforms,
    sun: SunLight,
    posWorld: vec3f,
    N: vec3f,
) -> f32 {
    if (sun.color.a < 0.5) { return 1.0; }

    let bias = sun.shadow_params.y;
    let biasedPos = posWorld + N * bias;

    let level = vsmSelectClipmapLevel(vsm, biasedPos);
    if (level >= vsm.clipmap_count) { return 1.0; }

    let lightClip = vsm.clipmap_vp[level] * vec4f(biasedPos, 1.0);
    let lightNDC = lightClip.xyz / lightClip.w;
    let uv = vec2f(lightNDC.x * 0.5 + 0.5, -lightNDC.y * 0.5 + 0.5);
    let depth = lightNDC.z;

    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || depth > 1.0) {
        return 1.0;
    }

    // Square grid layout (same as calculateShadowVSM)
    let tile = vsmTileInfo(vsm, level);
    let ssi = vec2i(
        i32(f32(tile.x) + uv.x * f32(tile.z)),
        i32(f32(tile.y) + uv.y * f32(tile.z))
    );
    let shadowDepth = textureLoad(physAtlas, ssi, 0);
    return select(0.0, 1.0, depth <= shadowDepth + 0.005);
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

    // Inset UVs by half texel to avoid sampling from uninitialized border
    let inset = 0.5 / f32(texelDim);
    let safeUV = clamp(octUV, vec2f(inset), vec2f(1.0 - inset));

    // UV within probe texel region
    let texelX = cornerX + safeUV.x * f32(texelDim - 1);
    let texelY = cornerY + safeUV.y * f32(texelDim - 1);

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

    // Inset UVs by half texel to avoid sampling from uninitialized border
    let inset = 0.5 / f32(texelDim);
    let safeUV = clamp(octUV, vec2f(inset), vec2f(1.0 - inset));

    let texelX = cornerX + safeUV.x * f32(texelDim - 1);
    let texelY = cornerY + safeUV.y * f32(texelDim - 1);

    return vec2f(texelX / atlasWidth, texelY / atlasHeight);
}

// Flatten 3D grid index to linear probe index
fn ddgiProbeLinearIndex(gridIdx: vec3i, ddgi: DDGIUniforms) -> i32 {
    return gridIdx.z * ddgi.grid_count.x * ddgi.grid_count.y
         + gridIdx.y * ddgi.grid_count.x
         + gridIdx.x;
}


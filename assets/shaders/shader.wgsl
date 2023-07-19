struct CameraUniform {
    vp_matrix: mat4x4<f32>,
};
@group(0) @binding(0)
var<uniform> camera: CameraUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
    // @location(1) view_vector: vec3<f32>,
};

@vertex
fn vs_main(model: VertexInput,) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = camera.vp_matrix * vec4<f32>(model.position, 1.0);
    out.tex_coords = model.tex_coords;
    // out.view_vector =
    return out;
}

// Based on GPU Gems
// Optimised by Alan Zucconi
fn bump3y (x: vec3<f32>, yoffset: vec3<f32>) -> vec3<f32>
{
    var y = vec3<f32>(1., 1., 1.) - x * x;
    y = saturate(y-yoffset);
    return y;
}

fn spectral_zucconi (w: f32) -> vec3<f32>
{
    // w: [400, 700]
    // x: [0,   1]
    let x = saturate((w - 400.0)/ 300.0);

    let cs = vec3<f32>(3.54541723, 2.86670055, 2.29421995);
    let xs = vec3<f32>(0.69548916, 0.49416934, 0.28269708);
    let ys = vec3<f32>(0.02320775, 0.15936245, 0.53520021);

    return bump3y (    cs * (x - xs), ys);
}

fn spectral_zucconi6 (w: f32) -> vec3<f32>
{
    // w: [400, 700]
    // x: [0,   1]
    let x = saturate((w - 400.0)/ 300.0);

    let c1 = vec3<f32>(3.54585104, 2.93225262, 2.41593945);
    let x1 = vec3<f32>(0.69549072, 0.49228336, 0.27699880);
    let y1 = vec3<f32>(0.02312639, 0.15225084, 0.52607955);

    let c2 = vec3<f32>(3.90307140, 3.21182957, 3.96587128);
    let x2 = vec3<f32>(0.11748627, 0.86755042, 0.66077860);
    let y2 = vec3<f32>(0.84897130, 0.88445281, 0.73949448);

    return
        bump3y(c1 * (x - x1), y1) +
        bump3y(c2 * (x - x2), y2) ;
}

fn hueToRgb(p0: f32, q0: f32, t0: f32) -> f32 {
    let p = p0;
    let q = q0;
    var t = t0;

    if (t < 0f) {
        t += 1.;
    }
    if (t > 1.) {
        t -= 1.;
    }
    if (t < 1./6.) {
        return p + (q - p) * 6. * t;
    }
    if (t < 1./2.) {
        return q;
    }
    if (t < 2./3.) {
        return p + (q - p) * (2./3. - t) * 6.;
    }
    return p;
}

fn hslToRgb(h: f32, s: f32, l: f32) -> vec4<f32> {
    var r: f32;
    var g: f32;
    var b: f32;

    if(s == 0.)
    {
        r = l; g = l; b = l;
    }
    else
    {
        var q: f32;
        if(l < 0.5) {
            q = l * (1. + s);
        } else {
            q = l + s - l * s;
        }
        let p = 2. * l - q;
        r = hueToRgb(p, q, h + 1./3.);
        g = hueToRgb(p, q, h);
        b = hueToRgb(p, q, h - 1./3.);
    }

    return vec4(r, g, b, 1.);
}

fn colorize(x: f32, y: f32) -> vec4<f32> {
    var r = sqrt(x*x+y*y);
    var arg = atan2(y, x);
    var real = x;
    var imag = y;

    var log2r = log2(r);
    var h = arg / 6.283185307179586476925286766559;
    var l = (log2r - floor(log2r)) / 5. + 2. / 5.;

    return hslToRgb(h, 0.9, l);
}

fn f(x: f32, y: f32) -> vec2<f32> {
    return vec2(exp(x) * cos(y), exp(x) * sin(y));
}

fn diffraction(sin_thetaL: f32, sin_thetaV: f32, d: f32) -> vec3<f32> {
    var color = vec3<f32>(0., 0., 0.);
    for (var n = 1; n <= 8; n++)
    {
        let wavelength = abs(sin_thetaL - sin_thetaV) * d / f32(n);
        color += spectral_zucconi6(wavelength);
    }
    color = saturate(color);
    return color;
}

// Fragment shader

// @group(0) @binding(0)
// var t_diffuse: texture_2d<f32>;
// @group(0) @binding(1)
// var s_diffuse: sampler;
@group(1) @binding(0)
var<uniform> view_bounds: vec4<f32>;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let u = in.tex_coords[0];
    let v = in.tex_coords[1];

    let right  = view_bounds[0];
    let top    = view_bounds[1];
    let left   = view_bounds[2];
    let bottom = view_bounds[3];

    let width = right - left;
    let height = top - bottom;

    let x = left + width * u;
    let y = top - height * v;

    return colorize(x, y);
}

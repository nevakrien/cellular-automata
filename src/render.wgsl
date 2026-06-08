struct RpsParams {
    width: u32,
    height: u32,
};


struct DisplayInfo {
    offset:vec2<f32>,
    scale:f32,
    pad:u32,
};

struct DisplayColors {
    colors: array<vec4<f32>,8>,
};


@group(0) @binding(0)
var<uniform> params: RpsParams;

@group(0) @binding(1)
var<uniform> display_info: DisplayInfo;

@group(0) @binding(2)
var<uniform> display_colors: DisplayColors;

@group(0) @binding(3)
var<storage, read> input_grid: array<i32>;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

struct FragmentInput {
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 3.0,  1.0),
    );

    let position = positions[vertex_index];

    var out: VertexOutput;
    out.clip_position = vec4<f32>(position, 0.0, 1.0);

    // For the visible screen area, this becomes roughly 0..1.
    out.uv = position * 0.5 + vec2<f32>(0.5, 0.5);

    return out;
}

@fragment
fn fs_main(in: FragmentInput) -> @location(0) vec4<f32> {
    // Clamp because uv can hit exactly 1.0 at the edge,
    // and u32(1.0 * width) would become width, which is out of bounds.

    let uv = display_info.offset + in.uv/display_info.scale;

    if(uv.x<0 || uv.x>1 || uv.y<0 || uv.y>1){
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }

    let x = min(u32(uv.x * f32(params.width)), params.width - 1u);
    let y = min(u32(uv.y * f32(params.height)), params.height - 1u);

    let index = x + y * params.width;

    let cell = input_grid[index];

    if (cell < 0 || cell > 7) {
        return vec4<f32>(1.0, 0.0, 1.0, 1.0); // debug magenta for bad state
    }

    return display_colors.colors[u32(cell)];
}

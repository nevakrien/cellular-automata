use std::collections::HashSet;
use std::env;
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::time::{SystemTime, UNIX_EPOCH};

use egui_wgpu::wgpu;
use egui_wgpu::{Renderer, RendererOptions, ScreenDescriptor};
use egui_winit::winit::application::ApplicationHandler;
use egui_winit::winit::event::{
    ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent,
};
use egui_winit::winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use egui_winit::winit::keyboard::{Key, KeyCode, ModifiersState, NamedKey, PhysicalKey};
use egui_winit::winit::window::{Window, WindowId};

mod brush;
mod pipelines;

use brush::{BrushEdit, BrushStroke, MAX_BRUSH_EDITS};
use pipelines::{GameMode, ScreenSize, Shaders};

const GRID_SIZE: ScreenSize = ScreenSize {
    width: 1024,
    height: 1024,
};

const MIN_VIEW_SCALE: f32 = 1.0;
const MAX_VIEW_SCALE: f32 = 128.0;
const WHEEL_ZOOM_STEP: f32 = 1.15;
const PAN_SCREENS_PER_SECOND: f32 = 0.75;

const REASONABLE_MAX_TARGET_STEPS_PER_SECOND: f32 = 240.0;
const REASONABLE_MAX_SIMULATION_STEPS_PER_TICK: usize = 16;
const REASONABLE_MAX_STARTUP_CLUSTER_SIZE: u32 = 128;

#[derive(Default)]
struct NavigationInput {
    left: bool,
    right: bool,
    up: bool,
    down: bool,
}

impl NavigationInput {
    fn any_pressed(&self) -> bool {
        self.left || self.right || self.up || self.down
    }
}

struct Simulation {
    shaders: Shaders,
    size: ScreenSize,
    view_offset: [f32; 2],
    view_scale: f32,
    cursor_uv: Option<[f32; 2]>,
    navigation: NavigationInput,
    last_navigation_update: Instant,
    seed: u64,
    pending_seed: String,
    cluster_size: u32,
    game_mode: GameMode,
    running: bool,
    step_once: bool,
    target_steps_per_second: f32,
    max_simulation_steps_per_tick: usize,
    measured_steps_per_second: f32,
    steps_in_sample: u32,
    generation: u64,
    show_steps_counter: bool,
    next_step_at: Instant,
    step_sample_started_at: Instant,
    brush_value: u32,
    brush_radius: u32,
    brush_down: bool,
    last_brush_cell: Option<(u32, u32)>,
    in_progress_brush_strokes: Vec<BrushStroke>,
    undo_stack: Vec<BrushUndoUnit>,
}

struct FrameStats {
    last_presented_at: Option<Instant>,
    sample_started_at: Instant,
    presented_frames_in_sample: u32,
    frame_time_ms: f32,
    measured_frames_per_second: f32,
    show_fps_counter: bool,
}

#[derive(Default)]
struct HudAction {
    reset_requested: bool,
    undo_requested: bool,
}

struct BrushUndoUnit {
    strokes: Vec<BrushStroke>,
}

impl FrameStats {
    fn new() -> Self {
        let now = Instant::now();

        Self {
            last_presented_at: None,
            sample_started_at: now,
            presented_frames_in_sample: 0,
            frame_time_ms: 0.0,
            measured_frames_per_second: 0.0,
            show_fps_counter: false,
        }
    }

    fn record_presented_frame(&mut self) {
        let now = Instant::now();
        if let Some(last_presented_at) = self.last_presented_at {
            self.frame_time_ms = (now - last_presented_at).as_secs_f32() * 1_000.0;
        }
        self.last_presented_at = Some(now);

        self.presented_frames_in_sample += 1;
        let sample_time = now - self.sample_started_at;
        if sample_time >= Duration::from_secs(1) {
            self.measured_frames_per_second =
                self.presented_frames_in_sample as f32 / sample_time.as_secs_f32();
            self.presented_frames_in_sample = 0;
            self.sample_started_at = now;
        }
    }
}

impl Simulation {
    fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        config: StartupConfig,
    ) -> Self {
        let contents = initial_board(GRID_SIZE, config.seed, config.cluster_size, GameMode::Rps);
        let shaders = Shaders::new(device, GRID_SIZE, &contents, surface_format);
        let now = Instant::now();

        Self {
            shaders,
            size: GRID_SIZE,
            view_offset: [0.0, 0.0],
            view_scale: 1.0,
            cursor_uv: None,
            navigation: NavigationInput::default(),
            last_navigation_update: now,
            seed: config.seed,
            pending_seed: config.seed.to_string(),
            cluster_size: config.cluster_size,
            game_mode: GameMode::Rps,
            running: true,
            step_once: false,
            target_steps_per_second: 30.0,
            max_simulation_steps_per_tick: 8,
            measured_steps_per_second: 0.0,
            steps_in_sample: 0,
            generation: 0,
            show_steps_counter: false,
            next_step_at: now,
            step_sample_started_at: now,
            brush_value: 1,
            brush_radius: 4,
            brush_down: false,
            last_brush_cell: None,
            in_progress_brush_strokes: Vec::new(),
            undo_stack: Vec::new(),
        }
    }

    fn reset(&mut self, queue: &wgpu::Queue) {
        let seed = self.pending_seed.trim().parse().unwrap_or(self.seed);
        let cluster_size = self.cluster_size.max(1);
        let contents = initial_board(self.size, seed, cluster_size, self.game_mode);
        self.shaders.reset(queue, &contents);
        self.shaders.set_game_mode(queue, self.game_mode);

        let now = Instant::now();
        self.seed = seed;
        self.pending_seed = seed.to_string();
        self.cluster_size = cluster_size;
        self.step_once = false;
        self.measured_steps_per_second = 0.0;
        self.steps_in_sample = 0;
        self.generation = 0;
        self.brush_down = false;
        self.last_brush_cell = None;
        self.in_progress_brush_strokes.clear();
        self.undo_stack.clear();
        self.next_step_at = now;
        self.step_sample_started_at = now;
        println!(
            "map seed: {} (cluster size: {})",
            self.seed, self.cluster_size
        );
    }

    fn update_cursor_for_window(
        &mut self,
        position: egui_winit::winit::dpi::PhysicalPosition<f64>,
        width: u32,
        height: u32,
    ) {
        self.cursor_uv = Some([
            (position.x as f32 / width.max(1) as f32).clamp(0.0, 1.0),
            1.0 - (position.y as f32 / height.max(1) as f32).clamp(0.0, 1.0),
        ]);
    }

    fn handle_navigation_key(&mut self, key: &KeyEvent) -> bool {
        if key.repeat && key.state == ElementState::Pressed {
            return matches!(
                key.physical_key,
                PhysicalKey::Code(KeyCode::KeyA)
                    | PhysicalKey::Code(KeyCode::ArrowLeft)
                    | PhysicalKey::Code(KeyCode::KeyD)
                    | PhysicalKey::Code(KeyCode::ArrowRight)
                    | PhysicalKey::Code(KeyCode::KeyW)
                    | PhysicalKey::Code(KeyCode::ArrowUp)
                    | PhysicalKey::Code(KeyCode::KeyS)
                    | PhysicalKey::Code(KeyCode::ArrowDown)
            );
        }

        let pressed = key.state == ElementState::Pressed;
        match key.physical_key {
            PhysicalKey::Code(KeyCode::KeyA) | PhysicalKey::Code(KeyCode::ArrowLeft) => {
                self.navigation.left = pressed;
                true
            }
            PhysicalKey::Code(KeyCode::KeyD) | PhysicalKey::Code(KeyCode::ArrowRight) => {
                self.navigation.right = pressed;
                true
            }
            PhysicalKey::Code(KeyCode::KeyW) | PhysicalKey::Code(KeyCode::ArrowUp) => {
                self.navigation.up = pressed;
                true
            }
            PhysicalKey::Code(KeyCode::KeyS) | PhysicalKey::Code(KeyCode::ArrowDown) => {
                self.navigation.down = pressed;
                true
            }
            _ => false,
        }
    }

    fn handle_mouse_wheel(&mut self, queue: &wgpu::Queue, delta: MouseScrollDelta) {
        let wheel_delta = match delta {
            MouseScrollDelta::LineDelta(_, y) => y,
            MouseScrollDelta::PixelDelta(position) => position.y as f32 / 120.0,
        };
        if wheel_delta == 0.0 {
            return;
        }

        let cursor_uv = self.cursor_uv.unwrap_or([0.5, 0.5]);
        let old_scale = self.view_scale;
        let new_scale =
            (old_scale * WHEEL_ZOOM_STEP.powf(wheel_delta)).clamp(MIN_VIEW_SCALE, MAX_VIEW_SCALE);
        let world_under_cursor = [
            self.view_offset[0] + cursor_uv[0] / old_scale,
            self.view_offset[1] + cursor_uv[1] / old_scale,
        ];

        self.view_scale = new_scale;
        self.view_offset = [
            world_under_cursor[0] - cursor_uv[0] / new_scale,
            world_under_cursor[1] - cursor_uv[1] / new_scale,
        ];
        self.clamp_view();
        self.write_view(queue);
    }

    fn update_keyboard_navigation(&mut self, queue: &wgpu::Queue, now: Instant) -> bool {
        let dt = (now - self.last_navigation_update).as_secs_f32().min(0.1);
        self.last_navigation_update = now;

        if !self.navigation.any_pressed() || dt == 0.0 {
            return false;
        }

        let step = PAN_SCREENS_PER_SECOND * dt / self.view_scale;
        let old_offset = self.view_offset;
        if self.navigation.left {
            self.view_offset[0] -= step;
        }
        if self.navigation.right {
            self.view_offset[0] += step;
        }
        if self.navigation.down {
            self.view_offset[1] -= step;
        }
        if self.navigation.up {
            self.view_offset[1] += step;
        }

        self.clamp_view();
        if self.view_offset == old_offset {
            return false;
        }

        self.write_view(queue);
        true
    }

    fn clamp_view(&mut self) {
        let max_offset = 1.0 - 1.0 / self.view_scale;
        self.view_offset[0] = self.view_offset[0].clamp(0.0, max_offset);
        self.view_offset[1] = self.view_offset[1].clamp(0.0, max_offset);
    }

    fn write_view(&self, queue: &wgpu::Queue) {
        self.shaders
            .set_view(queue, self.view_offset, self.view_scale);
    }

    fn cursor_cell(&self) -> Option<(u32, u32)> {
        let cursor_uv = self.cursor_uv?;
        let world_uv = [
            self.view_offset[0] + cursor_uv[0] / self.view_scale,
            self.view_offset[1] + cursor_uv[1] / self.view_scale,
        ];

        if !(0.0..1.0).contains(&world_uv[0]) || !(0.0..1.0).contains(&world_uv[1]) {
            return None;
        }

        Some((
            (world_uv[0] * self.size.width as f32) as u32,
            (world_uv[1] * self.size.height as f32) as u32,
        ))
    }

    fn brush_edits_between(&self, from: Option<(u32, u32)>, to: (u32, u32)) -> Vec<BrushEdit> {
        let mut cells = HashSet::new();
        if let Some(from) = from {
            let mut x0 = from.0 as i32;
            let mut y0 = from.1 as i32;
            let x1 = to.0 as i32;
            let y1 = to.1 as i32;
            let dx = (x1 - x0).abs();
            let sx = if x0 < x1 { 1 } else { -1 };
            let dy = -(y1 - y0).abs();
            let sy = if y0 < y1 { 1 } else { -1 };
            let mut err = dx + dy;

            loop {
                self.insert_brush_cells((x0, y0), &mut cells);
                if x0 == x1 && y0 == y1 {
                    break;
                }
                let e2 = 2 * err;
                if e2 >= dy {
                    err += dy;
                    x0 += sx;
                }
                if e2 <= dx {
                    err += dx;
                    y0 += sy;
                }
            }
        } else {
            self.insert_brush_cells((to.0 as i32, to.1 as i32), &mut cells);
        }

        cells
            .into_iter()
            .take(MAX_BRUSH_EDITS)
            .map(|(x, y)| BrushEdit::new(x, y, self.brush_value))
            .collect()
    }

    fn insert_brush_cells(&self, center: (i32, i32), cells: &mut HashSet<(u32, u32)>) {
        let radius = self.brush_radius as i32;
        let radius_squared = radius * radius;
        for y in (center.1 - radius)..=(center.1 + radius) {
            for x in (center.0 - radius)..=(center.0 + radius) {
                let dx = x - center.0;
                let dy = y - center.1;
                if dx * dx + dy * dy > radius_squared {
                    continue;
                }
                if x >= 0 && y >= 0 && x < self.size.width as i32 && y < self.size.height as i32 {
                    cells.insert((x as u32, y as u32));
                }
            }
        }
    }

    fn clear_undo(&mut self) {
        self.in_progress_brush_strokes.clear();
        self.undo_stack.clear();
    }

    fn finish_brush_unit(&mut self) {
        if !self.in_progress_brush_strokes.is_empty() {
            self.undo_stack.push(BrushUndoUnit {
                strokes: std::mem::take(&mut self.in_progress_brush_strokes),
            });
        }
    }

    fn cancel_brush_unit(&mut self) {
        self.brush_down = false;
        self.last_brush_cell = None;
        self.in_progress_brush_strokes.clear();
    }

    fn step_interval(&self) -> Duration {
        let seconds = 1.0 / self.target_steps_per_second.max(1.0);

        if seconds.is_finite() && seconds >= 1e-9 {
            Duration::from_secs_f32(seconds)
        } else {
            Duration::from_nanos(1)
        }
    }

    fn wants_step(&self, now: Instant) -> bool {
        self.step_once || (self.running && now >= self.next_step_at)
    }

    fn record_step(&mut self, now: Instant) {
        if self.step_once {
            self.step_once = false;
        }

        self.generation += 1;
        self.steps_in_sample += 1;

        let sample_time = now - self.step_sample_started_at;
        if sample_time >= Duration::from_secs(1) {
            self.measured_steps_per_second =
                self.steps_in_sample as f32 / sample_time.as_secs_f32();
            self.steps_in_sample = 0;
            self.step_sample_started_at = now;
        }

        if self.running {
            self.next_step_at += self.step_interval();
        } else {
            self.next_step_at = now;
        }
    }

    fn discard_step_debt(&mut self, now: Instant) {
        let interval = self.step_interval();
        while self.next_step_at <= now {
            self.next_step_at += interval;
        }
    }
}

#[derive(Clone, Copy)]
struct StartupConfig {
    seed: u64,
    cluster_size: u32,
    perfect_vsync: bool,
}

impl StartupConfig {
    fn from_args() -> Result<Self, String> {
        let mut seed = None;
        let mut cluster_size = 24;
        let mut perfect_vsync = false;
        let mut args = env::args().skip(1);

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--seed" | "-s" => {
                    let value = args
                        .next()
                        .ok_or_else(|| format!("missing value after {arg}"))?;
                    seed = Some(parse_u64_arg("seed", &value)?);
                }
                "--cluster-size" | "-c" => {
                    let value = args
                        .next()
                        .ok_or_else(|| format!("missing value after {arg}"))?;
                    cluster_size = parse_u32_arg("cluster-size", &value)?.max(1);
                }
                "--pv" => {
                    perfect_vsync = true;
                }
                "--help" | "-h" => {
                    return Err(
                        "usage: cellular-automata [--seed <u64>] [--cluster-size <u32>] [--pv]"
                            .to_string(),
                    );
                }
                _ => return Err(format!("unknown argument: {arg}")),
            }
        }

        Ok(Self {
            seed: seed.unwrap_or_else(random_seed),
            cluster_size,
            perfect_vsync,
        })
    }
}

fn parse_u64_arg(name: &str, value: &str) -> Result<u64, String> {
    value
        .parse()
        .map_err(|_| format!("{name} must be an unsigned integer, got {value:?}"))
}

fn parse_u32_arg(name: &str, value: &str) -> Result<u32, String> {
    value
        .parse()
        .map_err(|_| format!("{name} must be an unsigned integer, got {value:?}"))
}

fn random_seed() -> u64 {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;
    mix64(nanos ^ std::process::id() as u64)
}

fn mix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

fn initial_board(size: ScreenSize, seed: u64, cluster_size: u32, game_mode: GameMode) -> Vec<u32> {
    let mut board = Vec::with_capacity(size.width as usize * size.height as usize);
    let cluster_size = cluster_size.max(1);

    for y in 0..size.height {
        for x in 0..size.width {
            let cluster_x = x / cluster_size;
            let cluster_y = y / cluster_size;
            let noise = mix64(
                seed ^ (cluster_x as u64).wrapping_mul(0xD1B5_4A32_D192_ED03)
                    ^ (cluster_y as u64).wrapping_mul(0xABC9_83AD_8EBC_8A63),
            );
            let cell = match game_mode {
                GameMode::Rps => match noise % 3 {
                    0 => 1,
                    1 => 2,
                    _ => 3,
                },
                GameMode::Life => {
                    if noise % 4 == 0 {
                        1
                    } else {
                        2
                    }
                }
            };
            board.push(cell);
        }
    }

    board
}

struct GpuState {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface_config: wgpu::SurfaceConfiguration,
    frame_stats: FrameStats,
    egui_context: egui::Context,
    egui_state: egui_winit::State,
    egui_renderer: Renderer,

    simulation: Simulation,
}

impl GpuState {
    async fn new(window: Arc<Window>, config: StartupConfig) -> Result<Self, anyhow::Error> {
        let requested_backends = if config.perfect_vsync {
            wgpu::Backends::VULKAN
        } else {
            wgpu::Backends::all().with_env()
        };
        let strict_backend = env::var_os("CELLULAR_AUTOMATA_STRICT_BACKEND").is_some();
        let (surface, adapter) = match Self::request_adapter(
            window.clone(),
            requested_backends,
            config.perfect_vsync,
        )
        .await
        {
            Ok(pair) => pair,
            Err(error)
                if !config.perfect_vsync
                    && wgpu::Backends::from_env().is_some()
                    && !strict_backend =>
            {
                eprintln!(
                    "requested WGPU_BACKEND={:?} is not usable with this window surface: {error}",
                    env::var("WGPU_BACKEND").unwrap_or_default()
                );
                eprintln!("falling back to the default native wgpu backends");
                Self::request_adapter(window.clone(), wgpu::Backends::PRIMARY, false).await?
            }
            Err(error) => return Err(error),
        };
        let adapter_info = adapter.get_info();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("main device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
            })
            .await?;

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(wgpu::TextureFormat::is_srgb)
            .unwrap_or(surface_caps.formats[0]);

        let present_mode = if config.perfect_vsync {
            surface_caps
                .present_modes
                .iter()
                .copied()
                .find(|mode| *mode == wgpu::PresentMode::Fifo)
                .ok_or_else(|| {
                    anyhow::anyhow!("selected surface does not support FIFO present mode")
                })?
        } else {
            surface_caps
                .present_modes
                .iter()
                .copied()
                .find(|mode| *mode == wgpu::PresentMode::Fifo)
                .unwrap_or(surface_caps.present_modes[0])
        };

        let size = window.inner_size();
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        println!("wgpu adapter: {adapter_info:#?}");
        println!("surface formats: {:?}", surface_caps.formats);
        println!("surface present modes: {:?}", surface_caps.present_modes);
        println!("surface alpha modes: {:?}", surface_caps.alpha_modes);
        println!(
            "selected surface config: format={:?}, present_mode={:?}, alpha_mode={:?}, size={}x{}, desired_maximum_frame_latency={}",
            surface_config.format,
            surface_config.present_mode,
            surface_config.alpha_mode,
            surface_config.width,
            surface_config.height,
            surface_config.desired_maximum_frame_latency
        );

        surface.configure(&device, &surface_config);

        let (egui_context, egui_state, egui_renderer) =
            Self::create_egui(&window, &device, surface_config.format);
        let simulation = Simulation::new(&device, surface_config.format, config);

        Ok(Self {
            surface,
            device,
            queue,
            surface_config,
            frame_stats: FrameStats::new(),
            egui_context,
            egui_state,
            egui_renderer,
            simulation,
        })
    }

    async fn request_adapter(
        window: Arc<Window>,
        backends: wgpu::Backends,
        prefer_intel_vulkan: bool,
    ) -> Result<(wgpu::Surface<'static>, wgpu::Adapter), anyhow::Error> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends,
            backend_options: wgpu::BackendOptions::from_env_or_default(),
            flags: wgpu::InstanceFlags::from_env_or_default(),
            ..Default::default()
        });
        let surface = instance.create_surface(window)?;
        let adapter = if prefer_intel_vulkan {
            instance
                .enumerate_adapters(backends)
                .into_iter()
                .find(|adapter| {
                    let info = adapter.get_info();
                    let is_intel =
                        info.vendor == 0x8086 || info.name.to_ascii_lowercase().contains("intel");

                    info.backend == wgpu::Backend::Vulkan
                        && is_intel
                        && adapter.is_surface_supported(&surface)
                })
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "no Intel Vulkan adapter compatible with surface for backends {backends:?}"
                    )
                })?
        } else {
            instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: Some(&surface),
                    force_fallback_adapter: false,
                })
                .await
                .map_err(|error| {
                    anyhow::anyhow!(
                        "no adapter for backends {backends:?} compatible with surface: {error}"
                    )
                })?
        };

        Ok((surface, adapter))
    }

    fn create_egui(
        window: &Window,
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
    ) -> (egui::Context, egui_winit::State, Renderer) {
        let egui_context = egui::Context::default();
        let scale_factor = window.scale_factor() as f32;
        egui_context.set_pixels_per_point(scale_factor);
        let egui_state = egui_winit::State::new(
            egui_context.clone(),
            egui::ViewportId::ROOT,
            window,
            Some(scale_factor),
            None,
            None,
        );
        let egui_renderer = Renderer::new(device, surface_format, RendererOptions::default());

        (egui_context, egui_state, egui_renderer)
    }

    fn resize(&mut self, window: &Window) {
        let size = window.inner_size();
        if size.width == 0 || size.height == 0 {
            return;
        }

        self.egui_context
            .set_pixels_per_point(window.scale_factor() as f32);

        self.surface_config.width = size.width;
        self.surface_config.height = size.height;
        self.configure_surface();
    }

    fn configure_surface(&self) {
        self.surface.configure(&self.device, &self.surface_config);
    }

    fn run_hud(&mut self, window: &Window) -> (egui::FullOutput, HudAction) {
        let raw_input = self.egui_state.take_egui_input(window);
        let mut action = HudAction::default();
        let output = self.egui_context.run(raw_input, |ctx| {
            action = Self::draw_hud(ctx, &mut self.simulation, &mut self.frame_stats);
        });
        (output, action)
    }

    fn draw_hud(
        ctx: &egui::Context,
        simulation: &mut Simulation,
        frame_stats: &mut FrameStats,
    ) -> HudAction {
        let mut action = HudAction::default();

        egui::Window::new("GPU Cellular Automaton")
            .default_open(false)
            .default_pos([12.0, 12.0])
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.checkbox(&mut frame_stats.show_fps_counter, "FPS");
                    ui.checkbox(&mut simulation.show_steps_counter, "Steps");
                });
                egui::CollapsingHeader::new("Simulation")
                    .default_open(true)
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.label("Mode");
                            let old_mode = simulation.game_mode;
                            egui::ComboBox::from_id_salt("game_mode")
                                .selected_text(simulation.game_mode.label())
                                .show_ui(ui, |ui| {
                                    ui.selectable_value(
                                        &mut simulation.game_mode,
                                        GameMode::Rps,
                                        GameMode::Rps.label(),
                                    );
                                    ui.selectable_value(
                                        &mut simulation.game_mode,
                                        GameMode::Life,
                                        GameMode::Life.label(),
                                    );
                                });
                            if simulation.game_mode != old_mode {
                                action.reset_requested = true;
                            }
                        });
                        ui.horizontal(|ui| {
                            let label = if simulation.running { "Pause" } else { "Run" };
                            if ui.button(label).clicked() {
                                if !simulation.running {
                                    simulation.clear_undo();
                                    simulation.cancel_brush_unit();
                                }
                                simulation.running = !simulation.running;
                            }

                            if ui
                                .add_enabled(!simulation.running, egui::Button::new("Step"))
                                .clicked()
                            {
                                simulation.step_once = true;
                            }
                        });

                        ui.add(
                            egui::Slider::new(
                                &mut simulation.target_steps_per_second,
                                1.0..=REASONABLE_MAX_TARGET_STEPS_PER_SECOND,
                            )
                            .text("target steps / second")
                            .clamping(egui::SliderClamping::Never),
                        );

                        ui.add(
                            egui::Slider::new(
                                &mut simulation.max_simulation_steps_per_tick,
                                1..=REASONABLE_MAX_SIMULATION_STEPS_PER_TICK,
                            )
                            .text("max steps / tick")
                            .clamping(egui::SliderClamping::Never),
                        );

                        ui.label(format!("Generation: {}", simulation.generation));
                    });

                egui::CollapsingHeader::new("Map")
                    .default_open(false)
                    .show(ui, |ui| {
                        ui.label(format!("Seed: {}", simulation.seed));
                        ui.horizontal(|ui| {
                            ui.label("Set seed");
                            ui.text_edit_singleline(&mut simulation.pending_seed);
                        });
                        let seed_is_valid =
                            simulation.pending_seed.trim().parse::<u64>().is_ok();
                        ui.add(
                            egui::Slider::new(
                                &mut simulation.cluster_size,
                                1..=REASONABLE_MAX_STARTUP_CLUSTER_SIZE,
                            )
                            .text("startup cluster size")
                            .clamping(egui::SliderClamping::Never),
                        );
                        ui.horizontal(|ui| {
                            if ui
                                .add_enabled(seed_is_valid, egui::Button::new("Reset map"))
                                .clicked()
                            {
                                action.reset_requested = true;
                            }
                            if ui.button("Random seed").clicked() {
                                simulation.pending_seed = random_seed().to_string();
                                action.reset_requested = true;
                            }
                        });
                        if !seed_is_valid {
                            ui.label("Seed must be a whole number");
                        }
                    });

                egui::CollapsingHeader::new("Brush")
                    .default_open(false)
                    .show(ui, |ui| {
                        ui.add_enabled_ui(!simulation.running, |ui| {
                            ui.add(
                                egui::Slider::new(&mut simulation.brush_radius, 0..=128)
                                    .text("radius"),
                            );
                            match simulation.game_mode {
                                GameMode::Rps => {
                                    egui::ComboBox::from_id_salt("brush_value")
                                        .selected_text(match simulation.brush_value {
                                            0 => "Empty",
                                            1 => "Rock",
                                            2 => "Paper",
                                            3 => "Scissors",
                                            _ => "Invalid",
                                        })
                                        .show_ui(ui, |ui| {
                                            ui.selectable_value(
                                                &mut simulation.brush_value,
                                                0,
                                                "Empty",
                                            );
                                            ui.selectable_value(
                                                &mut simulation.brush_value,
                                                1,
                                                "Rock",
                                            );
                                            ui.selectable_value(
                                                &mut simulation.brush_value,
                                                2,
                                                "Paper",
                                            );
                                            ui.selectable_value(
                                                &mut simulation.brush_value,
                                                3,
                                                "Scissors",
                                            );
                                        });
                                }
                                GameMode::Life => {
                                    egui::ComboBox::from_id_salt("brush_value")
                                        .selected_text(if simulation.brush_value == 1 {
                                            "Live"
                                        } else {
                                            "Dead"
                                        })
                                        .show_ui(ui, |ui| {
                                            ui.selectable_value(
                                                &mut simulation.brush_value,
                                                1,
                                                "Live",
                                            );
                                            ui.selectable_value(
                                                &mut simulation.brush_value,
                                                2,
                                                "Dead",
                                            );
                                        });
                                    if simulation.brush_value != 1
                                        && simulation.brush_value != 2
                                    {
                                        simulation.brush_value = 1;
                                    }
                                }
                            }
                            if ui
                                .add_enabled(
                                    !simulation.undo_stack.is_empty(),
                                    egui::Button::new("Undo brush"),
                                )
                                .clicked()
                            {
                                action.undo_requested = true;
                            }
                        });
                        if simulation.running {
                            ui.label("Pause to paint; running clears brush undo history");
                        } else {
                            ui.label("Left-drag paints into the simulation grid");
                        }
                    });

                egui::CollapsingHeader::new("Info")
                    .default_open(false)
                    .show(ui, |ui| {
                        ui.separator();
                        match simulation.game_mode {
                            GameMode::Rps => {
                                ui.label("Red = rock, green = paper, blue = scissors")
                            }
                            GameMode::Life => {
                                ui.label("White = live, black = dead")
                            }
                        };
                        ui.label(format!(
                            "Steps: {:.1} / second",
                            simulation.measured_steps_per_second
                        ));
                        ui.label(format!(
                            "Frame: {:.2} ms",
                            frame_stats.frame_time_ms
                        ));
                        ui.label(format!(
                            "FPS: {:.1} presented / second",
                            frame_stats.measured_frames_per_second
                        ));
                        ui.label(format!(
                            "Grid: {} x {} cells",
                            simulation.size.width, simulation.size.height
                        ));
                        ui.label(format!(
                            "View: {:.1}x zoom, offset {:.3}, {:.3}",
                            simulation.view_scale,
                            simulation.view_offset[0],
                            simulation.view_offset[1]
                        ));
                        ui.label("Mouse wheel zooms; WASD or arrows move the view");
                    });
            });

        if frame_stats.show_fps_counter || simulation.show_steps_counter {
            egui::Area::new("counters".into())
                .anchor(egui::Align2::RIGHT_TOP, [-12.0, 12.0])
                .interactable(false)
                .show(ctx, |ui| {
                    egui::Frame::window(ui.style())
                        .inner_margin(egui::Margin::same(4))
                        .show(ui, |ui| {
                            ui.set_min_width(80.0);
                            if frame_stats.show_fps_counter {
                                ui.label(format!(
                                    "FPS: {:.1}",
                                    frame_stats.measured_frames_per_second
                                ));
                            }
                            if simulation.show_steps_counter {
                                ui.label(format!(
                                    "Steps: {:.1}/s",
                                    simulation.measured_steps_per_second
                                ));
                            }
                        });
                });
        }

        action
    }

    fn reset_simulation(&mut self) {
        self.simulation.reset(&self.queue);
    }

    fn undo_brush(&mut self) {
        let Some(unit) = self.simulation.undo_stack.pop() else {
            return;
        };
        for stroke in unit.strokes.into_iter().rev() {
            self.simulation.shaders.undo_brush_stroke(
                &self.device,
                &self.queue,
                self.simulation.size,
                stroke,
            );
        }
    }

    fn update_cursor_position(&mut self, position: egui_winit::winit::dpi::PhysicalPosition<f64>) {
        self.simulation.update_cursor_for_window(
            position,
            self.surface_config.width,
            self.surface_config.height,
        );
    }

    fn set_brush_down(&mut self, down: bool) {
        if self.simulation.running {
            self.simulation.cancel_brush_unit();
            return;
        }

        self.simulation.brush_down = down;
        if down {
            self.simulation.in_progress_brush_strokes.clear();
            self.paint_at_cursor();
        } else {
            self.simulation.last_brush_cell = None;
            self.simulation.finish_brush_unit();
        }
    }

    fn paint_at_cursor(&mut self) {
        if self.simulation.running || !self.simulation.brush_down {
            return;
        }
        let Some(cell) = self.simulation.cursor_cell() else {
            return;
        };
        if self.simulation.last_brush_cell == Some(cell) {
            return;
        }

        let edits = self
            .simulation
            .brush_edits_between(self.simulation.last_brush_cell, cell);
        self.simulation.last_brush_cell = Some(cell);
        if let Some(stroke) = self.simulation.shaders.apply_brush_edits(
            &self.device,
            &self.queue,
            self.simulation.size,
            &edits,
        ) {
            self.simulation.in_progress_brush_strokes.push(stroke);
        }
    }

    fn handle_mouse_wheel(&mut self, delta: MouseScrollDelta) {
        self.simulation.handle_mouse_wheel(&self.queue, delta);
    }

    fn handle_navigation_key(&mut self, key: &KeyEvent) -> bool {
        self.simulation.handle_navigation_key(key)
    }

    fn update_keyboard_navigation(&mut self, now: Instant) -> bool {
        self.simulation.update_keyboard_navigation(&self.queue, now)
    }

    fn run_due_simulation_steps(&mut self, now: Instant) -> bool {
        let mut stepped = false;
        let max_steps = self.simulation.max_simulation_steps_per_tick;
        let mut steps = 0;

        for _ in 0..max_steps {
            if !self.simulation.wants_step(now) {
                break;
            }

            self.run_simulation_step(now);
            stepped = true;
            steps += 1;
        }

        let finished_at = Instant::now();
        if steps == max_steps && self.simulation.wants_step(finished_at) {
            self.simulation.discard_step_debt(finished_at);
        }

        stepped
    }

    fn run_simulation_step(&mut self, now: Instant) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("simulation step encoder"),
            });
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rps compute pass"),
                timestamp_writes: None,
            });
            self.simulation
                .shaders
                .compute_step(&mut compute_pass, self.simulation.game_mode);
        }

        self.queue.submit(Some(encoder.finish()));
        self.simulation.record_step(now);
    }

    fn render_simulation_pass(&self, encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView) {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("rps render pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        self.simulation.shaders.render(&mut render_pass);
    }

    fn render_egui_pass(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        paint_jobs: &[egui::ClippedPrimitive],
        screen_descriptor: &ScreenDescriptor,
    ) {
        let render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("egui pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        self.egui_renderer.render(
            &mut render_pass.forget_lifetime(),
            paint_jobs,
            screen_descriptor,
        );
    }

    fn render(&mut self, window: &Window) {
        let output = match self.surface.get_current_texture() {
            Ok(output) => output,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                self.configure_surface();
                return;
            }
            Err(wgpu::SurfaceError::Timeout) => return,
            Err(wgpu::SurfaceError::OutOfMemory) => panic!("wgpu surface is out of memory"),
            Err(wgpu::SurfaceError::Other) => return,
        };
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let (full_output, action) = self.run_hud(window);
        self.egui_state
            .handle_platform_output(window, full_output.platform_output);

        if action.reset_requested {
            self.reset_simulation();
        }
        if action.undo_requested {
            self.undo_brush();
        }

        for (texture_id, image_delta) in &full_output.textures_delta.set {
            self.egui_renderer
                .update_texture(&self.device, &self.queue, *texture_id, image_delta);
        }

        let pixels_per_point = self.egui_context.pixels_per_point();
        let screen_descriptor = ScreenDescriptor {
            size_in_pixels: [self.surface_config.width, self.surface_config.height],
            pixels_per_point,
        };
        let paint_jobs = self
            .egui_context
            .tessellate(full_output.shapes, pixels_per_point);

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("main encoder"),
            });
        self.egui_renderer.update_buffers(
            &self.device,
            &self.queue,
            &mut encoder,
            &paint_jobs,
            &screen_descriptor,
        );

        self.render_simulation_pass(&mut encoder, &view);
        self.render_egui_pass(&mut encoder, &view, &paint_jobs, &screen_descriptor);

        for texture_id in &full_output.textures_delta.free {
            self.egui_renderer.free_texture(texture_id);
        }

        self.queue.submit(Some(encoder.finish()));
        output.present();
        self.frame_stats.record_presented_frame();
    }
}

struct App {
    window: Option<Arc<Window>>,
    gpu: Option<GpuState>,
    config: StartupConfig,
    modifiers: ModifiersState,
}

impl App {
    fn new(config: StartupConfig) -> Self {
        Self {
            window: None,
            gpu: None,
            config,
            modifiers: ModifiersState::default(),
        }
    }

    #[inline(always)]
    fn tick_and_render(&mut self, window: &Window) {
        let Some(gpu) = self.gpu.as_mut() else {
            return;
        };

        let now = Instant::now();
        gpu.update_keyboard_navigation(now);
        gpu.run_due_simulation_steps(now);
        gpu.render(window);
    }

    fn is_undo_shortcut(&self, event: &KeyEvent) -> bool {
        event.state == ElementState::Pressed
            && !event.repeat
            && self.modifiers.control_key()
            && matches!(event.physical_key, PhysicalKey::Code(KeyCode::KeyZ))
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes().with_title("cellular automata"))
                .unwrap(),
        );
        let gpu = match pollster::block_on(GpuState::new(window.clone(), self.config)) {
            Ok(gpu) => gpu,
            Err(error) => {
                eprintln!("failed to initialize GPU: {error:#}");
                event_loop.exit();
                return;
            }
        };
        self.gpu = Some(gpu);
        self.window = Some(window);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, id: WindowId, event: WindowEvent) {
        let Some(window) = self.window.clone() else {
            return;
        };
        if id != window.id() {
            return;
        }

        let consumed_by_egui = self
            .gpu
            .as_mut()
            .is_some_and(|gpu| gpu.egui_state.on_window_event(&window, &event).consumed);

        match event {
            WindowEvent::CloseRequested
            | WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key: Key::Named(NamedKey::Escape),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => event_loop.exit(),
            WindowEvent::Resized(_) => {
                if let Some(gpu) = self.gpu.as_mut() {
                    gpu.resize(&window);
                }
            }
            WindowEvent::ScaleFactorChanged { .. } => {
                if let Some(gpu) = self.gpu.as_mut() {
                    gpu.resize(&window);
                }
            }
            WindowEvent::ModifiersChanged(modifiers) => {
                self.modifiers = modifiers.state();
            }
            WindowEvent::CursorMoved { position, .. } => {
                if let Some(gpu) = self.gpu.as_mut() {
                    gpu.update_cursor_position(position);
                    if !consumed_by_egui {
                        gpu.paint_at_cursor();
                    }
                }
            }
            WindowEvent::MouseInput { state, button, .. } if button == MouseButton::Left => {
                if let Some(gpu) = self.gpu.as_mut() {
                    if state == ElementState::Released || !consumed_by_egui {
                        gpu.set_brush_down(state == ElementState::Pressed);
                    }
                }
            }
            WindowEvent::MouseWheel { delta, .. } if !consumed_by_egui => {
                if let Some(gpu) = self.gpu.as_mut() {
                    gpu.handle_mouse_wheel(delta);
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if self.is_undo_shortcut(&event) {
                    if let Some(gpu) = self.gpu.as_mut() {
                        gpu.undo_brush();
                    }
                    return;
                }

                if let Some(gpu) = self.gpu.as_mut()
                    && (!consumed_by_egui || event.state == ElementState::Released)
                {
                    gpu.handle_navigation_key(&event);
                }
            }
            WindowEvent::RedrawRequested => {
                self.tick_and_render(window.as_ref());
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        let Some(window) = self.window.clone() else {
            event_loop.set_control_flow(ControlFlow::Wait);
            return;
        };

        event_loop.set_control_flow(ControlFlow::Poll);
        self.tick_and_render(window.as_ref());
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = StartupConfig::from_args().map_err(|error| {
        eprintln!("{error}");
        std::io::Error::new(std::io::ErrorKind::InvalidInput, error)
    })?;
    println!(
        "map seed: {} (cluster size: {}, perfect vsync: {})",
        config.seed, config.cluster_size, config.perfect_vsync
    );

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new(config);
    event_loop.run_app(&mut app)?;
    Ok(())
}

use std::env;
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::time::{SystemTime, UNIX_EPOCH};

use egui_wgpu::wgpu;
use egui_wgpu::{Renderer, RendererOptions, ScreenDescriptor};
use egui_winit::winit::application::ApplicationHandler;
use egui_winit::winit::event::{ElementState, KeyEvent, MouseScrollDelta, WindowEvent};
use egui_winit::winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use egui_winit::winit::keyboard::{Key, KeyCode, NamedKey, PhysicalKey};
use egui_winit::winit::window::{Window, WindowId};

mod rps;

use rps::{RpsParams, RpsShaders};

const GRID_SIZE: RpsParams = RpsParams {
    width: 1024,
    height: 1024,
};

const MIN_VIEW_SCALE: f32 = 1.0;
const MAX_VIEW_SCALE: f32 = 128.0;
const WHEEL_ZOOM_STEP: f32 = 1.15;
const PAN_SCREENS_PER_SECOND: f32 = 0.75;
const MAX_SIMULATION_STEPS_PER_TICK: usize = 8;

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
    shaders: RpsShaders,
    size: RpsParams,
    view_offset: [f32; 2],
    view_scale: f32,
    cursor_uv: Option<[f32; 2]>,
    navigation: NavigationInput,
    last_navigation_update: Instant,
    seed: u64,
    pending_seed: String,
    cluster_size: u32,
    running: bool,
    step_once: bool,
    target_steps_per_second: f32,
    measured_steps_per_second: f32,
    steps_in_sample: u32,
    generation: u64,
    frame_time_ms: f32,
    last_frame: Instant,
    next_step_at: Instant,
    step_sample_started_at: Instant,
}

impl Simulation {
    fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        config: StartupConfig,
    ) -> Self {
        let contents = initial_board(GRID_SIZE, config.seed, config.cluster_size);
        let shaders = RpsShaders::new(device, GRID_SIZE, &contents, surface_format);
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
            running: true,
            step_once: false,
            target_steps_per_second: 30.0,
            measured_steps_per_second: 0.0,
            steps_in_sample: 0,
            generation: 0,
            frame_time_ms: 0.0,
            last_frame: now,
            next_step_at: now,
            step_sample_started_at: now,
        }
    }

    fn reset(&mut self, queue: &wgpu::Queue) {
        let seed = self.pending_seed.trim().parse().unwrap_or(self.seed);
        let cluster_size = self.cluster_size.max(1);
        let contents = initial_board(self.size, seed, cluster_size);
        self.shaders.reset(queue, &contents);

        let now = Instant::now();
        self.seed = seed;
        self.pending_seed = seed.to_string();
        self.cluster_size = cluster_size;
        self.step_once = false;
        self.measured_steps_per_second = 0.0;
        self.steps_in_sample = 0;
        self.generation = 0;
        self.next_step_at = now;
        self.step_sample_started_at = now;
        println!(
            "map seed: {} (cluster size: {})",
            self.seed, self.cluster_size
        );
    }

    fn begin_frame(&mut self) {
        let now = Instant::now();
        self.frame_time_ms = (now - self.last_frame).as_secs_f32() * 1_000.0;
        self.last_frame = now;
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

    fn step_interval(&self) -> Duration {
        Duration::from_secs_f32(1.0 / self.target_steps_per_second.max(1.0))
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
            let interval = self.step_interval();
            while self.next_step_at <= now {
                self.next_step_at += interval;
            }
        } else {
            self.next_step_at = now;
        }
    }
}

#[derive(Clone, Copy)]
struct StartupConfig {
    seed: u64,
    cluster_size: u32,
}

impl StartupConfig {
    fn from_args() -> Result<Self, String> {
        let mut seed = None;
        let mut cluster_size = 24;
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
                "--help" | "-h" => {
                    return Err(
                        "usage: cellular-automata [--seed <u64>] [--cluster-size <u32>]"
                            .to_string(),
                    );
                }
                _ => return Err(format!("unknown argument: {arg}")),
            }
        }

        Ok(Self {
            seed: seed.unwrap_or_else(random_seed),
            cluster_size,
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

fn initial_board(size: RpsParams, seed: u64, cluster_size: u32) -> Vec<u32> {
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
            let cell = match noise % 3 {
                0 => 1,
                1 => 2,
                _ => 3,
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
    egui_context: egui::Context,
    egui_state: egui_winit::State,
    egui_renderer: Renderer,

    simulation: Simulation,
}

impl GpuState {
    async fn new(window: Arc<Window>, config: StartupConfig) -> Result<Self, anyhow::Error> {
        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(window.clone()).unwrap();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await?;
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

        let size = window.inner_size();
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        surface.configure(&device, &surface_config);

        let (egui_context, egui_state, egui_renderer) =
            Self::create_egui(&window, &device, surface_config.format);
        let simulation = Simulation::new(&device, surface_config.format, config);

        Ok(Self {
            surface,
            device,
            queue,
            surface_config,
            egui_context,
            egui_state,
            egui_renderer,
            simulation,
        })
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

    fn run_hud(&mut self, window: &Window) -> (egui::FullOutput, bool) {
        let raw_input = self.egui_state.take_egui_input(window);
        let mut reset_requested = false;
        let output = self.egui_context.run(raw_input, |ctx| {
            reset_requested = Self::draw_hud(ctx, &mut self.simulation);
        });
        (output, reset_requested)
    }

    fn draw_hud(ctx: &egui::Context, simulation: &mut Simulation) -> bool {
        let mut reset_requested = false;

        egui::Window::new("")
            .default_open(false)
            .default_pos([12.0, 12.0])
            .show(ctx, |ui| {
                ui.label("GPU cellular automaton");
                ui.separator();
                ui.horizontal(|ui| {
                    let label = if simulation.running { "Pause" } else { "Run" };
                    if ui.button(label).clicked() {
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
                    egui::Slider::new(&mut simulation.target_steps_per_second, 1.0..=240.0)
                        .text("target steps / second"),
                );

                ui.separator();
                ui.label(format!("Seed: {}", simulation.seed));
                ui.horizontal(|ui| {
                    ui.label("Set seed");
                    ui.text_edit_singleline(&mut simulation.pending_seed);
                });
                let seed_is_valid = simulation.pending_seed.trim().parse::<u64>().is_ok();
                ui.add(
                    egui::Slider::new(&mut simulation.cluster_size, 1..=128)
                        .text("startup cluster size"),
                );
                ui.horizontal(|ui| {
                    if ui
                        .add_enabled(seed_is_valid, egui::Button::new("Reset map"))
                        .clicked()
                    {
                        reset_requested = true;
                    }
                    if ui.button("Random seed").clicked() {
                        simulation.pending_seed = random_seed().to_string();
                        reset_requested = true;
                    }
                });
                if !seed_is_valid {
                    ui.label("Seed must be a whole number");
                }

                ui.separator();
                ui.label(format!(
                    "Grid: {} x {} cells",
                    simulation.size.width, simulation.size.height
                ));
                ui.label(format!(
                    "View: {:.1}x zoom, offset {:.3}, {:.3}",
                    simulation.view_scale, simulation.view_offset[0], simulation.view_offset[1]
                ));
                ui.label("Mouse wheel zooms; WASD or arrows move the view");
                ui.label(format!("Generation: {}", simulation.generation));
                ui.label(format!(
                    "Steps: {:.1} / second",
                    simulation.measured_steps_per_second
                ));
                ui.label(format!("Frame: {:.2} ms", simulation.frame_time_ms));
                ui.label("Red = rock, green = paper, blue = scissors");
            });

        reset_requested
    }

    fn reset_simulation(&mut self) {
        self.simulation.reset(&self.queue);
    }

    fn update_cursor_position(&mut self, position: egui_winit::winit::dpi::PhysicalPosition<f64>) {
        self.simulation.update_cursor_for_window(
            position,
            self.surface_config.width,
            self.surface_config.height,
        );
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
        for _ in 0..MAX_SIMULATION_STEPS_PER_TICK {
            if !self.simulation.wants_step(now) {
                break;
            }

            self.run_simulation_step(now);
            stepped = true;
        }

        if stepped && self.simulation.wants_step(now) {
            self.simulation.next_step_at = Instant::now();
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
            self.simulation.shaders.compute_step(&mut compute_pass);
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

        self.simulation.begin_frame();
        let (full_output, reset_requested) = self.run_hud(window);
        self.egui_state
            .handle_platform_output(window, full_output.platform_output);

        if reset_requested {
            self.reset_simulation();
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
    }
}

struct App {
    window: Option<Arc<Window>>,
    gpu: Option<GpuState>,
    config: StartupConfig,
}

impl App {
    fn new(config: StartupConfig) -> Self {
        Self {
            window: None,
            gpu: None,
            config,
        }
    }

    fn tick_and_render(&mut self, window: &Window) {
        let Some(gpu) = self.gpu.as_mut() else {
            return;
        };

        let now = Instant::now();
        gpu.update_keyboard_navigation(now);
        gpu.run_due_simulation_steps(now);
        gpu.render(window);
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
        self.gpu = Some(pollster::block_on(GpuState::new(window.clone(), self.config)).unwrap());
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
            WindowEvent::CursorMoved { position, .. } => {
                if let Some(gpu) = self.gpu.as_mut() {
                    gpu.update_cursor_position(position);
                }
            }
            WindowEvent::MouseWheel { delta, .. } if !consumed_by_egui => {
                if let Some(gpu) = self.gpu.as_mut() {
                    gpu.handle_mouse_wheel(delta);
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let Some(gpu) = self.gpu.as_mut()
                    && (!consumed_by_egui || event.state == ElementState::Released)
                {
                    gpu.handle_navigation_key(&event);
                }
            }
            WindowEvent::RedrawRequested => {
                let Some(gpu) = self.gpu.as_mut() else {
                    return;
                };

                let now = Instant::now();
                gpu.update_keyboard_navigation(now);
                gpu.run_due_simulation_steps(now);
                gpu.render(&window);
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
        "map seed: {} (cluster size: {})",
        config.seed, config.cluster_size
    );

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new(config);
    event_loop.run_app(&mut app)?;
    Ok(())
}

use std::sync::Arc;
use std::time::{Duration, Instant};

use egui_wgpu::wgpu;
use egui_wgpu::{Renderer, RendererOptions, ScreenDescriptor};
use egui_winit::winit::application::ApplicationHandler;
use egui_winit::winit::event::{ElementState, KeyEvent, WindowEvent};
use egui_winit::winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use egui_winit::winit::keyboard::{Key, NamedKey};
use egui_winit::winit::window::{Window, WindowId};

mod rps;

use rps::{GridSize, RpsShaders};

const GRID_SIZE: GridSize = GridSize {
    width: 512,
    height: 512,
};

struct Simulation {
    shaders: RpsShaders,
    size: GridSize,
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
    fn new(device: &wgpu::Device, surface_format: wgpu::TextureFormat) -> Self {
        let contents = initial_board(GRID_SIZE);
        let shaders = RpsShaders::new(device, GRID_SIZE, &contents, surface_format);
        let now = Instant::now();

        Self {
            shaders,
            size: GRID_SIZE,
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

    fn begin_frame(&mut self) {
        let now = Instant::now();
        self.frame_time_ms = (now - self.last_frame).as_secs_f32() * 1_000.0;
        self.last_frame = now;
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
            self.next_step_at = now + self.step_interval();
        } else {
            self.next_step_at = now;
        }
    }
}

fn initial_board(size: GridSize) -> Vec<u32> {
    let mut board = Vec::with_capacity(size.width as usize * size.height as usize);

    for y in 0..size.height {
        for x in 0..size.width {
            let quadrant = (x >= size.width / 2) as u32 + 2 * (y >= size.height / 2) as u32;
            let noise = ((x.wrapping_mul(73_856_093)) ^ (y.wrapping_mul(19_349_663))) % 17;
            let cell = match (quadrant + noise) % 3 {
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
    async fn new(window: Arc<Window>) -> Result<Self, anyhow::Error> {
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
        let simulation = Simulation::new(&device, surface_config.format);

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

    fn run_hud(&mut self, window: &Window) -> egui::FullOutput {
        let raw_input = self.egui_state.take_egui_input(window);
        self.egui_context.run(raw_input, |ctx| {
            Self::draw_hud(ctx, &mut self.simulation);
        })
    }

    fn draw_hud(ctx: &egui::Context, simulation: &mut Simulation) {
        egui::Window::new("Rock Paper Scissors")
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
                ui.label(format!(
                    "Grid: {} x {} cells",
                    simulation.size.width, simulation.size.height
                ));
                ui.label(format!("Generation: {}", simulation.generation));
                ui.label(format!(
                    "Steps: {:.1} / second",
                    simulation.measured_steps_per_second
                ));
                ui.label(format!("Frame: {:.2} ms", simulation.frame_time_ms));
                ui.label("Red = rock, green = paper, blue = scissors");
            });
    }

    fn run_simulation_step(&mut self) {
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
        self.simulation.record_step(Instant::now());
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
        let full_output = self.run_hud(window);
        self.egui_state
            .handle_platform_output(window, full_output.platform_output);

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

#[derive(Default)]
struct App {
    window: Option<Arc<Window>>,
    gpu: Option<GpuState>,
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
        self.gpu = Some(pollster::block_on(GpuState::new(window.clone())).unwrap());
        self.window = Some(window);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, id: WindowId, event: WindowEvent) {
        let Some(window) = self.window.as_ref() else {
            return;
        };
        if id != window.id() {
            return;
        }

        let consumed_by_egui = self
            .gpu
            .as_mut()
            .is_some_and(|gpu| gpu.egui_state.on_window_event(window, &event).consumed);

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
                    gpu.resize(window);
                }
            }
            WindowEvent::ScaleFactorChanged { .. } => {
                if let Some(gpu) = self.gpu.as_mut() {
                    gpu.resize(window);
                }
            }
            WindowEvent::RedrawRequested => {
                if let Some(gpu) = self.gpu.as_mut() {
                    gpu.render(window);
                }
            }
            _ if consumed_by_egui => window.request_redraw(),
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        let (Some(window), Some(gpu)) = (self.window.as_ref(), self.gpu.as_mut()) else {
            event_loop.set_control_flow(ControlFlow::Wait);
            return;
        };

        let now = Instant::now();
        if gpu.simulation.wants_step(now) {
            gpu.run_simulation_step();
            window.request_redraw();
        }

        if gpu.simulation.running {
            event_loop.set_control_flow(ControlFlow::WaitUntil(gpu.simulation.next_step_at));
        } else {
            event_loop.set_control_flow(ControlFlow::Wait);
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Wait);

    let mut app = App::default();
    event_loop.run_app(&mut app)?;
    Ok(())
}

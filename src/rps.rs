use egui_wgpu::wgpu;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RpsParams {
    pub width: u32,
    pub height: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DisplayInfo {
    pub offset: [f32; 2],
    pub scale: f32,
    pub _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DisplayColors {
    pub colors: [[f32; 4]; 4],
}

const DISPLAY_COLORS: [[f32; 4]; 4] = [
    [0.0, 0.0, 0.0, 1.0], // 0 empty
    [1.0, 0.1, 0.1, 1.0], // 1 rock
    [0.1, 0.8, 0.1, 1.0], // 2 paper
    [0.1, 0.2, 1.0, 1.0], // 3 scissors
];

impl DisplayInfo {
    fn new(offset: [f32; 2], scale: f32) -> Self {
        Self {
            offset,
            scale,
            _pad: 0,
        }
    }
}

pub struct RpsShaders {
    compute_pipeline: wgpu::ComputePipeline,
    _size: wgpu::Buffer,
    rw_buffers: [wgpu::Buffer; 2],
    rw_groups: [wgpu::BindGroup; 2],
    idx: usize,

    render_pipeline: wgpu::RenderPipeline,
    display_info_buffer: wgpu::Buffer,
    _display_colors_buffer: wgpu::Buffer,
    render_groups: [wgpu::BindGroup; 2],
}

impl RpsShaders {
    pub fn new(
        device: &wgpu::Device,
        size: RpsParams,
        contents: &[u32],
        surface_format: wgpu::TextureFormat,
    ) -> Self {
        assert_eq!(size.width as usize * size.height as usize, contents.len());

        let compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rps compute shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("rps.wgsl").into()),
        });

        let render_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rps render shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("rps_render.wgsl").into()),
        });

        // -------------------------
        // Shared buffers
        // -------------------------

        let size_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rps grid size uniform"),
            contents: bytemuck::bytes_of(&size),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let display_data = DisplayInfo::new([0.0, 0.0], 1.0);

        let display_info_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rps display info uniform"),
            contents: bytemuck::bytes_of(&display_data),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let display_colors = DisplayColors {
            colors: DISPLAY_COLORS,
        };

        let display_colors_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rps display colors uniform"),
            contents: bytemuck::bytes_of(&display_colors),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let board_usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;

        let rw_buffers = [
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("rps board 0"),
                contents: bytemuck::cast_slice(contents),
                usage: board_usage,
            }),
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("rps board 1"),
                contents: bytemuck::cast_slice(contents),
                usage: board_usage,
            }),
        ];

        // -------------------------
        // Compute pipeline + groups
        // group(0):
        //   binding 0 = RpsParams uniform
        //   binding 1 = input storage read
        //   binding 2 = output storage read_write
        // -------------------------

        let compute_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("rps compute bind group layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("rps compute pipeline layout"),
                bind_group_layouts: &[&compute_bind_group_layout],
                push_constant_ranges: &[],
            });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rps compute pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &compute_shader,
            entry_point: Some("cs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let rw_groups = [
            // idx 0: read board 0, write board 1
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rps compute bind group 0 -> 1"),
                layout: &compute_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: size_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: rw_buffers[0].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: rw_buffers[1].as_entire_binding(),
                    },
                ],
            }),
            // idx 1: read board 1, write board 0
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rps compute bind group 1 -> 0"),
                layout: &compute_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: size_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: rw_buffers[1].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: rw_buffers[0].as_entire_binding(),
                    },
                ],
            }),
        ];

        // -------------------------
        // Render pipeline + groups
        // group(0):
        //   binding 0 = RpsParams uniform
        //   binding 1 = DisplayInfo uniform
        //   binding 2 = DisplayColors uniform
        //   binding 3 = input storage read
        // -------------------------

        let render_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("rps render bind group layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("rps render pipeline layout"),
                bind_group_layouts: &[&render_bind_group_layout],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("rps render pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &render_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &render_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let render_groups = [
            // render board 0
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rps render bind group board 0"),
                layout: &render_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: size_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: display_info_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: display_colors_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: rw_buffers[0].as_entire_binding(),
                    },
                ],
            }),
            // render board 1
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rps render bind group board 1"),
                layout: &render_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: size_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: display_info_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: display_colors_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: rw_buffers[1].as_entire_binding(),
                    },
                ],
            }),
        ];

        println!(
            "rps pipelines: grid={}x{}, board_bytes={}, compute_workgroups_per_step={}, render_format={:?}, render_topology={:?}, render_samples={}, render_draw=3 vertices",
            size.width,
            size.height,
            rw_buffers[0].size(),
            (rw_buffers[0].size() as u32 / 4).div_ceil(256),
            surface_format,
            wgpu::PrimitiveTopology::TriangleList,
            wgpu::MultisampleState::default().count
        );

        Self {
            compute_pipeline,
            _size: size_buffer,
            rw_buffers,
            rw_groups,
            idx: 0,

            render_pipeline,
            display_info_buffer,
            _display_colors_buffer: display_colors_buffer,
            render_groups,
        }
    }

    pub fn compute_step<'a>(&'a mut self, pass: &mut wgpu::ComputePass<'a>) {
        let cell_count = self.rw_buffers[0].size() as u32 / 4; //size of a single cell is i32
        let workgroup_count = (cell_count + 255) / 256;

        pass.set_pipeline(&self.compute_pipeline);
        pass.set_bind_group(0, &self.rw_groups[self.idx], &[]);
        pass.dispatch_workgroups(workgroup_count, 1, 1);

        // idx always means "current readable board"
        self.idx = 1 - self.idx;
    }

    pub fn reset(&mut self, queue: &wgpu::Queue, contents: &[u32]) {
        for buffer in &self.rw_buffers {
            queue.write_buffer(buffer, 0, bytemuck::cast_slice(contents));
        }
        self.idx = 0;
    }

    pub fn set_view(&self, queue: &wgpu::Queue, offset: [f32; 2], scale: f32) {
        let display_data = DisplayInfo::new(offset, scale);
        queue.write_buffer(
            &self.display_info_buffer,
            0,
            bytemuck::bytes_of(&display_data),
        );
    }

    pub fn render<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        pass.set_pipeline(&self.render_pipeline);
        pass.set_bind_group(0, &self.render_groups[self.idx], &[]);
        pass.draw(0..3, 0..1);
    }
}

use egui_wgpu::wgpu;
use wgpu::util::DeviceExt;

use crate::pipelines::ScreenSize;

pub const MAX_BRUSH_EDITS: usize = 65_536;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BrushParams {
    count: u32,
    width: u32,
    height: u32,
    current_board: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BrushEdit {
    pub x: u32,
    pub y: u32,
    pub value: u32,
    pub previous_value: u32,
}

impl BrushEdit {
    pub fn new(x: u32, y: u32, value: u32) -> Self {
        Self {
            x,
            y,
            value,
            previous_value: 0,
        }
    }
}

pub struct BrushStroke {
    buffer: wgpu::Buffer,
    count: u32,
}

pub struct BrushGpu {
    apply_pipeline: wgpu::ComputePipeline,
    undo_pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    params_buffer: wgpu::Buffer,
    edit_buffer: wgpu::Buffer,
}

impl BrushGpu {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("brush shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("brush.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("brush bind group layout"),
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
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
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

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("brush pipeline layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let apply_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("brush apply pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("apply_brush"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let undo_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("brush undo pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("undo_brush"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("brush params"),
            contents: bytemuck::bytes_of(&BrushParams {
                count: 0,
                width: 1,
                height: 1,
                current_board: 0,
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let edit_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("brush edits"),
            size: (MAX_BRUSH_EDITS * std::mem::size_of::<BrushEdit>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            apply_pipeline,
            undo_pipeline,
            bind_group_layout,
            params_buffer,
            edit_buffer,
        }
    }

    pub fn apply(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        board_buffers: [&wgpu::Buffer; 2],
        size: ScreenSize,
        current_board: usize,
        edits: &[BrushEdit],
    ) -> Option<BrushStroke> {
        if edits.is_empty() {
            return None;
        }
        assert!(edits.len() <= MAX_BRUSH_EDITS);

        let count = edits.len() as u32;
        let params = BrushParams {
            count,
            width: size.width,
            height: size.height,
            current_board: current_board as u32,
        };
        let applied_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("brush applied edits"),
            size: (edits.len() * std::mem::size_of::<BrushEdit>()) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let bind_group = self.bind_group(device, &applied_buffer, board_buffers);

        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));
        queue.write_buffer(&self.edit_buffer, 0, bytemuck::cast_slice(edits));

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("brush apply encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("brush apply pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.apply_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(count.div_ceil(256), 1, 1);
        }
        queue.submit(Some(encoder.finish()));

        Some(BrushStroke {
            buffer: applied_buffer,
            count,
        })
    }

    pub fn undo(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        board_buffers: [&wgpu::Buffer; 2],
        size: ScreenSize,
        current_board: usize,
        stroke: BrushStroke,
    ) {
        let params = BrushParams {
            count: stroke.count,
            width: size.width,
            height: size.height,
            current_board: current_board as u32,
        };
        let bind_group = self.bind_group(device, &stroke.buffer, board_buffers);

        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("brush undo encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("brush undo pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.undo_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(stroke.count.div_ceil(256), 1, 1);
        }
        queue.submit(Some(encoder.finish()));
    }

    fn bind_group(
        &self,
        device: &wgpu::Device,
        applied_buffer: &wgpu::Buffer,
        board_buffers: [&wgpu::Buffer; 2],
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("brush bind group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.edit_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: applied_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: board_buffers[0].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: board_buffers[1].as_entire_binding(),
                },
            ],
        })
    }
}

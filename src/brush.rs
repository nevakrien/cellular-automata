use std::collections::VecDeque;
use std::sync::mpsc;

use egui_wgpu::wgpu;
use wgpu::util::DeviceExt;

pub const MAX_BRUSH_EDITS: usize = 65_536*8;
pub const MAX_BRUSH_UNDO_UNITS: usize = 50_000;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BrushParams {
    count: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BrushEdit {
    pub idx: u32,
    pub value: u32,
}

impl BrushEdit {
    pub fn new(idx: u32, value: u32) -> Self {
        Self { idx, value }
    }
}

pub struct BrushStroke {
    edits: Vec<BrushEdit>,
}

struct BrushUndoUnit {
    strokes: Vec<BrushStroke>,
}

#[derive(Default)]
pub struct BrushUndoStack {
    in_progress_strokes: Vec<BrushStroke>,
    undo_units: VecDeque<BrushUndoUnit>,
}

impl BrushUndoStack {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn clear(&mut self) {
        self.in_progress_strokes.clear();
        self.undo_units.clear();
    }

    pub fn begin_unit(&mut self) {
        self.in_progress_strokes.clear();
    }

    pub fn record_stroke(&mut self, stroke: BrushStroke) {
        self.in_progress_strokes.push(stroke);
    }

    pub fn finish_unit(&mut self) {
        if self.in_progress_strokes.is_empty() {
            return;
        }

        if self.undo_units.len() == MAX_BRUSH_UNDO_UNITS {
            self.undo_units.pop_front();
        }
        self.undo_units.push_back(BrushUndoUnit {
            strokes: std::mem::take(&mut self.in_progress_strokes),
        });
    }

    pub fn cancel_unit(&mut self) {
        self.in_progress_strokes.clear();
    }

    pub fn has_undo(&self) -> bool {
        !self.undo_units.is_empty()
    }

    pub fn pop_undo_strokes(&mut self) -> Option<Vec<BrushStroke>> {
        let mut strokes = self.undo_units.pop_back()?.strokes;
        strokes.reverse();
        Some(strokes)
    }
}

pub struct BrushGpu {
    pipeline: wgpu::ComputePipeline,
    params_buffer: wgpu::Buffer,
    edit_buffer: wgpu::Buffer,
    readback_buffer: wgpu::Buffer,
    bind_groups: [wgpu::BindGroup; 2],
}

impl BrushGpu {
    pub fn new(device: &wgpu::Device, board_buffers: [&wgpu::Buffer; 2]) -> Self {
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
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
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

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("brush pipeline layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("brush pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("do_edit"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("brush params"),
            contents: bytemuck::bytes_of(&BrushParams { count: 0 }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let edit_buffer_size = (MAX_BRUSH_EDITS * std::mem::size_of::<BrushEdit>()) as u64;
        let edit_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("brush edits"),
            size: edit_buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("brush edit readback"),
            size: edit_buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let bind_groups = [
            Self::bind_group(
                device,
                &bind_group_layout,
                &params_buffer,
                &edit_buffer,
                board_buffers[0],
                "brush bind group board 0",
            ),
            Self::bind_group(
                device,
                &bind_group_layout,
                &params_buffer,
                &edit_buffer,
                board_buffers[1],
                "brush bind group board 1",
            ),
        ];

        Self {
            pipeline,
            params_buffer,
            edit_buffer,
            readback_buffer,
            bind_groups,
        }
    }

    pub fn apply(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        current_board: usize,
        edits: &[BrushEdit],
    ) -> Option<BrushStroke> {
        if edits.is_empty() {
            return None;
        }

        let mut undo_edits = Vec::with_capacity(edits.len());
        for batch in edits.chunks(MAX_BRUSH_EDITS) {
            let inverse = self.apply_batch(device, queue, current_board, batch);
            undo_edits.extend(inverse);
        }

        Some(BrushStroke { edits: undo_edits })
    }

    pub fn undo(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        current_board: usize,
        stroke: BrushStroke,
    ) {
        for batch in stroke.edits.chunks(MAX_BRUSH_EDITS) {
            self.write_batch(device, queue, current_board, batch);
        }
    }

    fn apply_batch(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        current_board: usize,
        edits: &[BrushEdit],
    ) -> Vec<BrushEdit> {
        let byte_len = self.dispatch_batch(device, queue, current_board, edits, true);

        let slice = self.readback_buffer.slice(..byte_len);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        rx.recv()
            .expect("brush readback callback should run")
            .expect("brush readback should map");

        let mapped = slice.get_mapped_range();
        let inverse = bytemuck::cast_slice(&mapped).to_vec();
        drop(mapped);
        self.readback_buffer.unmap();
        inverse
    }

    fn write_batch(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        current_board: usize,
        edits: &[BrushEdit],
    ) {
        self.dispatch_batch(device, queue, current_board, edits, false);
    }

    fn dispatch_batch(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        current_board: usize,
        edits: &[BrushEdit],
        readback: bool,
    ) -> u64 {
        let count = edits.len() as u32;
        let byte_len = std::mem::size_of_val(edits) as u64;
        queue.write_buffer(
            &self.params_buffer,
            0,
            bytemuck::bytes_of(&BrushParams { count }),
        );
        queue.write_buffer(&self.edit_buffer, 0, bytemuck::cast_slice(edits));

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("brush encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("brush pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_groups[current_board], &[]);
            pass.dispatch_workgroups(count.div_ceil(256), 1, 1);
        }
        if readback {
            encoder.copy_buffer_to_buffer(&self.edit_buffer, 0, &self.readback_buffer, 0, byte_len);
        }
        queue.submit(Some(encoder.finish()));
        byte_len
    }

    fn bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        params_buffer: &wgpu::Buffer,
        edit_buffer: &wgpu::Buffer,
        board_buffer: &wgpu::Buffer,
        label: &str,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: edit_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: board_buffer.as_entire_binding(),
                },
            ],
        })
    }
}

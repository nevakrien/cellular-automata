struct BrushParams {
    count: u32,
    width: u32,
    height: u32,
    current_board: u32,
};

struct BrushEdit {
    x: u32,
    y: u32,
    value: u32,
    previous_value: u32,
};

@group(0) @binding(0)
var<uniform> params: BrushParams;

@group(0) @binding(1)
var<storage, read> edits: array<BrushEdit>;

@group(0) @binding(2)
var<storage, read_write> applied_edits: array<BrushEdit>;

@group(0) @binding(3)
var<storage, read_write> board_0: array<u32>;

@group(0) @binding(4)
var<storage, read_write> board_1: array<u32>;

@compute @workgroup_size(256)
fn apply_brush(@builtin(global_invocation_id) id: vec3<u32>) {
    let edit_index = id.x;
    if (edit_index >= params.count) {
        return;
    }

    let edit = edits[edit_index];
    if (edit.x >= params.width || edit.y >= params.height) {
        return;
    }

    let board_index = edit.x + edit.y * params.width;
    let previous_value = select(board_0[board_index], board_1[board_index], params.current_board == 1u);

    applied_edits[edit_index] = BrushEdit(edit.x, edit.y, edit.value, previous_value);
    board_0[board_index] = edit.value;
    board_1[board_index] = edit.value;
}

@compute @workgroup_size(256)
fn undo_brush(@builtin(global_invocation_id) id: vec3<u32>) {
    let edit_index = id.x;
    if (edit_index >= params.count) {
        return;
    }

    let edit = applied_edits[edit_index];
    if (edit.x >= params.width || edit.y >= params.height) {
        return;
    }

    let board_index = edit.x + edit.y * params.width;
    board_0[board_index] = edit.previous_value;
    board_1[board_index] = edit.previous_value;
}

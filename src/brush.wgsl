struct BrushParams {
    count: u32,
};

struct BrushEdit {
    idx: u32,
    value: u32,
};

@group(0) @binding(0)
var<uniform> params: BrushParams;

@group(0) @binding(1)
var<storage, read_write> edits: array<BrushEdit>;

@group(0) @binding(2)
var<storage, read_write> board: array<u32>;


@compute @workgroup_size(256)
fn do_edit(@builtin(global_invocation_id) id: vec3<u32>) {
    let edit_index = id.x;
    if (edit_index >= params.count) {
        return;
    }

    let edit = edits[edit_index];
    let old = board[edit.idx];
    board[edit.idx] = edit.value;
    edits[edit_index].value = old;
}


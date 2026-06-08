struct RpsParams {
    width: u32,
    height: u32,
};

@group(0) @binding(0)
var<uniform> params: RpsParams;

@group(0) @binding(1)
var<storage, read> input_grid: array<i32>;

@group(0) @binding(2)
var<storage, read_write> output_grid: array<i32>;

// 0 = empty / none
// 1 = rock
// 2 = paper
// 3 = scissors
fn compute_rps(me: i32, other: i32) -> i32 {
    if me == 0 || other == 0 {
        return 0;
    }

    if me == other {
        return 0;
    }

    let diff = (other - me + 3) % 3;

    if diff == 1 {
        return -1; // other beats me
    } else {
        return 1; // me beats other
    }
}

@compute @workgroup_size(256)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x % params.width;
    let y = id.x/params.width;

    if(y>params.width){
        return;
    }
    if(input_grid[id.x]==0){
        return;
    }

    output_grid[id.x]=input_grid[id.x];
    
    var score = 0;
    for(var dy:i32 = -1; dy<2;dy++){
        for(var dx:i32 = -1; dx<2;dx++){
            if(dx==0 && dy==0) {continue;}

            let new_x = i32(x)+dx;
            let new_y = i32(y)+dy;

            if(new_x<0 || u32(new_x)>params.width) {continue;}
            if(new_y<0 || u32(new_y)>params.height) {continue;}

            let other_idx = u32(new_x)+u32(new_y)*params.width;

            let ans = compute_rps(input_grid[id.x],input_grid[other_idx]);
            if(ans>0){
                score += ans;
            }
        }
    }

    if(score<3){
        return;
    }

    for(var dy:i32 = -1; dy<2;dy++){
        for(var dx:i32 = -1; dx<2;dx++){
            if(dx==0 && dy==0) {continue;}

            let new_x = i32(x)+dx;
            let new_y = i32(y)+dy;

            if(new_x<0 || u32(new_x)>params.width) {continue;}
            if(new_y<0 || u32(new_y)>params.height) {continue;}

            let other_idx = u32(new_x)+u32(new_y)*params.width;

            let ans = compute_rps(input_grid[id.x],input_grid[other_idx]);
            if(ans==1){
                output_grid[id.x]=input_grid[other_idx];
                return;
            }
        }
    }

    
}
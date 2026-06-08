struct RpsParams {
    width: u32,
    height: u32,
};

@group(0) @binding(0)
var<uniform> params: RpsParams;

// 0 = none / wall
// 1 = alive
// 2 = dead
// 3+ = invalid
@group(0) @binding(1)
var<storage, read> input_grid: array<i32>;

@group(0) @binding(2)
var<storage, read_write> output_grid: array<i32>;


@compute @workgroup_size(256)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x % params.width;
    let y = id.x/params.width;

    if(y>params.height){
        return;
    }
    if(input_grid[id.x]==0){
        output_grid[id.x] = 0;
        return;
    }

    output_grid[id.x]=input_grid[id.x];
    
    var score = 0;
    for(var dy:i32 = -1; dy<=1;dy++){
        for(var dx:i32 = -1; dx<=1;dx++){
            if(dx==0 && dy==0) {continue;}

            let new_x = i32(x)+dx;
            let new_y = i32(y)+dy;

            if(new_x<0 || u32(new_x)>=params.width) {continue;}
            if(new_y<0 || u32(new_y)>=params.height) {continue;}

            let other_idx = u32(new_x)+u32(new_y)*params.width;

            
            if(input_grid[other_idx]==1){
                score+=1;
            }
        }
    }

    if(score<2){
        output_grid[id.x]=2;
        return;
    } 

    if(score==3 ){
        output_grid[id.x]=1;
        return;
    } 

    if(score>3){
        output_grid[id.x]=2;
        return;
    }
}
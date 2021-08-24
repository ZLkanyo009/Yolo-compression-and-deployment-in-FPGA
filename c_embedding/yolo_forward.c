#include<stdint.h>
#include<stdlib.h>
#include<stdbool.h>
#include<math.h>
#include "weight.h"

#define NO_PADDING      0x00
#define PADDING_UP      0x01
#define PADDING_DOWN    0x02
#define PADDING_LEFT    0x04
#define PADDING_RIGHT   0x08

#define MAX(a,b)(a>b?a:b)
#define MIN(a,b)(a<=b?a:b)

int *vga_bram = 0x42000000;
int *camera_bram = 0x40000000;
int *camera_bram2 = 0x40025800;

#define inp_buf 0x00000000
#define inp_buf2 0x0000018c
#define kernel_buf 0x00000000
#define kernel_buf2  0x00000009
#define bias_buf 0x00000000
#define bias_buf2 0x00000001
#define out_buf 0x00000000
#define psram  0x00000000
#define psram2 0x0004b000    //307200
#define psram_max 0x00800000 //由于本demo所有层都可以完整放在psram中，故不再考虑


const char scale_w[10] ={6, 8, 8, 9, 9, 9, 10, 10, 10 ,9};
const char scale_b[10] ={7, 6, 5, 5, 5, 6, 5, 5, 5 ,10};
const char scale_a[11] = {65536, 4, 8, 8, 8, 8, 8, 8, 8, 16, 4};
const char retune[10] = {11, 10, 10, 11, 11, 10, 11, 11, 11, 10};
const float anchor_size[5][2] = {{0.53, 0.79}, {1.71, 2.36}, {2.89, 6.44}, {6.33, 3.79}, {9.03, 9.74}};
const char stride = 16;

struct INFO{
    int psram_addr;
    int out_w;
    char Tm;
    char out_Tr;
    char out_Tc;
};

struct BOX{
    int x_max;
    int x_min;
    int y_max;
    int y_min;
    float conf;
    char cls;
    bool supression; //是否被nms抑制
};

void pixel_norm_quantize(short int pixel_0bgr, const char scale_a_single,  char *pixel_all_channel_q_pointer){

    float pixel_norm;
    double scale_a_single_pow = pow(2, (double) scale_a_single);

    //red channel
    pixel_norm = pixel_0bgr & 0x000f;
    pixel_norm /= 255.;
    pixel_norm -= 0.485;
    pixel_norm /= 0.229;

    *pixel_all_channel_q_pointer = (char) (pixel_norm * scale_a_single_pow);//需要把第一层的scale变成2的指数幂

    //green channel
    pixel_norm = pixel_0bgr & 0x00f0;
    pixel_norm /= 255.;
    pixel_norm -= 0.456;
    pixel_norm /= 0.224;

    *(pixel_all_channel_q_pointer + 1) = (char) (pixel_norm * scale_a_single_pow);//需要把第一层的scale变成2的指数幂

    //blue channel
    pixel_norm = pixel_0bgr & 0x0f00;
    pixel_norm /= 255.;
    pixel_norm -= 0.406;
    pixel_norm /= 0.225;

    *(pixel_all_channel_q_pointer + 2) = (char) (pixel_norm * scale_a_single_pow);//需要把第一层的scale变成2的指数幂
}

void camera_to_inpBuf(short int *camera_bram_pointer, int inp_buf_addr, \
                      const char TRow, const char TCol, \
                      int inp_w, \
                      const char scale_a_single, \
                      bool is_up_pad, bool is_down_pad, bool is_left_pad, bool is_right_pad){
    char TRow_no_pad = TRow;
    char TCol_no_pad = TCol;
    
    int pixel_all_channel_q = 0;
    char *pixel_all_channel_q_pointer = &pixel_all_channel_q;
    //col 22 
    //row 18
    //最后几层的时候会出现上下都有pad
    if(is_up_pad == 1){TRow_no_pad -= 1;}
    if(is_down_pad == 1){TRow_no_pad -= 1;}
    if(is_left_pad == 1){TCol_no_pad -= 1;}
    if(is_right_pad == 1){TCol_no_pad -= 1;}
    
    //如果是上padding, inp_buf的地址需要在一开始多加一行
    //如果是下padding, 直接使用原地址即可，因为循环到最后时不会去写最后一行的inp_buf
    if(is_up_pad == 1){inp_buf_addr += TCol;}

    for(char h = 0; h < TRow_no_pad; h++){
        for(char w = 0; w < TCol_no_pad; w ++){
            //如果是左padding, inp_buf的地址需要在每行开头多加1
            //如果是右padding, 直接使用原地址即可，因为循环到最后时不会去写最后一个的inp_buf
            if(is_left_pad == 1 && (inp_buf_addr - inp_buf) % TCol == 0){inp_buf_addr += 1;}

            pixel_norm_quantize(*camera_bram_pointer, scale_a_single, pixel_all_channel_q_pointer);            
            write_input_buffer((uint32_t *) pixel_all_channel_q_pointer, (uint32_t) (inp_buf_addr), 0);
            inp_buf_addr += 1;
            camera_bram_pointer += 1;//地址+2，转跳到下一个像素位置

        }
        camera_bram_pointer += (inp_w - TCol_no_pad);
    }
}

void psram_to_inpBuf(int psram_addr, int inp_buf_addr, \
                     const char TRow, const char TCol, \
                     int inp_w, \
                     int Tn, \
                     bool is_up_pad, bool is_down_pad, bool is_left_pad, bool is_right_pad){
    char TRow_no_pad = TRow;
    char TCol_no_pad = TCol;
    
    int pixel_all_channel_q[4] = {0, 0, 0, 0};
    uint32_t *pixel_all_channel_q_pointer = &pixel_all_channel_q;
    //col 22 
    //row 18
    //最后几层的时候会出现上下都有pad
    if(is_up_pad == 1){TRow_no_pad -= 1;}
    if(is_down_pad == 1){TRow_no_pad -= 1;}
    if(is_left_pad == 1){TCol_no_pad -= 1;}
    if(is_right_pad == 1){TCol_no_pad -= 1;}

    //如果是上padding, inp_buf的地址需要在一开始多加一行
    //如果是下padding, 直接使用原地址即可，因为循环到最后时不会去写最后一行的inp_buf
    if(is_up_pad == 1){inp_buf_addr += TCol;}

    for(char h = 0; h < TRow_no_pad; h++){
        for(char w = 0; w < TCol_no_pad; w ++){
            //如果是左padding, inp_buf的地址需要在每行开头多加1
            //如果是右padding, 直接使用原地址即可，因为循环到最后时不会去写最后一个的inp_buf
            if(is_left_pad == 1 && (inp_buf_addr - inp_buf) % TCol == 0){inp_buf_addr += 1;}
            
            read_psram((uint32_t *) pixel_all_channel_q_pointer, (uint32_t) (psram_addr), (int) (Tn * 8 - 1));//此处size是bit
            write_input_buffer((uint32_t *) pixel_all_channel_q_pointer, (uint32_t) (inp_buf_addr), (int) (Tn * 8 / 32 - 1));//此处size是word

            inp_buf_addr += 1;
            psram_addr += Tn;//转跳到下一个像素位置

        }
        psram_addr += Tn * (inp_w - TCol_no_pad);
    }
}

                                                          //h w c*n
void load_weight(int kernel_buf_addr, int kernel_channel, int kernel_num , \
                 char *weight, int kernel_size){
    for(int pixel_index = 0; pixel_index < kernel_size * kernel_size; pixel_index++){
        //a[h + k*2 + j*2*2 + i*2*2*2]
        write_kernel_buffer((uint32_t*) (weight + pixel_index*kernel_channel*kernel_num), \
                            (uint32_t) (kernel_buf_addr + pixel_index), \
                            (int) (kernel_num * kernel_channel * 8 / 32 - 1)); //-1是因为从0开始
    }
}

void load_bias(int bias_buf_addr, int *bias){
    write_bias_buffer((uint32_t*) bias, (uint32_t) bias_buf_addr, 7);
}

void config_ctl_reg(bool is_up_pad, bool is_down_pad, bool is_left_pad, bool is_right_pad, \
                    bool pingpong_i, bool pingpong_w, bool pingpong_b, \
                    bool first_channel_group, bool last_channel_group, bool activ, bool pool){
    char pad_pos = NO_PADDING;
    if(is_up_pad == 1){
        pad_pos = pad_pos | PADDING_UP;
    }
    else if(is_down_pad == 1){
        pad_pos = pad_pos | PADDING_DOWN;
    }
    else if(is_left_pad == 1){
        pad_pos = pad_pos | PADDING_LEFT;
    }
    else if(is_right_pad == 1){
        pad_pos = pad_pos | PADDING_RIGHT;
    }
    set_tile_detile(pad_pos, pingpong_i, pingpong_w, pingpong_b, first_channel_group, last_channel_group, activ, pool);
    
}

void outBuf_to_psRam(int out_buf_addr, int psram_addr, \
                     int *tile_output_save_pointer,\
                     int out_w, \
                     char Tm, char out_Tr, char out_Tc){

    for(char h = 0; h < out_Tr; h++){
        for(char w = 0; w < out_Tc; w++){
            read_output_buffer((uint32_t*) tile_output_save_pointer, (uint32_t) (out_buf_addr), \
                        (int) (Tm * 8 / 32 - 1));//一个像素8位，所以乘8
            write_psram((uint32_t *) tile_output_save_pointer, (uint32_t) (psram_addr), \
                        (int) (Tm * 8 - 1)); 
            out_buf_addr += 1;
            psram_addr += Tm;
        }
        psram_addr += Tm * (out_w - out_Tc);
    }
}

void outBuf_to_array(int out_buf_addr, char *net_output,\
                     int out_w, \
                     char Tm, char out_Tr, char out_Tc, \
                     int c_out){

    for(char h = 0; h < out_Tr; h++){
        for(char w = 0; w < out_Tc; w++){
            read_output_buffer((uint32_t*) net_output, (uint32_t) (out_buf_addr), \
                        (int) (Tm * 8 / 32 - 1));//一个像素8位，所以乘8
            out_buf_addr += 1;
            net_output += c_out;
        }
        net_output += Tm * (out_w - out_Tc);//正常来说Tm * (out_w - out_Tc) = 0
    }
}

void set_quantize_scale(const char scale_a_single_i, const char scale_w_single, const char scale_b_single, \
                        const char retune_single, const char scale_a_single_o){
    char iofs = scale_a_single_i + scale_w_single - retune_single;
    char bofs = scale_b_single - retune_single;
    char oofs = retune_single - scale_a_single_o;

    char idir = 0;
    char bdir = 0;
    char odir = 0;

    if(iofs < 0){
        idir = 1;
        iofs = 0 - iofs;
    }
    if(bofs < 0){
        bdir = 1;
        bofs = 0 - bofs;
    }
    if(oofs < 0){
        odir = 1;
        oofs = 0 - oofs;
    }

    set_offset(iofs, idir, bofs,  bdir,  oofs, odir);
}

bool pingpong_invert(bool pingpong){
    if(pingpong == 0){
        pingpong = 1;
    }
    else if(pingpong == 1){
        pingpong = 0;
    }
    return pingpong;
}

struct INFO first_conv(int inp_w, int inp_h, \
                       char Tr, char Tc, char Tm, char Tn, \
                       char TRow, char TCol, char layer_index, \
                       short int *camera_bram_pointer, int *tile_output_save_pointer, \
                       char *b_conv, char *w_conv, \
                       bool activ, bool pool, \
                       bool pingpong[], \
                       int kernel_size){
    struct INFO layer_info = {0, 0, 0, 0, 0};
    int out_w = inp_w;
    int out_h = inp_h;
    char out_Tr = Tr;
    char out_Tc = Tc;

    char delay_out_Tr = Tr;

    int tile_col_num = inp_w / Tc;
    int tile_row_num = inp_h / Tr; //向下取整
    int tile_total = tile_col_num * (tile_row_num + 1);
    int tile_remain_start_index = tile_col_num * tile_row_num;
    char tile_row_remain_Tr = inp_h - tile_row_num * Tr; //由于16的引入，行会存在remain的情况
    char out_tile_row_remain_Tr = tile_row_remain_Tr;

    int psram_addr = psram;

    bool first_channel_group, last_channel_group;

    bool is_up_pad, is_down_pad;
    bool is_left_pad, is_right_pad;

    first_channel_group = 1, last_channel_group = 1;

    if(pool == 1){//在这里添加output的tile并在下方添加进来
        out_w /= 2;
        out_h /= 2;
        out_Tr /= 2;
        out_Tc /= 2;
        out_tile_row_remain_Tr /= 2;
    }

    //设置tile基本信息
    set_tile_info(Tn, Tm, Tc, Tr);
    //设置每层的量化scale指数
    const char scale_a_single_i = scale_a[layer_index];
    const char scale_w_single = scale_w[layer_index];
    const char scale_b_single = scale_b[layer_index];
    const char retune_single = retune[layer_index];
    const char scale_a_single_o = scale_a[layer_index + 1];

    set_quantize_scale(scale_a_single_i, scale_w_single, scale_b_single, \
                        retune_single, scale_a_single_o);

    //每层结束时的tile_index只能hard code
    //每层输入后等待输出，输出后还需要送入psram
    //需要bias数组
    //char *b_conv1_pointer = (char*) b_conv1;
    load_bias(bias_buf, (char *)b_conv); //所有层的bias都可以一次性全部送入
    //pingpong_b = pingpong_invert(pingpong_b);

    //需要weight数组
    //char *w_conv1_pointer = (char*) w_conv1;
    //weight需要load 
    if(pingpong[1] == 0){
        load_weight(kernel_buf, Tn, Tm, (char *)w_conv, kernel_size);//第一、二层的weight可以一次性全部送入
    }
    else if(pingpong[1] == 1){
        load_weight(kernel_buf2, Tn, Tm, (char *)w_conv, kernel_size);//第一、二层的weight可以一次性全部送入
    }
    pingpong[1] = pingpong_invert(pingpong[1]);
    //load_weight(kernel_buf_addr2, 3, 16, (char *)w_conv1, kernel_size);//第一、二层的weight可以一次性全部送入
    for(int tile_index = 0; tile_index < tile_total; tile_index++){
        if(tile_index % tile_col_num == 0){is_left_pad = 1;}
        else{is_left_pad = 0;}

        if(tile_index % tile_col_num == tile_col_num - 1){is_right_pad = 1;}
        else{is_right_pad = 0;}

        if(tile_index / tile_col_num == 0){is_up_pad = 1;} //tile_index < tile_col_num
        else{is_up_pad = 0;}

        if(tile_index / tile_col_num == (tile_row_num + 1) - 1){is_down_pad = 1;}
        else{is_down_pad = 0;}

        if(tile_index == tile_remain_start_index){
            Tr = tile_row_remain_Tr;
            TRow = tile_row_remain_Tr + 2; //如果kernel size和stride改变，这边会需要更改
            out_Tr = out_tile_row_remain_Tr;
            set_tile_info(Tn, Tm, Tc, Tr);
        }

        else if(tile_index == tile_remain_start_index + 1){
            //delay_Tr = tile_row_remain_Tr;
            //delay_TRow = tile_row_remain_Tr + 2; //如果kernel size和stride改变，这边会需要更改
            delay_out_Tr = out_tile_row_remain_Tr;
        }
        
        //需要第一层的scale_a用于int 8量化
        if(pingpong[0] == 0){
            camera_to_inpBuf(camera_bram_pointer, inp_buf, TRow, TCol, inp_w, scale_a_single_i, \
                             is_up_pad, is_down_pad, is_left_pad, is_right_pad);
        }
        else if(pingpong[0] == 1){
            camera_to_inpBuf(camera_bram_pointer, inp_buf2, TRow, TCol, inp_w, scale_a_single_i, \
                             is_up_pad, is_down_pad, is_left_pad, is_right_pad);
        }
        
        //判断下一个tile_index时取inp的地址应当如何变化
        //有些情况（虽然不在我们的模型里出现过这种情况）中左pad和右pad会同时出现，但上pad和下pad不同时出现，这种情况暂时不考虑
        //最后几层是四种pad都同时出现，这时tile只有一个，不需要考虑在平面上移动取值地址了
        //这里的Tc和Tr不可以使用camera_to_inpBuf函数里的no pad来代替
        //因为camera_to_inpBuf里考虑的是TRow和TCol,跟这边不太一样的
        //camera内一个像素int4
        if(is_right_pad == 1 && is_up_pad == 1){camera_bram_pointer += (Tc + 1) + (Tr - 2) * inp_w;}//加上Tc后已经主动加上一行，再考虑pad，故-2
        else if(is_right_pad == 1 && is_up_pad == 0){camera_bram_pointer += (Tc + 1) + (Tr - 1) * inp_w;}
        else if(is_left_pad == 1){camera_bram_pointer += Tc - 1;}
        else{camera_bram_pointer += Tc;}
        //camera_bram_pointer += Tc; //第一层第一个tile remain两格像素点需要给后续继续使用
        
        if(tile_index != 0){
            wait_cal_done();
            // psram_addr 考虑空间大小和存放feature map的方式，和如何能够准确的找出来对应tile的对应channel
            outBuf_to_psRam(out_buf, psram_addr, tile_output_save_pointer,\
                            out_w, Tm, delay_out_Tr, out_Tc);
            // 最后几层不能使用这种is_right_pad == 1的方式来判断
            // 由于这一部分代码是把上一次tile的output写到psram，所以上次是right pad时
            // 需要psram_addr变到新的一行去，而上次是right pad，这次就是left pad
            if(is_left_pad == 1){psram_addr += Tm * out_Tc + (delay_out_Tr - 1) * (Tm * out_w);}
            else{psram_addr += Tm * out_Tc;}
        }
        
        //配置tile细节
        config_ctl_reg(is_up_pad, is_down_pad, is_left_pad, is_right_pad, \
                       pingpong[0], pingpong[1], pingpong[2], \
                       first_channel_group, last_channel_group, activ, pool);

        start_calculate();

        pingpong[0] = pingpong_invert(pingpong[0]);
        //pingpong_w = pingpong_invert(pingpong_w);
        //pingpong_b = pingpong_invert(pingpong_b);
        //return pingpong;
    }
    layer_info.psram_addr = psram_addr;
    layer_info.out_w = out_w;
    layer_info.Tm = Tm;
    layer_info.out_Tr = out_Tr;
    layer_info.out_Tc = out_Tc;

    return layer_info; //返回最后一个tile需要write到psram的addr，给下一层的一开始使用
}

struct INFO second_conv(int inp_w, int inp_h, \
                        char Tr, char Tc, char Tm, char Tn, \
                        char TRow, char TCol, char layer_index, \
                        int *tile_output_save_pointer, \
                        char *b_conv, char *w_conv, \
                        int psram_in_addr, int psram_out_addr, \
                        struct INFO pre_layer_info, \
                        bool activ, bool pool, \
                        bool pingpong[], \
                        int kernel_size){
    struct INFO layer_info = {0, 0, 0, 0, 0};

    int out_w = inp_w;
    int out_h = inp_h;
    char out_Tr = Tr;
    char out_Tc = Tc;

    char delay_out_Tr = Tr;

    int tile_col_num = inp_w / Tc;
    int tile_row_num = inp_h / Tr; //向下取整
    int tile_total = tile_col_num * (tile_row_num + 1);
    int tile_remain_start_index = tile_col_num * tile_row_num;
    char tile_row_remain_Tr = inp_h - tile_row_num * Tr; //由于16的引入，行会存在remain的情况
    char out_tile_row_remain_Tr = tile_row_remain_Tr;
    

    bool first_channel_group, last_channel_group;

    bool is_up_pad, is_down_pad;
    bool is_left_pad, is_right_pad;

    first_channel_group = 1, last_channel_group = 1;

    if(pool == 1){//在这里添加output的tile并在下方添加进来
        out_w /= 2;
        out_h /= 2;
        out_Tr /= 2;
        out_Tc /= 2;
        out_tile_row_remain_Tr /= 2;
    }

    //设置tile基本信息
    set_tile_info(Tn, Tm, Tc, Tr);
    //设置每层的量化scale指数
    const char scale_a_single_i = scale_a[layer_index];
    const char scale_w_single = scale_w[layer_index];
    const char scale_b_single = scale_b[layer_index];
    const char retune_single = retune[layer_index];
    const char scale_a_single_o = scale_a[layer_index + 1];

    set_quantize_scale(scale_a_single_i, scale_w_single, scale_b_single, \
                        retune_single, scale_a_single_o);

    //每层结束时的tile_index只能hard code
    //每层输入后等待输出，输出后还需要送入psram
    //需要bias数组
    //char *b_conv1_pointer = (char*) b_conv1;
    load_bias(bias_buf, (char *)b_conv); //所有层的bias都可以一次性全部送入
    //pingpong_b = pingpong_invert(pingpong_b);

    //需要weight数组
    //char *w_conv1_pointer = (char*) w_conv1;
    //weight需要load 
    if(pingpong[1] == 0){
        load_weight(kernel_buf, Tn, Tm, (char *)w_conv, kernel_size);//第一、二层的weight可以一次性全部送入
    }
    else if(pingpong[1] == 1){
        load_weight(kernel_buf2, Tn, Tm, (char *)w_conv, kernel_size);//第一、二层的weight可以一次性全部送入
    }
    pingpong[1] = pingpong_invert(pingpong[1]);

    for(int tile_index = 0; tile_index < tile_total; tile_index++){
        //写入channel group
        if(tile_index % tile_col_num == 0){is_left_pad = 1;}
        else{is_left_pad = 0;}

        if(tile_index % tile_col_num == tile_col_num - 1){is_right_pad = 1;}
        else{is_right_pad = 0;}

        if(tile_index / tile_col_num == 0){is_up_pad = 1;} //tile_index < tile_col_num
        else{is_up_pad = 0;}

        if(tile_index / tile_col_num == (tile_row_num + 1) - 1){is_down_pad = 1;}
        else{is_down_pad = 0;}

        if(tile_index == tile_remain_start_index){
            Tr = tile_row_remain_Tr;
            TRow = tile_row_remain_Tr + 2; //如果kernel size和stride改变，这边会需要更改
            out_Tr = out_tile_row_remain_Tr;
            set_tile_info(Tn, Tm, Tc, Tr);
        }
        else if(tile_index == tile_remain_start_index + 1){
            //delay_Tr = tile_row_remain_Tr;
            //delay_TRow = tile_row_remain_Tr + 2; //如果kernel size和stride改变，这边会需要更改
            delay_out_Tr = out_tile_row_remain_Tr;
        }
        
        //需要第一层的scale_a用于int 8量化
        if(pingpong[0] == 0){
            psram_to_inpBuf(psram_in_addr, inp_buf, TRow, TCol, inp_w, Tn, \
                            is_up_pad, is_down_pad, is_left_pad, is_right_pad);
        }
        else if(pingpong[0] == 1){
            psram_to_inpBuf(psram_in_addr, inp_buf2, TRow, TCol, inp_w, Tn, \
                            is_up_pad, is_down_pad, is_left_pad, is_right_pad);
        }
        
        //判断下一个tile_index时取inp的地址应当如何变化
        //有些情况（虽然不在我们的模型里出现过这种情况）中左pad和右pad会同时出现，但上pad和下pad不同时出现，这种情况暂时不考虑
        //最后几层是四种pad都同时出现，这时tile只有一个，不需要考虑在平面上移动取值地址了
        //psram内一个像素int8
        if(is_right_pad == 1 && is_up_pad == 1){psram_in_addr += Tn * ((Tc + 1) + (Tr - 2) * inp_w);}//加上Tc后已经主动加上一行，再考虑pad，故-2
        else if(is_right_pad == 1 && is_up_pad == 0){psram_in_addr += Tn * ((Tc + 1) + (Tr - 2) * inp_w);}
        else if(is_left_pad == 1){psram_in_addr += Tn * (Tc - 1);}
        else{psram_in_addr += Tn * Tc;}
        //camera_bram_pointer += Tc; //第一层第一个tile remain两格像素点需要给后续继续使用
        
        wait_cal_done();
        // psram_addr 考虑空间大小和存放feature map的方式，和如何能够准确的找出来对应tile的对应channel
        //还需要考虑上一层的最后一个output的write
        if(tile_index == 0){
            outBuf_to_psRam(out_buf, pre_layer_info.psram_addr, tile_output_save_pointer,\
                            pre_layer_info.out_w, pre_layer_info.Tm, \
                            pre_layer_info.out_Tr, pre_layer_info.out_Tc);
        }
        else{
            outBuf_to_psRam(out_buf, psram_out_addr, tile_output_save_pointer,\
                            out_w, Tm, delay_out_Tr, out_Tc);
            if(is_left_pad == 1){psram_out_addr += Tm * out_Tc + (delay_out_Tr - 1) * (Tm * out_w);}
            else{psram_out_addr += Tm * out_Tc;}
        }
        
        //配置tile细节
        config_ctl_reg(is_up_pad, is_down_pad, is_left_pad, is_right_pad, \
                        pingpong[0], pingpong[1], pingpong[2], \
                        first_channel_group, last_channel_group, activ, pool);

        start_calculate();

        pingpong[0] = pingpong_invert(pingpong[0]);
        //pingpong_w = pingpong_invert(pingpong_w);
        //pingpong_b = pingpong_invert(pingpong_b);
        //return pingpong;
    }
    layer_info.psram_addr = psram_out_addr;
    layer_info.out_w = out_w;
    layer_info.Tm = Tm;
    layer_info.out_Tr = out_Tr;
    layer_info.out_Tc = out_Tc;

    return layer_info; //返回最后一个tile需要write到psram的addr，给下一层的一开始使用
}


struct INFO conv_normal(int inp_w, int inp_h, \
                        int c_in, int c_out, \
                        char Tr, char Tc, char Tm, char Tn, \
                        char TRow, char TCol, char layer_index, \
                        int *tile_output_save_pointer, \
                        char *b_conv, char *w_conv, \
                        int psram_in_addr2d, int psram_out_addr2d, struct INFO pre_layer_info, \
                        bool activ, bool pool, \
                        bool pingpong[], \
                        int kernel_size){
    struct INFO layer_info = {0, 0, 0, 0, 0};

    int out_w = inp_w;
    int out_h = inp_h;
    char out_Tr = Tr;
    char out_Tc = Tc;

    //char delay_Tr = Tr;   //delay是用于out psram的延时性而声明的
    char delay_out_Tr = Tr;
    //char delay_TRow = TRow;

    int psram_in_addr3d = psram_in_addr2d;
    int psram_out_addr3d = psram_out_addr2d;
    bool flag_psram_in = 0;//用于判断input_psram是应该加到下一页去还是加到下16个channel去

    char channel_repeat = c_in / Tn; //每层硬编码的input_channel数 / Tn;
    char kernel_num_repeat = c_out / Tm;

    int tile_col_num = inp_w / Tc;
    int tile_row_num = inp_h / Tr; //向下取整
    int tile_total = tile_col_num * (tile_row_num + 1);
    int tile_remain_start_index = tile_col_num * tile_row_num;
    char tile_row_remain_Tr = inp_h - tile_row_num * Tr; //由于16的引入，行会存在remain的情况
    char out_tile_row_remain_Tr = tile_row_remain_Tr;
    

    bool first_channel_group, last_channel_group;

    bool is_up_pad, is_down_pad;
    bool is_left_pad, is_right_pad;

    first_channel_group = 1, last_channel_group = 1;

    if(pool == 1){//在这里添加output的tile并在下方添加进来
        out_w /= 2;
        out_h /= 2;
        out_Tr /= 2;
        out_Tc /= 2;
        out_tile_row_remain_Tr /= 2;
    }

    //设置tile基本信息
    set_tile_info(Tn, Tm, Tc, Tr);
    //设置每层的量化scale指数
    const char scale_a_single_i = scale_a[layer_index];
    const char scale_w_single = scale_w[layer_index];
    const char scale_b_single = scale_b[layer_index];
    const char retune_single = retune[layer_index];
    const char scale_a_single_o = scale_a[layer_index + 1];

    set_quantize_scale(scale_a_single_i, scale_w_single, scale_b_single, \
                        retune_single, scale_a_single_o);

    //每层结束时的tile_index只能hard code
    //每层输入后等待输出，输出后还需要送入psram
    //需要bias数组
    //char *b_conv1_pointer = (char*) b_conv1;
    load_bias(bias_buf, (char *)b_conv); //所有层的bias都可以一次性全部送入
    //pingpong_b = pingpong_invert(pingpong_b);

    for(int tile_index = 0; tile_index < tile_total; tile_index++){
        //tile index指的是平面的、二维的tile index
        //写入channel group
        if(tile_index % tile_col_num == 0){is_left_pad = 1;}
        else{is_left_pad = 0;}

        if(tile_index % tile_col_num == tile_col_num - 1){is_right_pad = 1;}
        else{is_right_pad = 0;}

        if(tile_index / tile_col_num == 0){is_up_pad = 1;} //tile_index < tile_col_num
        else{is_up_pad = 0;}

        if(tile_index / tile_col_num == (tile_row_num + 1) - 1){is_down_pad = 1;}
        else{is_down_pad = 0;}

        //这里改了之后还需要考虑out2psram的延时性，是不能完全对应上的。-->已修改
        if(tile_index == tile_remain_start_index){
            Tr = tile_row_remain_Tr;
            TRow = tile_row_remain_Tr + 2; //如果kernel size和stride改变，这边会需要更改
            out_Tr = out_tile_row_remain_Tr;
            set_tile_info(Tn, Tm, Tc, Tr);
        }
        else if(tile_index == tile_remain_start_index + 1){
            //delay_Tr = tile_row_remain_Tr;
            //delay_TRow = tile_row_remain_Tr + 2; //如果kernel size和stride改变，这边会需要更改
            delay_out_Tr = out_tile_row_remain_Tr;
        }

        psram_in_addr3d = psram_in_addr2d; //进入channel循环前先用2d给3d赋值，将初始地址变成channel = 0的地方
        psram_out_addr3d = psram_out_addr2d;//这个还需要仔细考虑，跟input是不一样的喔
        flag_psram_in = 0;
        for(char kernel_group_index = 0; kernel_group_index < kernel_num_repeat; kernel_group_index++){
            for(char channel_group_index = 0; channel_group_index < channel_repeat; channel_group_index++){    
                if(channel_group_index == 0){
                    first_channel_group = 1;
                    last_channel_group = 0;
                } 
                else if(channel_group_index == channel_repeat - 1){
                    first_channel_group = 0;
                    last_channel_group = 1;                
                }
                else{
                    first_channel_group = 0;
                    last_channel_group = 0;                
                }

                //w_conv的addr需要更改
                //h w c*n
                if(pingpong[1] == 0){
                    //第一二层因为c in和c out都不大于tile的Tn和Tm，所以就直接填入即可(防止传参消耗)
                    //第三层开始需要填c in和c out
                    load_weight(kernel_buf, c_in, c_out, (char *)w_conv, kernel_size);
                    w_conv += Tn * Tm;
                }
                else if(pingpong[1] == 1){
                    load_weight(kernel_buf2, c_in, c_out, (char *)w_conv, kernel_size);
                    w_conv += Tn * Tm;
                }

                if(pingpong[0] == 0){
                    psram_to_inpBuf(psram_in_addr3d, inp_buf, TRow, TCol, inp_w, Tn, \
                                    is_up_pad, is_down_pad, is_left_pad, is_right_pad);
                }
                else if(pingpong[0] == 1){
                    psram_to_inpBuf(psram_in_addr3d, inp_buf2, TRow, TCol, inp_w, Tn, \
                                    is_up_pad, is_down_pad, is_left_pad, is_right_pad);
                }
                
                if(flag_psram_in == 0){ //针对Tn和上一层的Tm不相等的问题作出解决
                    psram_in_addr3d += Tn;
                }
                else if(flag_psram_in == 1){
                    psram_in_addr3d += Tm * (inp_w * inp_h) - Tn;
                }
                
                wait_cal_done();
                // psram_addr 考虑空间大小和存放feature map的方式，和如何能够准确的找出来对应tile的对应channel
                if(tile_index == 0 && channel_group_index == 0){
                    outBuf_to_psRam(out_buf, pre_layer_info.psram_addr, tile_output_save_pointer,\
                                    pre_layer_info.out_w, pre_layer_info.Tm, \
                                    pre_layer_info.out_Tr, pre_layer_info.out_Tc);
                }
                else if(channel_group_index == 0){
                    outBuf_to_psRam(out_buf, psram_out_addr3d, tile_output_save_pointer,\
                                out_w, Tm, delay_out_Tr, out_Tc);
                    psram_out_addr3d += Tm * (out_Tc * delay_out_Tr);//存完一组channel连续放的out_tile后会跳转到后一张图
                }
                
                //配置tile细节
                config_ctl_reg(is_up_pad, is_down_pad, is_left_pad, is_right_pad, \
                               pingpong[0], pingpong[1], pingpong[2], \
                               first_channel_group, last_channel_group, activ, pool);

                start_calculate();

                pingpong[0] = pingpong_invert(pingpong[0]);
                pingpong[1] = pingpong_invert(pingpong[1]);
                flag_psram_in = pingpong_invert(flag_psram_in);
                //pingpong_w = pingpong_invert(pingpong_w);
                //pingpong_b = pingpong_invert(pingpong_b);
                //return pingpong;
            }
        }
        //判断下一个tile_index时取inp的地址应当如何变化
        //有些情况（虽然不在我们的模型里出现过这种情况）中左pad和右pad会同时出现，但上pad和下pad不同时出现，这种情况暂时不考虑
        //最后几层是四种pad都同时出现，这时tile只有一个，不需要考虑在平面上移动取值地址了
        if(is_right_pad == 1 && is_up_pad == 1){psram_in_addr2d += Tn * ((Tc + 1) + (Tr - 2) * inp_w);}//加上Tc后已经主动加上一行，再考虑pad，故-2
        //最后一列时不用tile倒退两格，而是进到下一行，所以+2，但因为右pad，所以又-1
        else if(is_right_pad == 1 && is_up_pad == 0){psram_in_addr2d += Tn * ((Tc + 1) + (Tr - 2) * inp_w);}
        else if(is_left_pad == 1){psram_in_addr2d += Tn * (Tc - 1);}
        else{psram_in_addr2d += Tn * Tc;}
        //camera_bram_pointer += Tc; //第一层第一个tile remain两格像素点需要给后续继续使用
        if(tile_index == 0){psram_out_addr2d = psram_out_addr2d;}
        else if(is_left_pad == 1){psram_out_addr2d += Tm * out_Tc + (delay_out_Tr - 1) * (Tm * out_w);}
        else{psram_out_addr2d += Tm * out_Tc;}
    }
    ////为了防止delay_out_Tr的那个tile index大于tile total，而无法赋值的情况
    //delay_out_Tr = out_tile_row_remain_Tr;
    layer_info.psram_addr = psram_out_addr3d;
    layer_info.out_w = out_w;
    layer_info.Tm = Tm;
    layer_info.out_Tr = out_Tr;
    layer_info.out_Tc = out_Tc;

    return layer_info; //返回最后一个tile需要write到psram的addr，给下一层的一开始使用
}

struct INFO conv_last(int inp_w, int inp_h, \
                        int c_in, int c_out, \
                        char Tr, char Tc, char Tm, char Tn, \
                        char TRow, char TCol, char layer_index, \                        
                        int *tile_output_save_pointer, \
                        char *b_conv, char *w_conv, \
                        int psram_in_addr2d, char *net_output, struct INFO pre_layer_info, \
                        bool activ, bool pool, \
                        bool pingpong[], \
                        int kernel_size){
    struct INFO layer_info = {0, 0, 0, 0, 0};

    int out_w = inp_w;
    int out_h = inp_h;
    char out_Tr = Tr;
    char out_Tc = Tc;

    char delay_out_Tr = Tr;
    char delay_Tm = Tm;

    int psram_in_addr3d = psram_in_addr2d;
    bool flag_psram_in = 0;//用于判断input_psram是应该加到下一页去还是加到下16个channel去

    char channel_repeat = c_in / Tn; //每层硬编码的input_channel数 / Tn;
    char kernel_num_repeat = (c_out / Tm) + 1;//最后一层存在余数
    char kernel_num_remain = c_out - kernel_num_repeat * Tm;
    char kernel_num_remain_start_index = kernel_num_repeat - 1;

    int tile_col_num = inp_w / Tc;
    int tile_row_num = inp_h / Tr; //向下取整
    int tile_total = tile_col_num * (tile_row_num + 1);
    int tile_remain_start_index = tile_col_num * tile_row_num;
    char tile_row_remain_Tr = inp_h - tile_row_num * Tr; //由于16的引入，行会存在remain的情况
    char out_tile_row_remain_Tr = tile_row_remain_Tr;

    bool first_channel_group, last_channel_group;

    bool is_up_pad, is_down_pad;
    bool is_left_pad, is_right_pad;

    first_channel_group = 1, last_channel_group = 1;

    if(pool == 1){//在这里添加output的tile并在下方添加进来
        out_w /= 2;
        out_h /= 2;
        out_Tr /= 2;
        out_Tc /= 2;
        out_tile_row_remain_Tr /= 2;
    }

    //设置tile基本信息
    set_tile_info(Tn, Tm, Tc, Tr);
    //设置每层的量化scale指数
    const char scale_a_single_i = scale_a[layer_index];
    const char scale_w_single = scale_w[layer_index];
    const char scale_b_single = scale_b[layer_index];
    const char retune_single = retune[layer_index];
    const char scale_a_single_o = scale_a[layer_index + 1];

    set_quantize_scale(scale_a_single_i, scale_w_single, scale_b_single, \
                        retune_single, scale_a_single_o);


    load_bias(bias_buf, (char *)b_conv); //所有层的bias都可以一次性全部送入

    for(int tile_index = 0; tile_index < tile_total; tile_index++){
        //tile index指的是平面的、二维的tile index
        //写入channel group
        if(tile_index % tile_col_num == 0){is_left_pad = 1;}
        else{is_left_pad = 0;}

        if(tile_index % tile_col_num == tile_col_num - 1){is_right_pad = 1;}
        else{is_right_pad = 0;}

        if(tile_index / tile_col_num == 0){is_up_pad = 1;} //tile_index < tile_col_num
        else{is_up_pad = 0;}

        if(tile_index / tile_col_num == (tile_row_num + 1) - 1){is_down_pad = 1;}
        else{is_down_pad = 0;}

        if(tile_index == tile_remain_start_index){
            Tr = tile_row_remain_Tr;
            TRow = tile_row_remain_Tr + 2; //如果kernel size和stride改变，这边会需要更改
            out_Tr = out_tile_row_remain_Tr;
            set_tile_info(Tn, Tm, Tc, Tr);
        }
        else if(tile_index == tile_remain_start_index + 1){
            //由于delay_Tr和delay_TRow下文不会用到, 只用到delay_out_Tr,故注释掉
            //delay_Tr = tile_row_remain_Tr;  
            //delay_TRow = tile_row_remain_Tr + 2; //如果kernel size和stride改变，这边会需要更改
            delay_out_Tr = out_tile_row_remain_Tr;
        }

        psram_in_addr3d = psram_in_addr2d; //进入channel循环前先用2d给3d赋值，将初始地址变成channel = 0的地方
        flag_psram_in = 0;
        for(char kernel_group_index = 0; kernel_group_index < kernel_num_repeat; kernel_group_index++){
            if(kernel_group_index == kernel_num_remain_start_index){
                Tm = kernel_num_remain;
                set_tile_info(Tn, Tm, Tc, Tr);
            }
            // else if(kernel_group_index == kernel_num_remain_start_index + 1){
            //     delay_Tm = kernel_num_remain;
            // }
            for(char channel_group_index = 0; channel_group_index < channel_repeat; channel_group_index++){    
                if(channel_group_index == 0){
                    first_channel_group = 1;
                    last_channel_group = 0;
                } 
                else if(channel_group_index == channel_repeat - 1){
                    first_channel_group = 0;
                    last_channel_group = 1;                
                }
                else{
                    first_channel_group = 0;
                    last_channel_group = 0;                
                }

                //w_conv的addr需要更改
                //h w c*n
                if(pingpong[1] == 0){
                    //第一二层因为c in和c out都不大于tile的Tn和Tm，所以就直接填入即可(防止传参消耗)
                    //第三层开始需要填c in和c out
                    load_weight(kernel_buf, c_in, c_out, (char *)w_conv, kernel_size);
                    w_conv += Tn * Tm;
                }
                else if(pingpong[1] == 1){
                    load_weight(kernel_buf2, c_in, c_out, (char *)w_conv, kernel_size);
                    w_conv += Tn * Tm;
                }

                if(pingpong[0] == 0){
                    psram_to_inpBuf(psram_in_addr3d, inp_buf, TRow, TCol, inp_w, Tn, \
                                    is_up_pad, is_down_pad, is_left_pad, is_right_pad);
                }
                else if(pingpong[0] == 1){
                    psram_to_inpBuf(psram_in_addr3d, inp_buf2, TRow, TCol, inp_w, Tn, \
                                    is_up_pad, is_down_pad, is_left_pad, is_right_pad);
                }
                
                if(flag_psram_in == 0){ //针对Tn和上一层的Tm不相等的问题作出解决
                    psram_in_addr3d += Tn;
                }
                else if(flag_psram_in == 1){
                    psram_in_addr3d += Tm * (inp_w * inp_h) - Tn;
                }
                
                wait_cal_done();
                // psram_addr 考虑空间大小和存放feature map的方式，和如何能够准确的找出来对应tile的对应channel
                if(tile_index == 0 && channel_group_index == 0){
                    outBuf_to_psRam(out_buf, pre_layer_info.psram_addr, tile_output_save_pointer,\
                                    pre_layer_info.out_w, pre_layer_info.Tm, \
                                    pre_layer_info.out_Tr, pre_layer_info.out_Tc);
                }
                else if(channel_group_index == 0){
                    outBuf_to_array(out_buf, net_output,\
                                    out_w, delay_Tm, delay_out_Tr, out_Tc, c_out);
                    net_output += delay_Tm; //存完一组channel连续放的out_tile后不会跳一张图，
                                            //最后一层将所有channel都连续放
                }
                
                //配置tile细节
                config_ctl_reg(is_up_pad, is_down_pad, is_left_pad, is_right_pad, \
                               pingpong[0], pingpong[1], pingpong[2], \
                               first_channel_group, last_channel_group, activ, pool);

                start_calculate();

                pingpong[0] = pingpong_invert(pingpong[0]);
                pingpong[1] = pingpong_invert(pingpong[1]);
                flag_psram_in = pingpong_invert(flag_psram_in);
                //pingpong_w = pingpong_invert(pingpong_w);
                //pingpong_b = pingpong_invert(pingpong_b);
                //return pingpong;
            }
        }
        delay_Tm = kernel_num_remain;
    }
    delay_out_Tr = out_tile_row_remain_Tr;
    wait_cal_done();
    //因为最后一层所以最后还是等
    outBuf_to_array(out_buf, net_output,\
                    out_w, delay_Tm, delay_out_Tr, out_Tc, c_out);

    //layer_info.psram_addr = net_output; 不需要这个信息了
    layer_info.out_w = out_w;
    layer_info.Tm = Tm;
    layer_info.out_Tr = out_Tr;
    layer_info.out_Tc = out_Tc;

    return layer_info; //返回最后一个tile需要write到psram的addr，给下一层的一开始使用
}


float sigmoid(float x){
    float result_data = (float) (1 / (exp((double) x) + 1));
    return result_data;
}

float dequantize(char output, const char scale_a_single){
    double scale_a_single_pow = pow(2, (double) scale_a_single);
    float output_fp = output / scale_a_single_pow;
    return output_fp;
}

void softmax(float cls_lst[]){
    int i = 0;
    float sum = 0;
    for(i = 0; i < 2; i++)
    {
        cls_lst[i] = exp((double) cls_lst[i]);
        sum += cls_lst[i];
    }
    for(i = 0; i < 2; i++){
        cls_lst[i] = cls_lst[i] / sum;
    }
}

char cls_sort(int cls_lst[])
{
    if(cls_lst[0] >= cls_lst[1]){
        return 0;
    }
    else if(cls_lst[0] < cls_lst[1]){
        return 1;
    }
}

//传入两个候选框变量某一维度坐标值，返回重叠部分的宽和高
int overlap(int a_min, int a_max, int b_max, int b_min) {    
    int sum_sides = a_max - a_min + b_max - b_min;    //同一方向两个边长之和
    int new_sides = MAX(a_max, b_max) - MIN(a_min, b_min);    //同一方向长度并集
    return  sum_sides - new_sides;
}
    
//传入两个候选框变量a、b，返回相交面积值
int box_intersection(int a_x_max, int a_x_min, int a_y_max, int a_y_min,\
                     int b_x_max, int b_x_min, int b_y_max, int b_y_min)    
{
    int w = overlap(a_x_min, a_x_max, b_x_max, b_x_min);
    int h = overlap(a_y_min, a_y_max, b_y_max, b_y_min);
    if (w <= 0 || h <= 0) return 0;
    int area = w * h;
    return w * h;
}

//传入两个候选框变量a、b，返回并集面积值
                //右下角x 左上角x 右下角y 左上角y
int box_union(int a_x_max, int a_x_min, int a_y_max, int a_y_min,\
                int b_x_max, int b_x_min, int b_y_max, int b_y_min) {    
    int insersection = box_intersection(a_x_max, a_x_min, a_y_max, a_y_min,\
                                          b_x_max, b_x_min, b_y_max, b_y_min);
    int areaA = (a_x_max - a_x_min) * (a_y_max - a_y_min);    
    printf("areaA %f \n", areaA);
    int areaB = (b_x_max - b_x_min) * (b_y_max - b_y_min);
    printf("areaB %f \n", areaB);
    int area = areaA + areaB - insersection;
    return area;
}

float box_iou(struct BOX a,\
              struct BOX b)
{
    return (float) box_intersection(a.x_max, a.x_min, a.y_max, a.y_min, b.x_max, b.x_min, b.y_max, b.y_min) / \
           (float) box_union(a.x_max, a.x_min, a.y_max, a.y_min, b.x_max, b.x_min, b.y_max, b.y_min);    
}

struct BOX decode_txtytwth(float tx, float ty, float tw, float th, char cx, char cy, \
                           char anchor_remain_index){
    struct BOX bbox = {0, 0, 0, 0, 0, 0, 0}; 
    float x_center = (sigmoid(tx) + cx) * stride;
    float y_center = (sigmoid(ty) + cy) * stride;
    float w = anchor_size[anchor_remain_index][0] * exp((double) tw) * stride;
    float h = anchor_size[anchor_remain_index][0] * exp((double) th) * stride;
    bbox.x_min =(int) (x_center - w / 2);
    bbox.x_max =(int) (x_center + w / 2);
    bbox.y_min =(int) (y_center - h / 2);
    bbox.y_max =(int) (y_center + h / 2);
    return bbox;
}

int get_boxes(char out_h, char out_w, char anchor_num, float conf_thresh, \
               struct BOX bbox_lst[], \
               char *conf_pointer, char *cls_pointer, char *txtytwth_pointer){
    float conf, cls, tx, ty, tw, th;

    float cls_lst[2];
    char cls;

    int anchor_index = 0;
    int selected_anchor_num = 0;

    for(char h = 0; h < out_h; h++){
        for(char w = 0; w < out_w; w ++){ 
            for(char anchor = 0; anchor < anchor_num; anchor++){
                conf = dequantize(*(conf_pointer), scale_a[10]);
                conf = sigmoid(conf);
                conf_pointer ++;

                cls_lst[0] = dequantize(*(cls_pointer), scale_a[10]);
                cls_pointer ++;
                cls_lst[1] = dequantize(*(cls_pointer), scale_a[10]);
                cls_pointer ++;
                softmax(cls_lst);
                cls = cls_sort(cls_lst);
                conf = conf * cls_lst[(int) cls];

                if(conf > conf_thresh){
                    tx = (float) (*(txtytwth_pointer));
                    txtytwth_pointer ++;
                    ty = (float) (*(txtytwth_pointer));
                    txtytwth_pointer ++;
                    tw = (float) (*(txtytwth_pointer));
                    txtytwth_pointer ++;
                    th = (float) (*(txtytwth_pointer));
                    txtytwth_pointer ++;

                    struct BOX bbox = decode_txtytwth(tx, ty, tw, th, w, h, anchor);
                    bbox.conf = conf;
                    bbox.cls = cls;

                    bbox_lst[selected_anchor_num] = bbox;

                    selected_anchor_num ++;
                }
                anchor_index ++;

                
                // cls先做反量化
                // cls再做softmax（2个做）
                // 比较cls，返回最大的index以及数值(2个求max)
                // sigmoid(conf) * cls_max_val
                // 与conf_thresh比较，若大于，则将此数值以及对应的index(anchor + w * anchor_num + h * out_w * anchor_num)
                // 以及分类的index记录下
            }
            conf_pointer += 30;
            cls_pointer += 20;
            txtytwth_pointer += 15;
        }
    }
    return (selected_anchor_num + 1);
}

void conf_sort(struct BOX src[], int size)
{
    int i, j;
    for (i = 0; i < size - 1; i++){
        for (j = i + 1; j < size; j++){
            if (src[j].conf > src[i].conf){
                struct BOX tmp = src[j];
                src[j] = src[i];
                src[i] = tmp;
            }             
        }
    }
}

int NMS(struct BOX src[], int anchor_num, float nms_thresh){
    int max_index = 0, current_index = 0;
    int j;
    float iou;
    struct BOX tmp;
    while (current_index < anchor_num) {    //探究一轮循环的方法，与所以输出框比较，递归？？
        printf("current_index: %d\n", current_index);
        if (!(src[current_index].supression)) {
            tmp = src[current_index];
            for (j = current_index + 1; j < anchor_num; j++) {
                iou = box_iou(tmp, src[j]);
                if (iou >= nms_thresh)
                    src[j].supression = 1;
            }
            max_index++;
        }
        current_index++;
    }
    return (max_index + 1);
}

void draw_rectangle(struct BOX bbox_lst[], int bbox_lst_length, \
                    short int *camera_bram_pointer,
                    int inp_w){
    short int red = 0x000f;
    short int green = 0x00f0;
    short int pixel = 0x0000;
    int current_index = 0;
    short int *camera_bram_start = camera_bram_pointer;
    int bbox_left_up_index, bbox_right_up_index, bbox_left_down_index, bbox_right_down_index;
    while (current_index < bbox_lst_length){
        printf("current_index: %d\n", current_index);
        if (!(bbox_lst[current_index].supression)){
            bbox_left_up_index = bbox_lst[current_index].x_min + bbox_lst[current_index].y_min * inp_w;
            bbox_right_up_index = bbox_lst[current_index].x_max + bbox_lst[current_index].y_min * inp_w;
            bbox_left_down_index = bbox_lst[current_index].x_min + bbox_lst[current_index].y_max * inp_w;
            bbox_right_down_index = bbox_lst[current_index].x_max + bbox_lst[current_index].y_max * inp_w;
            if(bbox_lst[current_index].cls == 0){pixel = red;}
            else{pixel = green;}
            for(int index = bbox_left_up_index; index <= bbox_right_up_index; index++){
                *(camera_bram_pointer + bbox_left_up_index + index) = pixel;
                *(camera_bram_pointer + bbox_left_down_index + index) = pixel;
            }
            for(int index = bbox_left_up_index; index <= bbox_left_down_index; index += inp_w){
                *(camera_bram_pointer + bbox_left_up_index + index) = pixel;
                *(camera_bram_pointer + bbox_right_up_index + index) = pixel;
            }
        }
        current_index++;
    }
}


void yolo_forward(const char TRow, const char TCol, const char Tr, const char Tc, 
                  const char Tm, const char Tn, \
                  short int *camera_bram_pointer, int *vga_bram_pointer){

    bool is_up_pad, is_down_pad;
    bool is_left_pad, is_right_pad;

    struct INFO pre_layer_info;


    int *tile_output_save = (int *) malloc(Tm); //取一个像素点channel=Tm
    int *tile_output_save_pointer = tile_output_save;

    int inp_w = 320;
    //int inp_h = 240;
    int out_w = 20;
    int out_h = 15;
    int kernel_size = 3;
    bool activ, pool;
    bool pingpong[3] = {0, 0, 0}; //0:pingpong_i, 1:pingpong_w, 2:pingpong_b

    pre_layer_info = first_conv(320, 240, Tr, Tc, 16, 3, TRow, TCol, 0, \
                                camera_bram_pointer, tile_output_save_pointer, \
                                (char *)b_conv0, (char *)w_conv0, \
                                1, 1, pingpong, 3);

    //psram2和psram每过一层对换一下，做pingpong操作
    pre_layer_info = second_conv(160, 120, Tr, Tc, Tm, Tn, TRow, TCol, 1, \
                                 tile_output_save_pointer, \
                                 (char *)b_conv1, (char *)w_conv1, \
                                 psram2, psram, pre_layer_info, \
                                 1, 1, pingpong, 3);

    pre_layer_info = conv_normal(80, 60, 32, 64, Tr, Tc, Tm, Tn, TRow, TCol, 2, \
                                 tile_output_save_pointer, \
                                 (char *)b_conv2, (char *)w_conv2, \
                                 psram, psram2, pre_layer_info, \
                                 1, 0, pingpong, 3);
    
    pre_layer_info = conv_normal(80, 60, 64, 64, Tr, Tc, Tm, Tn, TRow, TCol, 3, \
                                 tile_output_save_pointer, \
                                 (char *)b_conv3, (char *)w_conv3, \
                                 psram2, psram, pre_layer_info, \
                                 1, 1, pingpong, 3);
    
    pre_layer_info = conv_normal(40, 30, 64, 128, Tr, Tc, Tm, Tn, TRow, TCol, 4, \
                                 tile_output_save_pointer, \
                                 (char *)b_conv4, (char *)w_conv4, \
                                 psram, psram2, pre_layer_info, \
                                 1, 0, pingpong, 3);
    
    pre_layer_info = conv_normal(40, 30, 128, 128, Tr, Tc, Tm, Tn, TRow, TCol, 5, \
                                 tile_output_save_pointer, \
                                 (char *)b_conv5, (char *)w_conv5, \
                                 psram2, psram, pre_layer_info, \
                                 1, 1, pingpong, 3);
    
    pre_layer_info = conv_normal(20, 15, 128, 256, 15, Tc, Tm, Tn, 17, TCol, 6, \
                                 tile_output_save_pointer, \
                                 (char *)b_conv6, (char *)w_conv6, \
                                 psram, psram2, pre_layer_info, \
                                 1, 0, pingpong, 3);

    pre_layer_info = conv_normal(20, 15, 256, 256, 15, Tc, Tm, Tn, 17, TCol, 7, \
                                 tile_output_save_pointer, \
                                 (char *)b_conv7, (char *)w_conv7, \
                                 psram2, psram, pre_layer_info, \
                                 1, 0, pingpong, 3);
    
    pre_layer_info = conv_normal(20, 15, 256, 256, 15, Tc, Tm, Tn, 17, TCol, 8, \
                                 tile_output_save_pointer, \
                                 (char *)b_conv8, (char *)w_conv8, \
                                 psram, psram2, pre_layer_info, \
                                 1, 0, pingpong, 3);
    
    char *net_output = (char *) malloc(35 * 20 * 15);
    char *net_output_pointer = net_output; // h, w, c
    pre_layer_info = conv_last(20, 15, 256, 35, 15, Tc, Tm, Tn, 17, TCol, 9, \
                               tile_output_save_pointer, \
                               (char *)b_conv9, (char *)w_conv9, \
                               psram2, net_output_pointer, pre_layer_info, \
                               0, 0, pingpong, 3); 
    free(tile_output_save_pointer);
    
    float conf_thresh=0.01, nms_thresh=0.5;
    char anchor_num = 5;
    int anchor_total = 1500; //anchor_num * out_w * out_h;
    
    struct BOX bbox_lst[anchor_total];  //后续可以考虑使用链表

    int bbox_lst_length = get_boxes(out_h, out_w, anchor_num, conf_thresh, \
                                    bbox_lst, \
                                    net_output_pointer, (net_output_pointer + 5), (net_output_pointer + 15));    
    free(net_output);

    conf_sort(bbox_lst, bbox_lst_length);
    // 将对应的conf的index的bbox做nms
    int NMS_length = NMS(bbox_lst, bbox_lst_length, nms_thresh);

    draw_rectangle(bbox_lst, bbox_lst_length, camera_bram_pointer, inp_w);
    memcpy(vga_bram_pointer, camera_bram_pointer, 76800*2);    
}
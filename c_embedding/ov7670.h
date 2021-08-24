#define DEV_ADDR_WRITE 0x42
#define DEV_ADDR_READ 0X43
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include "hbird_sdk_soc.h"

// send reg_value to device's reg_addr
void ov7670_set_reg_value(uint32_t reg_addr, uint32_t reg_value){

    i2c_send_data(I2C1, DEV_ADDR_WRITE);
    i2c_send_command(I2C1, I2C_START_WRITE);
    i2c_get_ack(I2C1);

    i2c_send_data(I2C1, reg_addr);
    i2c_send_command(I2C1, I2C_WRITE);
    i2c_get_ack(I2C1);

    i2c_send_data(I2C1, reg_value);
    i2c_send_command(I2C1, I2C_STOP_WRITE);
    i2c_get_ack(I2C1);
}

// recv reg_value to device's reg_addr
void ov7670_get_reg_value(uint32_t reg_addr, uint32_t reg_value){
    i2c_send_data(I2C1, DEV_ADDR_WRITE);
    i2c_send_command(I2C1, I2C_START_WRITE);
    i2c_get_ack(I2C1);

    i2c_send_data(I2C1, reg_addr);
    i2c_send_command(I2C1, I2C_STOP_WRITE);
    i2c_get_ack(I2C1);

    i2c_send_data(I2C1, DEV_ADDR_READ);
    i2c_send_command(I2C1, I2C_START_READ);
    i2c_get_ack(I2C1);

    i2c_send_command(I2C1, I2C_STOP_READ+I2C_NACK); // default send A
    i2c_get_ack(I2C1);
    i2c_get_data(I2C1);
}

/*

    ov7670_set_reg_value(0x3a, 0x04);
    ov7670_set_reg_value(0x40, 0xd0);
    ov7670_set_reg_value(0x12, 0x14);
    ov7670_set_reg_value(0x32, 0x80);
    ov7670_set_reg_value(0x17, 0x16);
    ov7670_set_reg_value(0x18, 0x04);
    ov7670_set_reg_value(0x19, 0x02);
    ov7670_set_reg_value(0x1a, 0x7b);
    ov7670_set_reg_value(0x03, 0x06);
    ov7670_set_reg_value(0x0c, 0x04);
    ov7670_set_reg_value(0x3e, 0x00);
    ov7670_set_reg_value(0x70, 0x3a);
    ov7670_set_reg_value(0x71, 0x35);
    ov7670_set_reg_value(0x72, 0x11);
    ov7670_set_reg_value(0x73, 0x00);
    ov7670_set_reg_value(0xa2, 0x02);
    ov7670_set_reg_value(0x11, 0x81);

    ov7670_set_reg_value(0x7a, 0x20);
    ov7670_set_reg_value(0x7b, 0x1c);
    ov7670_set_reg_value(0x7c, 0x28);
    ov7670_set_reg_value(0x7d, 0x3c);
    ov7670_set_reg_value(0x7e, 0x55);
    ov7670_set_reg_value(0x7f, 0x68);
    ov7670_set_reg_value(0x80, 0x76);
    ov7670_set_reg_value(0x81, 0x80);
    ov7670_set_reg_value(0x82, 0x88);
    ov7670_set_reg_value(0x83, 0x8f);
    ov7670_set_reg_value(0x84, 0x96);
    ov7670_set_reg_value(0x85, 0xa3);
    ov7670_set_reg_value(0x86, 0xaf);
    ov7670_set_reg_value(0x87, 0xc4);
    ov7670_set_reg_value(0x88, 0xd7);
    ov7670_set_reg_value(0x89, 0xe8);

    ov7670_set_reg_value(0x13, 0xe0);
    ov7670_set_reg_value(0x00, 0x00);

    ov7670_set_reg_value(0x10, 0x00);
    ov7670_set_reg_value(0x0d, 0x00);
    ov7670_set_reg_value(0x14, 0x28);
    ov7670_set_reg_value(0xa5, 0x05);
    ov7670_set_reg_value(0xab, 0x07);
    ov7670_set_reg_value(0x24, 0x75);
    ov7670_set_reg_value(0x25, 0x63);
    ov7670_set_reg_value(0x26, 0xA5);
    ov7670_set_reg_value(0x9f, 0x78);
    ov7670_set_reg_value(0xa0, 0x68);
    ov7670_set_reg_value(0xa1, 0x03);
    ov7670_set_reg_value(0xa6, 0xdf);
    ov7670_set_reg_value(0xa7, 0xdf);
    ov7670_set_reg_value(0xa8, 0xf0);
    ov7670_set_reg_value(0xa9, 0x90);
    ov7670_set_reg_value(0xaa, 0x94);
    ov7670_set_reg_value(0x13, 0xe5);

    ov7670_set_reg_value(0x0e, 0x61);
    ov7670_set_reg_value(0x0f, 0x4b);
    ov7670_set_reg_value(0x16, 0x02);
    ov7670_set_reg_value(0x1e, 0x07);
    ov7670_set_reg_value(0x21, 0x02);
    ov7670_set_reg_value(0x22, 0x91);
    ov7670_set_reg_value(0x29, 0x07);
    ov7670_set_reg_value(0x33, 0x0b);
    ov7670_set_reg_value(0x35, 0x0b);
    ov7670_set_reg_value(0x37, 0x1d);
    ov7670_set_reg_value(0x38, 0x71);
    ov7670_set_reg_value(0x39, 0x2a);
    ov7670_set_reg_value(0x3c, 0x78);
    ov7670_set_reg_value(0x4d, 0x40);
    ov7670_set_reg_value(0x4e, 0x20);
    ov7670_set_reg_value(0x69, 0x00);
    ov7670_set_reg_value(0x6b, 0x60);
    ov7670_set_reg_value(0x74, 0x19);
    ov7670_set_reg_value(0x8d, 0x4f);
    ov7670_set_reg_value(0x8e, 0x00);
    ov7670_set_reg_value(0x8f, 0x00);
    ov7670_set_reg_value(0x90, 0x00);
    ov7670_set_reg_value(0x91, 0x00);
    ov7670_set_reg_value(0x92, 0x00);
    ov7670_set_reg_value(0x96, 0x00);
    ov7670_set_reg_value(0x9a, 0x80);
    ov7670_set_reg_value(0xb0, 0x84);
    ov7670_set_reg_value(0xb1, 0x0c);
    ov7670_set_reg_value(0xb2, 0x0e);
    ov7670_set_reg_value(0xb3, 0x82);
    ov7670_set_reg_value(0xb8, 0x0a);



    ov7670_set_reg_value(0x43, 0x14);
    ov7670_set_reg_value(0x44, 0xf0);
    ov7670_set_reg_value(0x45, 0x34);
    ov7670_set_reg_value(0x46, 0x58);
    ov7670_set_reg_value(0x47, 0x28);
    ov7670_set_reg_value(0x48, 0x3a);
    ov7670_set_reg_value(0x59, 0x88);
    ov7670_set_reg_value(0x5a, 0x88);
    ov7670_set_reg_value(0x5b, 0x44);
    ov7670_set_reg_value(0x5c, 0x67);
    ov7670_set_reg_value(0x5d, 0x49);
    ov7670_set_reg_value(0x5e, 0x0e);
    ov7670_set_reg_value(0x64, 0x04);
    ov7670_set_reg_value(0x65, 0x20);
    ov7670_set_reg_value(0x66, 0x05);
    ov7670_set_reg_value(0x94, 0x04);
    ov7670_set_reg_value(0x95, 0x08);
    ov7670_set_reg_value(0x6c, 0x0a);
    ov7670_set_reg_value(0x6d, 0x55);
    ov7670_set_reg_value(0x6e, 0x11);
    ov7670_set_reg_value(0x6f, 0x9f);
    ov7670_set_reg_value(0x6a, 0x40);
    ov7670_set_reg_value(0x01, 0x40);
    ov7670_set_reg_value(0x02, 0x40);
    ov7670_set_reg_value(0x13, 0xe7);
    ov7670_set_reg_value(0x15, 0x00);


    ov7670_set_reg_value(0x4f, 0x80);
    ov7670_set_reg_value(0x50, 0x80);
    ov7670_set_reg_value(0x51, 0x00);
    ov7670_set_reg_value(0x52, 0x22);
    ov7670_set_reg_value(0x53, 0x5e);
    ov7670_set_reg_value(0x54, 0x80);
    ov7670_set_reg_value(0x58, 0x9e);

    ov7670_set_reg_value(0x41, 0x08);
    ov7670_set_reg_value(0x3f, 0x00);
    ov7670_set_reg_value(0x75, 0x05);
    ov7670_set_reg_value(0x76, 0xe1);
    ov7670_set_reg_value(0x4c, 0x00);
    ov7670_set_reg_value(0x77, 0x01);
    ov7670_set_reg_value(0x3d, 0xc2);
    ov7670_set_reg_value(0x4b, 0x09);
    ov7670_set_reg_value(0xc9, 0x60);
    ov7670_set_reg_value(0x41, 0x38);
    ov7670_set_reg_value(0x56, 0x40);

    ov7670_set_reg_value(0x34, 0x11);
    ov7670_set_reg_value(0x3b, 0x02);

    ov7670_set_reg_value(0xa4, 0x89);
    ov7670_set_reg_value(0x96, 0x00);
    ov7670_set_reg_value(0x97, 0x30);
    ov7670_set_reg_value(0x98, 0x20);
    ov7670_set_reg_value(0x99, 0x30);
    ov7670_set_reg_value(0x9a, 0x84);
    ov7670_set_reg_value(0x9b, 0x29);
    ov7670_set_reg_value(0x9c, 0x03);
    ov7670_set_reg_value(0x9d, 0x4c);
    ov7670_set_reg_value(0x9e, 0x3f);
    ov7670_set_reg_value(0x78, 0x04);
    ov7670_set_reg_value(0x8c, 0x02);


    */

void camera_init(){
    // i2c module freq
    int i2c_freq = 40000000;
    int scl_freq = 100000;
    int prescaler = 32; // i2c_freq/(5*scl_freq-1);

    // gpio function enable i2c interface
    gpio_iof_config(GPIOB, IOF_I2C_MASK);

    // i2c setup
    i2c_setup(I2C1, prescaler, I2C_CTR_EN_INTEN);




    //ov7670 init
    ov7670_set_reg_value(0x3a, 0x04); //Bit[3]：Y V Y U

    ov7670_set_reg_value(0x40, 0x10); //Bit[5:4]为01：RGB 565（当RGB444没被设置时，若设置了则为RGB444），output range[00] to [FF]

    ov7670_set_reg_value(0x12, 0x14); //Bit[2],Bit[0]为10，显示为RGB ， QVGA CIF等selection都置0
                                      //目前改为QVGA格式
                                      //相比原来有更改

    ov7670_set_reg_value(0x8c, 0x02); //工作在RGB444 xRGB形式 com15(0x40)[4] is high


    ov7670_set_reg_value(0x32, 0x80); //HREF 暂时看不懂，此为default值
    ov7670_set_reg_value(0x17, 0x16); //与默认不同，与百度吻合 https://blog.csdn.net/zdzh1/article/details/21739881
    ov7670_set_reg_value(0x18, 0x04); //与默认不同，与百度吻合
    ov7670_set_reg_value(0x19, 0x02); //与默认不同，与百度吻合
    ov7670_set_reg_value(0x1a, 0x7b); //与默认 相 同，与百度吻合
    ov7670_set_reg_value(0x03, 0x06); //与默认不同，与百度吻合
                                      //相比原来有更改

    ov7670_set_reg_value(0x0c, 0x00); //与默认不同，与百度吻合
                                      //相比原来有更改

    ov7670_set_reg_value(0x3e, 0x00);//与默认 相 同，与百度吻合
    ov7670_set_reg_value(0x70, 0x00);//与默认不同，与百度吻合
                                     //相比原来有更改

    ov7670_set_reg_value(0x71, 0x00);//与默认不同，与百度吻合
                                     //相比原来有更改

    ov7670_set_reg_value(0x72, 0x11);//与默认 相 同，与百度吻合
    ov7670_set_reg_value(0x73, 0x09);//与默认不同，与百度吻合
                                     //相比原来有更改

    ov7670_set_reg_value(0xa2, 0x02);//与默认 相 同，与百度吻合
    ov7670_set_reg_value(0x11, 0x03);//与默认不同，与百度吻合
                                     //相比原来有更改
                        // 内部时钟 CLKRC，对于拍照来说，值越大，越清楚，刷屏现像 越不重，摄像时，值过大 会造成跟不上
                        // 00 时，可能是对于 2.8的屏，对于 2.4屏 会出现 7格分屏现象
                        // 值越大时，摄头移动 图像拖尾就会越严重，反之OK
                        // 3值适中，2已有3分屏，0有7分屏，5拖尾重，10已花屏
                        // [7] 保留
                        // [6] 直接使用外部时钟（没有预分频）
                        // [5~0] 内部时钟分频    内部时钟 ＝ 输入时钟/([5~0] + 1)    [5~0] = 00000~11111

    // GAM 下述全为伽马曲线有关

    ov7670_set_reg_value(0x7a, 0x24);
    ov7670_set_reg_value(0x7b, 0x04);
    ov7670_set_reg_value(0x7c, 0x07);
    ov7670_set_reg_value(0x7d, 0x10);
    ov7670_set_reg_value(0x7e, 0x28);
    ov7670_set_reg_value(0x7f, 0x36);
    ov7670_set_reg_value(0x80, 0x44);
    ov7670_set_reg_value(0x81, 0x52);
    ov7670_set_reg_value(0x82, 0x60);
    ov7670_set_reg_value(0x83, 0x6C);
    ov7670_set_reg_value(0x84, 0x78);
    ov7670_set_reg_value(0x85, 0x8c);
    ov7670_set_reg_value(0x86, 0x9e);
    ov7670_set_reg_value(0x87, 0xBB);
    ov7670_set_reg_value(0x88, 0xD2);
    ov7670_set_reg_value(0x89, 0xe5);
    /*
    ov7670_set_reg_value(0x7a, 0x20);//与默认不同，与百度吻合
    ov7670_set_reg_value(0x7b, 0x1c);//与默认不同，与百度吻合
    ov7670_set_reg_value(0x7c, 0x28);//与默认不同，与百度吻合
    ov7670_set_reg_value(0x7d, 0x3c);//与默认不同，与百度吻合
    ov7670_set_reg_value(0x7e, 0x55);//与默认不同，与百度吻合
    ov7670_set_reg_value(0x7f, 0x68);//与默认不同，与百度吻合
    ov7670_set_reg_value(0x80, 0x76);//与默认不同，与百度吻合
    ov7670_set_reg_value(0x81, 0x80);//与默认不同，与百度吻合
    ov7670_set_reg_value(0x82, 0x88);//与默认不同，与百度吻合
    ov7670_set_reg_value(0x83, 0x8f);//与默认不同，与百度吻合
    ov7670_set_reg_value(0x84, 0x96);//与默认不同，与百度吻合
    ov7670_set_reg_value(0x85, 0xa3);//与默认不同，与百度吻合
    ov7670_set_reg_value(0x86, 0xaf);//与默认不同，与百度吻合
    ov7670_set_reg_value(0x87, 0xc4);//与默认不同，与百度吻合
    ov7670_set_reg_value(0x88, 0xd7);//与默认不同，与百度吻合
    ov7670_set_reg_value(0x89, 0xe8);//与默认不同，与百度吻合
    */

    // ov7670_set_reg_value(0x13, 0xe7);//与默认不同，与百度吻合 尝试e4 e7 e0
    //                                  //相比原来有更改

    //ov7670_set_reg_value(0x00, 0x00); //AGC 自动增益控制 (值越大 能有效控制 黑像时刷花屏现象)
                                      //相比原来有更改
    ov7670_set_reg_value(0x10, 0x00);//与默认不同，与百度吻合
    ov7670_set_reg_value(0x0d, 0x00);//与默认 相 同，与百度吻合
                        // COM4 通用控制4
                        // [7~6] 保留
                        // [5~4] 平均选择（与COM17[7~6]一致）
                        //     00 全窗口
                        //     01 半窗口
                        //     10 1/4窗口
                        //     11 1/4窗口
                        // [3~0] 保留

    ov7670_set_reg_value(0x14, 0x1a);//与默认不同，与百度吻合
                                     //相比原来有更改
    ov7670_set_reg_value(0xa5, 0x05);//与默认不同，与百度吻合
    ov7670_set_reg_value(0xab, 0x07);//与默认不同，与百度吻合
    ov7670_set_reg_value(0x24, 0x75);//与默认 相 同，与百度吻合
    ov7670_set_reg_value(0x25, 0x63);//与默认不同，与百度吻合
    ov7670_set_reg_value(0x26, 0xA5);//与默认不同，与百度吻合
    ov7670_set_reg_value(0x9f, 0x78);//与默认不同，与百度吻合
    ov7670_set_reg_value(0xa0, 0x68);//与默认不同，与百度吻合
    ov7670_set_reg_value(0xa1, 0x03);//reserved
    ov7670_set_reg_value(0xa6, 0xdf);//与默认不同，与百度吻合
    ov7670_set_reg_value(0xa7, 0xdf);//与默认不同，与百度吻合
    ov7670_set_reg_value(0xa8, 0xf0);//与默认不同，与百度吻合
    ov7670_set_reg_value(0xa9, 0x90);//与默认不同，与百度吻合
    ov7670_set_reg_value(0xaa, 0x94);//与默认不同，与百度吻合

    ov7670_set_reg_value(0x0e, 0x61);//reserved
    ov7670_set_reg_value(0x0f, 0x4b);//与默认不同，与百度吻合
    ov7670_set_reg_value(0x16, 0x02);//reserved
    ov7670_set_reg_value(0x1e, 0x27);//与默认不同，与百度吻合
    ov7670_set_reg_value(0x21, 0x02);//与默认 相 同，与百度吻合
    ov7670_set_reg_value(0x22, 0x91);//与默认不同，与百度吻合
    ov7670_set_reg_value(0x29, 0x07);//reserved
    ov7670_set_reg_value(0x33, 0x0b);//与默认不同，与百度吻合
    ov7670_set_reg_value(0x35, 0x0b);//reserved
    ov7670_set_reg_value(0x37, 0x1d);//与默认不同，与百度吻合
    ov7670_set_reg_value(0x38, 0x71);//与默认不同，与百度吻合
    ov7670_set_reg_value(0x39, 0x2a);//与默认不同，与百度吻合
    ov7670_set_reg_value(0x3c, 0x78);//与默认不同，与百度吻合
    ov7670_set_reg_value(0x4d, 0x40);//与默认不同，与百度吻合
    ov7670_set_reg_value(0x4e, 0x20);//reserved


    ov7670_set_reg_value(0x6b, 0xc0);//与默认不同，与百度吻合 尝试e4 e7 e0
                                     //11: input clock x8

                                     //相比原来有更改
                                     // DBLV
                        // [7~6] PLL控制
                        //    00 旁路PLL
                        //    01 输入时钟X4
                        //    10 输入时钟X6
                        //    11 输入时钟X8
                        // [5] 保留
                        // [4] 内部LDO   0 使能    1 旁路
                        // [3~0] 保留

    ov7670_set_reg_value(0x74, 0x19);//与默认不同，与百度吻合
    ov7670_set_reg_value(0x8d, 0x4f);//reserved
    ov7670_set_reg_value(0x8e, 0x00);//reserved
    ov7670_set_reg_value(0x8f, 0x00);//reserved
    ov7670_set_reg_value(0x90, 0x00);//reserved
    ov7670_set_reg_value(0x91, 0x00);//reserved
    ov7670_set_reg_value(0x92, 0x00);//与默认 相 同，与百度吻合
    ov7670_set_reg_value(0x96, 0x00);//reserved
    ov7670_set_reg_value(0x9a, 0x80);//reserved
    ov7670_set_reg_value(0xb0, 0x84);//reserved
    ov7670_set_reg_value(0xb1, 0x0c);//与默认不同，与百度吻合
    ov7670_set_reg_value(0xb2, 0x0e);//reserved
    ov7670_set_reg_value(0xb3, 0x82);//与默认不同，与百度吻合
    ov7670_set_reg_value(0xb8, 0x0a);//reserved



    ov7670_set_reg_value(0x43, 0x14);//reserved
    ov7670_set_reg_value(0x44, 0xf0);//reserved
    ov7670_set_reg_value(0x45, 0x34);//reserved
    ov7670_set_reg_value(0x46, 0x58);//reserved
    ov7670_set_reg_value(0x47, 0x28);//reserved
    ov7670_set_reg_value(0x48, 0x3a);//reserved
    ov7670_set_reg_value(0x59, 0x88);//reserved
    ov7670_set_reg_value(0x5a, 0x88);//reserved
    ov7670_set_reg_value(0x5b, 0x44);//reserved
    ov7670_set_reg_value(0x5c, 0x67);//reserved
    ov7670_set_reg_value(0x5d, 0x49);//reserved
    ov7670_set_reg_value(0x5e, 0x0e);//reserved
    ov7670_set_reg_value(0x64, 0x04);//与默认不同，与百度吻合
    ov7670_set_reg_value(0x65, 0x20);//与默认不同，与百度吻合
    ov7670_set_reg_value(0x66, 0x05);//与默认不同，与百度吻合
    ov7670_set_reg_value(0x94, 0x04);//与默认不同，与百度吻合
    ov7670_set_reg_value(0x95, 0x08);//与默认不同，与百度吻合
    ov7670_set_reg_value(0x6c, 0x0a);//与默认不同，与百度吻合
    ov7670_set_reg_value(0x6d, 0x55);//与默认 相 同，与百度吻合
    ov7670_set_reg_value(0x6e, 0x11);//与默认不同，与百度吻合
    ov7670_set_reg_value(0x6f, 0x9f);//与默认不同，与百度吻合

    // ov7670_set_reg_value(0x01, 0xff);//与默认 相 同，与百度吻合   BLUE 蓝色通道增益 00~FF
    //                                     //相比原来有更改
    // ov7670_set_reg_value(0x02, 0xff);//与默认 相 同，与百度吻合   RED 红色通道增益 00~FF
    //                                     //相比原来有更改
    ov7670_set_reg_value(0x15, 0x00);//与默认 相 同，与百度吻合
    ov7670_set_reg_value(0x69, 0x5d);//与默认不同，与百度吻合 尝试e4 e7 e0
                                     //相比原来有更改GFIX
    ov7670_set_reg_value(0x6a, 0x00);//与默认不同，与百度吻合
    //!!!!!!!!!!!!!!!!!!!!!!
    ov7670_set_reg_value(0x67, 0xc0);
    ov7670_set_reg_value(0x68, 0x80);



    ov7670_set_reg_value(0x4f, 0x80);//与默认不同，与百度吻合 // MTX1 色彩矩阵系数1
    ov7670_set_reg_value(0x50, 0x80);//与默认不同，与百度吻合 // MTX2 色彩矩阵系数2
    ov7670_set_reg_value(0x51, 0x00);//与默认不同，与百度吻合 // MTX3 色彩矩阵系数3
    ov7670_set_reg_value(0x52, 0x22);//与默认不同，与百度吻合 // MTX4 色彩矩阵系数4
    ov7670_set_reg_value(0x53, 0x5e);//与默认不同，与百度吻合 // MTX5 色彩矩阵系数5
    ov7670_set_reg_value(0x54, 0x80);//与默认不同，与百度吻合 // MTX3 色彩矩阵系数3
    ov7670_set_reg_value(0x58, 0x9e);//与默认不同，与百度吻合 // MTXS 色彩矩阵系数5~0的符号

    ov7670_set_reg_value(0x41, 0x38);//与默认不同，与百度吻合
                                    //相比原来有更改
    ov7670_set_reg_value(0x3f, 0x0a);//与默认不同，与百度吻合
                                    //相比原来有更改
    ov7670_set_reg_value(0x75, 0x05);//与默认不同，与百度吻合
    ov7670_set_reg_value(0x76, 0xe1);//与默认不同，与百度吻合
    ov7670_set_reg_value(0x4c, 0x0f);//与默认不同，与百度吻合
                                    //相比原来有更改
    ov7670_set_reg_value(0x77, 0x0a);//与默认不同，与百度吻合
                                    //相比原来有更改
    ov7670_set_reg_value(0x3d, 0xc8);//与默认不同，与百度吻合
    ov7670_set_reg_value(0x4b, 0x09);//与默认不同，与百度吻合
    ov7670_set_reg_value(0xc9, 0xf0);//与默认不同，与百度吻合
    ov7670_set_reg_value(0x56, 0x60);//与默认不同，与百度吻合
                                    //相比原来有更改

    ov7670_set_reg_value(0x34, 0x11);//reserved
    ov7670_set_reg_value(0x3b, 0x0a);//与默认不同，与百度吻合
                                    //相比原来有更改
                                    // COM11 通用控制11
                        // [7] 夜晚模式   0 禁止     1 使能－帧率自动降低，最小侦率在COM11[6~5]中设定，ADVFH 和 ADVHL 自动增加
                        // [6~5] 夜晚模式的最小帧率
                        //       00 和普通模式一样
                        //       01 1/2普通模式
                        //       10 1/4普通模式
                        //       11 1/8普通模式

    ov7670_set_reg_value(0xa4, 0x89);//与默认不同，与百度吻合
    ov7670_set_reg_value(0x96, 0x00);//与默认不同，与百度吻合
    ov7670_set_reg_value(0x97, 0x30);//reserved
    ov7670_set_reg_value(0x98, 0x20);//reserved
    ov7670_set_reg_value(0x99, 0x30);//reserved
    ov7670_set_reg_value(0x9a, 0x84);//reserved
    ov7670_set_reg_value(0x9b, 0x29);//reserved
    ov7670_set_reg_value(0x9c, 0x03);//reserved
    ov7670_set_reg_value(0x9d, 0x4c);//与默认不同，与百度吻合
    ov7670_set_reg_value(0x9e, 0x3f);//与默认不同，与百度吻合
    ov7670_set_reg_value(0x78, 0x04);//reserved

}



                // 8'd1:  dout <= 16'h1204;
                // 8'd2:  dout <= 16'h40d0;
                // 8'd3:  dout <= 16'h3a04;
                // 8'd4:  dout <= 16'h3dc8;
                // 8'd5:  dout <= 16'h1e31;
                // 8'd6:  dout <= 16'h6b00;
                // 8'd7:  dout <= 16'h32b6;
                // 8'd8:  dout <= 16'h1713;
                // 8'd9:  dout <= 16'h1801;
                // 8'd10: dout <= 16'h1902;
                // 8'd11: dout <= 16'h1a7a;
                // 8'd12: dout <= 16'h030a;
                // 8'd13: dout <= 16'h0c00;
                // 8'd14: dout <= 16'h3e10;
                // 8'd15: dout <= 16'h7000;
                // 8'd16: dout <= 16'h7100;
                // 8'd17: dout <= 16'h7211;

                // 8'd18: dout <= 16'h7300;
                // 8'd19: dout <= 16'ha202;
                // 8'd20: dout <= 16'h1180;
                // 8'd21: dout <= 16'h7a20;
                // 8'd22: dout <= 16'h7b1c;
                // 8'd23: dout <= 16'h7c28;
                // 8'd24: dout <= 16'h7d3c;
                // 8'd25: dout <= 16'h7e55;
                // 8'd26: dout <= 16'h7f68;
                // 8'd27: dout <= 16'h8076;
                // 8'd28: dout <= 16'h8180;
                // 8'd29: dout <= 16'h8288;
                // 8'd30: dout <= 16'h838f;
                // 8'd31: dout <= 16'h8496;
                // 8'd32: dout <= 16'h85a3;
                // 8'd33: dout <= 16'h86af;

                // 8'd34: dout <= 16'h87c4;
                // 8'd35: dout <= 16'h88d7;

                // 8'd36: dout <= 16'h89e8;
                // 8'd37: dout <= 16'h13e0;
                // 8'd38: dout <= 16'h0010;
                // 8'd39: dout <= 16'h1000;
                // 8'd40: dout <= 16'h0d00;
                // 8'd41: dout <= 16'h1428;
                // 8'd42: dout <= 16'ha505;
                // 8'd43: dout <= 16'hab07;
                // 8'd44: dout <= 16'h2475;
                // 8'd45: dout <= 16'h2563;
                // 8'd46: dout <= 16'h26a5;
                // 8'd47: dout <= 16'h9f78;
                // 8'd48: dout <= 16'ha068;
                // 8'd49: dout <= 16'ha103;
                // 8'd50: dout <= 16'ha6df;
                // 8'd51: dout <= 16'ha7df;
                // 8'd52: dout <= 16'ha8f0;

                // 8'd53: dout <= 16'ha990;
                // 8'd54: dout <= 16'haa94;
                // 8'd55: dout <= 16'h13ef;
                // 8'd56: dout <= 16'h0e61;
                // 8'd57: dout <= 16'h0f4b;
                // 8'd58: dout <= 16'h1602;
                // 8'd59: dout <= 16'h2102;
                // 8'd60: dout <= 16'h2291;
                // 8'd61: dout <= 16'h2907;
                // 8'd62: dout <= 16'h330b;
                // 8'd63: dout <= 16'h350b;
                // 8'd64: dout <= 16'h371d;
                // 8'd65: dout <= 16'h3871;
                // 8'd66: dout <= 16'h392a;
                // 8'd67: dout <= 16'h3c78;
                // 8'd68: dout <= 16'h4d40;
                // 8'd69: dout <= 16'h4e20;
                // 8'd70: dout <= 16'h6900;
                // 8'd71: dout <= 16'h7419;
                // 8'd72: dout <= 16'h8d4f;
                // 8'd73: dout <= 16'h8e00;
                // 8'd74: dout <= 16'h8f00;
                // 8'd75: dout <= 16'h9000;
                // 8'd76: dout <= 16'h9100;
                // 8'd77: dout <= 16'h9200;
                // 8'd78: dout <= 16'h9600;
                // 8'd79: dout <= 16'h9a80;
                // 8'd80: dout <= 16'hb084;
                // 8'd81: dout <= 16'hb10c;
                // 8'd82: dout <= 16'hb20e;
                // 8'd83: dout <= 16'hb382;

                // 8'd84:  dout <= 16'hb80a;
                // 8'd85:  dout <= 16'h4314;
                // 8'd86:  dout <= 16'h44f0;
                // 8'd87:  dout <= 16'h4534;
                // 8'd88:  dout <= 16'h4658;
                // 8'd89:  dout <= 16'h4728;
                // 8'd90:  dout <= 16'h483a;
                // 8'd91:  dout <= 16'h5988;
                // 8'd92:  dout <= 16'h5a88;
                // 8'd93:  dout <= 16'h5b44;
                // 8'd94:  dout <= 16'h5c67;
                // 8'd95:  dout <= 16'h5d49;
                // 8'd96:  dout <= 16'h5e0e;
                // 8'd97:  dout <= 16'h6404;
                // 8'd98:  dout <= 16'h6520;
                // 8'd99:  dout <= 16'h6605;
                // 8'd100: dout <= 16'h9404;
                // 8'd101: dout <= 16'h9508;
                // 8'd102: dout <= 16'h6c0a;
                // 8'd103: dout <= 16'h6d55;
                // 8'd104: dout <= 16'h6e11;
                // 8'd105: dout <= 16'h6f9f;
                // 8'd106: dout <= 16'h6a40;
                // 8'd107: dout <= 16'h0140;
                // 8'd108: dout <= 16'h0240;
                // 8'd109: dout <= 16'h13e7;

                // 8'd110: dout <= 16'h1500;
                // 8'd111: dout <= 16'h4f80;
                // 8'd112: dout <= 16'h5080;
                // 8'd113: dout <= 16'h5100;
                // 8'd114: dout <= 16'h5222;
                // 8'd115: dout <= 16'h535e;
                // 8'd116: dout <= 16'h5480;

                // 8'd117: dout <= 16'h589e;
                // 8'd118: dout <= 16'h4108;
                // 8'd119: dout <= 16'h3f00;
                // 8'd120: dout <= 16'h7505;
                // 8'd121: dout <= 16'h76e1;
                // 8'd122: dout <= 16'h4c00;
                // 8'd123: dout <= 16'h7701;
                // 8'd124: dout <= 16'h4b09;
                // 8'd125: dout <= 16'hc9F0;
                // 8'd126: dout <= 16'h4138;
                // 8'd127: dout <= 16'h5640;

                // 8'd128: dout <= 16'h3411;
                // 8'd129: dout <= 16'h3b02;

                // 8'd130: dout <= 16'ha489;
                // 8'd131: dout <= 16'h9600;
                // 8'd132: dout <= 16'h9730;
                // 8'd133: dout <= 16'h9820;
                // 8'd134: dout <= 16'h9930;
                // 8'd135: dout <= 16'h9a84;
                // 8'd136: dout <= 16'h9b29;
                // 8'd137: dout <= 16'h9c03;
                // 8'd138: dout <= 16'h9d4c;
                // 8'd139: dout <= 16'h9e3f;
                // 8'd140: dout <= 16'h7804;

                // 8'd141: dout <= 16'h7901;
                // 8'd142: dout <= 16'hc8f0;
                // 8'd143: dout <= 16'h790f;
                // 8'd144: dout <= 16'hc800;
                // 8'd145: dout <= 16'h7910;
                // 8'd146: dout <= 16'hc87e;
                // 8'd147: dout <= 16'h790a;
                // 8'd148: dout <= 16'hc880;
                // 8'd149: dout <= 16'h790b;
                // 8'd150: dout <= 16'hc801;
                // 8'd151: dout <= 16'h790c;
                // 8'd152: dout <= 16'hc80f;
                // 8'd153: dout <= 16'h790d;
                // 8'd154: dout <= 16'hc820;
                // 8'd155: dout <= 16'h7909;
                // 8'd156: dout <= 16'hc880;
                // 8'd157: dout <= 16'h7902;
                // 8'd158: dout <= 16'hc8c0;
                // 8'd159: dout <= 16'h7903;
                // 8'd160: dout <= 16'hc840;
                // 8'd161: dout <= 16'h7905;
                // 8'd162: dout <= 16'hc830;
                // 8'd163: dout <= 16'h7926;
                // 8'd164: dout <= 16'h0903;
                // 8'd165: dout <= 16'h3b42;

// See LICENSE for license details.

#include "ov7670.h"

#include "hbird_sdk_hal.h"
#include "yolo_forward.c"


int *vga_bram = (int *)0x42000000;
int *vga_bram2 = (int *)0x42025800;
int *camera_bram = (int *)0x40000000;
int *camera_bram2 = (int *)0x40025800;

#define EN_VGA_OFS 0
#define PINGPONG_VGA_OFS 1
#define EN_CAMERA_OFS 2
#define INTR_CLR_CAMERA_OFS 3
#define PINGPONG_CAMERA_OFS 7
#define INTR_CAMERA_OFS 8


#define EN_VGA_MASK (1 << EN_VGA_OFS)
#define PINGPONG_VGA_MASK (1 << PINGPONG_VGA_OFS)
#define EN_CAMERA_MASK (1 << EN_CAMERA_OFS)
#define INTR_CLR_CAMERA_MASK (1 << INTR_CLR_CAMERA_OFS)
#define PINGPONG_CAMERA_MASK (1 << PINGPONG_CAMERA_OFS)
#define INTR_CAMERA_MASK (1 << INTR_CAMERA_OFS)
#define OUTPUT_MASK 0x0000007F  // gpioB[6:0]
#define INPUT_MASK 0x00003f80   // gpioB[13:7]

int pingpong_camera = 1;
int *dest_bram = NULL;
int intr_time = 0;

void intr_camera_handler(unsigned long mcause, unsigned long sp)
{
    intr_time++;
    pingpong_camera = gpio_read(GPIOB, PINGPONG_CAMERA_MASK)>>PINGPONG_CAMERA_OFS;
    gpio_clear_interrupt(GPIOB);
    dest_bram = vga_bram;
    printf("1\n");

    if(pingpong_camera){
        yolo_forward(18, 22, 16, 20, 32, 16, \
                     (short int *)camera_bram2, dest_bram);
    }
    else {
        yolo_forward(18, 22, 16, 20, 32, 16, \
                     (short int *)camera_bram, dest_bram);
    }

    gpio_write(GPIOB, INTR_CLR_CAMERA_MASK, 1);
    gpio_write(GPIOB, INTR_CLR_CAMERA_MASK, 0);
}


int* buffer = NULL;
int* mrom = (int*)0x00001000;

int main(void)
{
    camera_init();
    printf("??\n");
    gpio_enable_output(GPIOB, OUTPUT_MASK);
    gpio_enable_input(GPIOB, INPUT_MASK);
    gpio_enable_interrupt(GPIOB, INTR_CAMERA_MASK, GPIO_INT_RISE);
    int res = PLIC_Register_IRQ(PLIC_GPIOB_IRQn, 1, intr_camera_handler);
    __enable_irq();

    gpio_write(GPIOB, EN_VGA_MASK, 1);
    gpio_write(GPIOB, EN_CAMERA_MASK, 1);
    gpio_write(GPIOB, PINGPONG_VGA_MASK, 0);



    while(1);

    return 0;
}


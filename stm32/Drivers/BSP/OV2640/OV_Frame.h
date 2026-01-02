#ifndef _OV_Frame_H
#define _OV_Frame_H

#include "sccb.h"
#include "LCD.h"
#include "Dis_Picture.h"
#include "ov2640.h"
#include "dcmi.h"

extern uint8_t RGB_DATA[];


void jpeg_data_process(void);
void rgblcd_dcmi_rx_callback(void);
void RGB565_mode(void);
void GetImage(void);


#endif
























/************************************************************************************************/
//OV2640 驱动代码

//STM32H7工程模板-HAL库函数版本





/************************************************************************************************/














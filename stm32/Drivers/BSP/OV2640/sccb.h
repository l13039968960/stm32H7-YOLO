#ifndef __SCCB_H
#define __SCCB_H
#include "./System/sys/sys.h"
//////////////////////////////////////////////////////////////////////////////////	 
 

/************************************************************************************************/

//OV系列摄像头 SCCB 驱动代码	  

//STM32H7工程模板-HAL库函数版本





/************************************************************************************************/

////////////////////////////////////////////////////////////////////////////////// 	
//IO方向设置


//IO操作
#define SCCB_SCL(n)  (n?HAL_GPIO_WritePin(GPIOB,GPIO_PIN_10,GPIO_PIN_SET):HAL_GPIO_WritePin(GPIOB,GPIO_PIN_10,GPIO_PIN_RESET)) //SCL--PB10
#define SCCB_SDA(n)  (n?HAL_GPIO_WritePin(GPIOB,GPIO_PIN_11,GPIO_PIN_SET):HAL_GPIO_WritePin(GPIOB,GPIO_PIN_11,GPIO_PIN_RESET)) //SDA--PB11

#define SCCB_READ_SDA    HAL_GPIO_ReadPin(GPIOB,GPIO_PIN_11)     //输入SDA--PB11
#define SCCB_ID          0X60                                    //OVxxxx 的ID

///////////////////////////////////////////
void SCCB_Init(void);
	
void SCCB_SDA_IN(void);//IO方向设置
void SCCB_SDA_OUT(void);//IO方向设置
	
void SCCB_Start(void);
void SCCB_Stop(void);
void SCCB_No_Ack(void);
u8 SCCB_WR_Byte(u8 dat);
u8 SCCB_RD_Byte(void);
u8 SCCB_WR_Reg(u8 reg,u8 data);
u8 SCCB_RD_Reg(u8 reg);
#endif





































/************************************************************************************************/


//STM32H7工程模板-HAL库函数版本





/************************************************************************************************/


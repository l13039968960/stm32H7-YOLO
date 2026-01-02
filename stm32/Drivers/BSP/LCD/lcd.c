

#include "lcd.h"
#include "stdlib.h"
#include "font.h" 
#include "./SYSTEM/usart/usart.h"	 
#include "Text.h"
#include "Dis_Picture.h" 

#include "spi.h"

#define show_width 112
#define show_height 112

//////////////////////////////////////////////////////////////////////////////////	 

////2.0寸 240x320 高清屏  LCD显示 液晶驱动	  
  
//STM32F4工程模板-库函数版本




//********************************************************************************

//////////////////////////////////////////////////////////////////////////////////	 
				 
//LCD的画笔颜色和背景色	  

u16 POINT_COLOR=0x0000;	//画笔颜色
u16 BACK_COLOR=0xFFFF;  //背景色 



extern SPI_HandleTypeDef hspi2;


//管理LCD重要参数
//默认为竖屏
_lcd_dev lcddev;


//本驱动程序使用的是MCU硬件SPI接口驱动


/******************************************************************************

  接口定义，请根据接线修改并修改相应IO初始化--驱动板子和IO要对应

////  液晶屏 接口定义  使用SPI2驱动

//	#define LCD_SDI        	//PB15-MOSI   //数据输入线
//	#define LCD_SCL        	//PB13-SCK    //时钟线
//	#define LCD_CS        	//PB12-CS     //片选	
//	#define LCD_SDO     		//PB14-MISO   //数据输出/复位
//	#define LCD_DC         	//PB1--DC     //命令/数据切换
//	#define LCD_BLK         //PB0--BLK    //背光控制 
*******************************************************************************/


/******************************************************************************/

//函数： void LCD_GPIO_Init(void)

//函数功能：驱动液晶IO初始化配置






////  液晶屏 接口定义  使用SPI2驱动

//	#define LCD_SDI        	//PB15-MOSI   //数据输入线
//	#define LCD_SCL        	//PB13-SCK    //时钟线
//	#define LCD_CS        	//PB12-CS     //片选	
//	#define LCD_SDO     		//PB14-MISO   //数据输出/复位
//	#define LCD_DC         	//PB1--DC     //命令/数据切换
//	#define LCD_BLK         //PB0--BLK    //背光控制 	

/******************************************************************************/

extern SPI_HandleTypeDef hspi2;

void LCD_GPIO_Init(void)
{

    GPIO_InitTypeDef GPIO_Initure;

    __HAL_RCC_GPIOB_CLK_ENABLE();					          //开启PB时钟


    GPIO_Initure.Pin=GPIO_PIN_0 |GPIO_PIN_1 | GPIO_PIN_12;	//PB
    GPIO_Initure.Mode=GPIO_MODE_OUTPUT_PP;  		    //推挽输出
    GPIO_Initure.Pull=GPIO_PULLUP;         			    //上拉
    GPIO_Initure.Speed=GPIO_SPEED_FREQ_VERY_HIGH;  	//高速
    HAL_GPIO_Init(GPIOB,&GPIO_Initure);     	     	//初始化

    HAL_GPIO_WritePin(GPIOB,GPIO_PIN_0,GPIO_PIN_SET);	  //PB0置1 	
    HAL_GPIO_WritePin(GPIOB,GPIO_PIN_1,GPIO_PIN_SET);	  //PB1置1 
    HAL_GPIO_WritePin(GPIOB,GPIO_PIN_12,GPIO_PIN_SET);	//PB12置1 


    LCD_HardwareRest();   //硬复位--如果IO连接，硬件复位可控有效


    LCD_BLK_Off;   //关闭背光
    //	LCD_BLK_On;    //开启背光
      
}





/************************************************************************************************/
//SPI2 读写一个字节
//TxData:要写入的字节
//返回值:读取到的字节


//STM32H7工程模板-HAL库函数版本




/************************************************************************************************/

u8 LCD_SPI2_ReadWriteByte(u8 TxData)
{
    u8 Rxdata;
    HAL_SPI_Transmit(&hspi2,&TxData,1, 10);       
 	return Rxdata;          		    //返回收到的数据		
}



//********************************************************************/

//函数：void LCD_WR_REG(u16 regval)
//函数功能：
//写寄存器函数
//regval:寄存器值




//*******************************************************************/

void LCD_WR_REG(u16 regval)
{   
	 LCD_CS_CLR;  //LCD_CS=0  //片选
   LCD_DC_CLR;  //LCD_DC=0  //设置命令状态
	 LCD_SPI2_ReadWriteByte(regval&0x00FF);
	 LCD_CS_SET;  //LCD_CS=1	 
}



//*******************************************************************/

//函数：void LCD_WR_DATA8(u8 data)   //写8位数据
//函数功能：
//液晶屏--写8位数据函数




//*******************************************************************/

void LCD_WR_DATA8_2(u8 data)   //写8位数据
{
//	LCD_CS_CLR;  //LCD_CS=0  //片选
//	LCD_DC_SET;	 //LCD_DC=1   //设置数据状态			    	   
	LCD_SPI2_ReadWriteByte(data);	
//	LCD_CS_SET;  //LCD_CS=1   			 
}


void LCD_WR_DATA8(u8 data)   //写8位数据
{
	LCD_CS_CLR;  //LCD_CS=0  //片选
	LCD_DC_SET;	 //LCD_DC=1   //设置数据状态			    	   
	LCD_SPI2_ReadWriteByte(data);	
	LCD_CS_SET;  //LCD_CS=1   			 
}


//*******************************************************************/
//函数：void LCD_WR_DATA16(u16 data)   //写16位数据
//函数功能：写LCD数据
//输入参数：
//data:要写入的值




//*******************************************************************/


void LCD_WR_DATA16(u16 data)
{	
	
 	LCD_CS_CLR;  //LCD_CS=0  //片选
	LCD_DC_SET;	 //LCD_DC=1   //设置数据状态
	LCD_SPI2_ReadWriteByte(data>>8);
	LCD_SPI2_ReadWriteByte(data);
	LCD_CS_SET;    //LCD_CS=1
	
}


//*******************************************************************/

//函数：void LCD_WriteReg(u16 LCD_Reg, u16 LCD_RegValue)  //写液晶屏寄存器

//函数功能：写寄存器

//输入参数: 
//LCD_Reg:寄存器地址
//LCD_RegValue:要写入的数据




//*******************************************************************/

void LCD_WriteReg(u16 LCD_Reg, u16 LCD_RegValue)
{	
	LCD_WR_REG(LCD_Reg);		   //写入要写的寄存器序号	 
	LCD_WR_DATA8(LCD_RegValue);//写入数据	    		 
}


//*******************************************************************/
//函数：void LCD_WriteRAM_Prepare(void)   
//函数功能：开始写GRAM  命令




//*******************************************************************/

void LCD_WriteRAM_Prepare(void)
{
 	LCD_WR_REG(lcddev.wramcmd);	  
}	



//*******************************************************************/
//函数：LCD_WriteRAM(u16 RGB_Code)   
//函数功能：写入点阵颜色值
//输入参数:
//RGB_Code:颜色值




//*******************************************************************/

void LCD_WriteRAM(u16 RGB_Code)
{							    
	LCD_WR_DATA16( RGB_Code );//写十六位GRAM
}





//********************************************************************************/
//函数：void LCD_DisplayOn(void)
//函数功能：
//LCD开启显示




//*******************************************************************/
void LCD_DisplayOn(void)
{					   
	LCD_WR_REG(0X29);	//开启显示
}	


//*******************************************************************/
//函数：void LCD_DisplayOff(void)
//函数功能：
//LCD关闭显示




//*******************************************************************/

void LCD_DisplayOff(void)
{	   
	LCD_WR_REG(0X28);	//关闭显示

} 

//********************************************************************************/
//函数：void LCD_SoftRest(void)
//函数功能：给屏幕发命令，执行软复位命令
//LCD开启显示




//*******************************************************************/
void LCD_SoftRest(void)
{					   
	LCD_WR_REG(0x01);	//发送软复位命令
	LCD_Delay_ms(50);      // delay 50 ms 
}	


//********************************************************************************/
//函数：void LCD_SoftRest(void)
//函数功能：给屏幕发命令，执行软复位命令
//LCD开启显示




//*******************************************************************/
void LCD_HardwareRest(void)
{					   
//	LCD_RST_Clr;     //液晶屏复位 --硬复位--使能       //PB14作为液晶屏复位控制引脚
	LCD_Delay_ms(50);      // delay 50 ms 
//	LCD_RST_Set;      //液晶屏复位 --硬复位--失能       //PB14作为液晶屏复位控制引脚
	LCD_Delay_ms(30);      // delay 30 ms 
}	



//*******************************************************************/
//函数：void LCD_SetCursor(u16 Xpos, u16 Ypos)
//函数功能：设置光标位置
//输入参数：
//Xpos:横坐标
//Ypos:纵坐标




//*******************************************************************/
void LCD_SetCursor(u16 Xpos, u16 Ypos)
{	 
//		if(LCD_DIR_Mode==2)Ypos=Ypos;
//	  if(LCD_DIR_Mode==3)Xpos=Xpos;
			
		LCD_WR_REG(lcddev.setxcmd); 
	  LCD_WR_DATA16(Xpos);
	
		LCD_WR_REG(lcddev.setycmd);
    LCD_WR_DATA16(Ypos);	

	
}





//*******************************************************************/
//函数：void LCD_DrawPoint(u16 x,u16 y)
//函数功能：画点
//输入参数：
//x,y:坐标
//POINT_COLOR:此点的颜色




//*******************************************************************/

void LCD_DrawPoint(u16 x,u16 y)
{
	LCD_SetCursor(x,y);		//设置光标位置 
	LCD_WriteRAM_Prepare();	//开始写入GRAM
	LCD_WR_DATA16(POINT_COLOR); 
}


//*******************************************************************/
//函数：void LCD_Fast_DrawPoint(u16 x,u16 y,u16 color)
//函数功能：快速画点
//输入参数：
//x,y:坐标
//color:颜色




//*******************************************************************/

void LCD_Fast_DrawPoint(u16 x,u16 y,u16 color)
{	
	
//    if(LCD_DIR_Mode==2)y=y; 
//	  if(LCD_DIR_Mode==3)x=x; 
	
		LCD_WR_REG(lcddev.setxcmd); 
	  LCD_WR_DATA16(x);

  	LCD_WR_REG(lcddev.setycmd); 
	  LCD_WR_DATA16(y);
			 
	  LCD_WR_REG(lcddev.wramcmd); 
	  LCD_WR_DATA16(color); //写入16位颜色
}	 



//*******************************************************************/
//函数：void LCD_Scan_Dir(u8 dir)
//函数功能：设置LCD的自动扫描方向
//输入参数：

//默认设置为L2R_U2D,如果设置为其他扫描方式,可能导致显示不正常.
//dir:0~7,代表8个方向




//*******************************************************************/

void LCD_Scan_Dir(u8 dir)
{
	u8 regval=0;

	//扫描方向定义--扫描方式有不同规格，可能定义不左右和上下的参照方向不同，总结方式，只有一下八种
	
	
	switch(dir)
		{
			case 0:
				regval|=(0<<7)|(0<<6)|(0<<5); 
				break;
			case 1:
				regval|=(0<<7)|(1<<6)|(1<<5); 
				break;
			case 2:
				regval|=(1<<7)|(1<<6)|(0<<5); 
				break;
			case 3:
				regval|=(1<<7)|(0<<6)|(1<<5); 
				break;	 
	 
		}
		
    		
		LCD_WriteReg(0x36,regval);//改变扫描方向命令  ---此处需要查看数据手册，确定RGB颜色交换位的配置


} 




/**************************************************************************/
//函数：void LCD_Display_Dir(u8 dir)
//函数功能：设置LCD的显示方向及像素参数

//输入参数：

//设置LCD显示方向

////dir:   0,竖屏  正
//         1,竖屏  反
//         2,横屏  左
//         3,横屏  右




//*************************************************************************/
void LCD_Display_Dir(u8 dir)
{
	
	u8 SCAN_DIR;
		
	if(dir==0)			     //竖屏  正
	{
		lcddev.dir=0;	     //竖屏
		lcddev.width=show_width;
		lcddev.height=show_height;
    
		lcddev.wramcmd=0X2C;
	 	lcddev.setxcmd=0X2A;
		lcddev.setycmd=0X2B;

    SCAN_DIR=0; //选择扫描方向		
		

	}
	
else if (dir==1)			 //横屏
	{	  				
		lcddev.dir=0;	     //横屏
		lcddev.width=show_height;
		lcddev.height=show_width;
    
		lcddev.wramcmd=0X2C;
	 	lcddev.setxcmd=0X2A;
		lcddev.setycmd=0X2B;
		
    SCAN_DIR=1; //选择扫描方向		

		
	} 	
	
	
	else if (dir==2)			//竖屏  
	{	  				
		lcddev.dir=1;	     //竖屏  
		lcddev.width=show_width;
		lcddev.height=show_height;

		lcddev.wramcmd=0X2C;
	 	lcddev.setxcmd=0X2A;
		lcddev.setycmd=0X2B;  
		
    SCAN_DIR=2; //选择扫描方向		
	
		
	} 
 else if (dir==3)				  //横屏
	{	  				
		lcddev.dir=1;	        //横屏
		lcddev.width=show_height;
		lcddev.height=show_width;

		lcddev.wramcmd=0X2C;
	 	lcddev.setxcmd=0X2A;
		lcddev.setycmd=0X2B; 
    
    SCAN_DIR=3; //选择扫描方向

	

	} 	
 else //设置默认为竖屏--正
 {
	  lcddev.dir=0;	     //竖屏
		lcddev.width=show_width;
		lcddev.height=show_height;
    
		lcddev.wramcmd=0X2C;
	 	lcddev.setxcmd=0X2A;
		lcddev.setycmd=0X2B;

    SCAN_DIR=0; //选择扫描方向		
	 

 }	 

 
 //////以下设置，为窗口参数设置，设置了全屏的显示范围			

    LCD_Set_Window(0,0,lcddev.width,lcddev.height);//设置全屏窗口	

 /////设置屏幕显示--扫描方向
	
	  LCD_Scan_Dir(SCAN_DIR);	//设置屏幕显示--扫描方向
	
     
}	


/**************************************************************************/
//函数：void LCD_Set_Window(u16 sx,u16 sy,u16 width,u16 height)
//函数功能：设置LCD的显示窗口

//设置窗口,并自动设置画点坐标到窗口左上角(sx,sy).
//sx,sy:窗口起始坐标(左上角)
//width,height:窗口宽度和高度,必须大于0!!
//窗体大小:width*height.




//*************************************************************************/

void LCD_Set_Window(u16 sx,u16 sy,u16 width,u16 height)
{   
	
	
	  width=sx+width-1;
	  height=sy+height-1;

	if(LCD_DIR_Mode==0)
		{	
			LCD_WR_REG(lcddev.setxcmd); 
			LCD_WR_DATA16(sx);      //设置 X方向起点
			LCD_WR_DATA16(width);   //设置 X方向终点	
		
			LCD_WR_REG(lcddev.setycmd);
			LCD_WR_DATA16(sy);      //设置 Y方向起点
			LCD_WR_DATA16(height);  //设置 Y方向终点
		}
	else if(LCD_DIR_Mode==1)
		{	
			LCD_WR_REG(lcddev.setxcmd); 
			LCD_WR_DATA16(sx);      //设置 X方向起点
			LCD_WR_DATA16(width);   //设置 X方向终点	
		
			LCD_WR_REG(lcddev.setycmd);
			LCD_WR_DATA16(sy);      //设置 Y方向起点
			LCD_WR_DATA16(height);  //设置 Y方向终点
		}	
		else if(LCD_DIR_Mode==2)
		{	
			LCD_WR_REG(lcddev.setxcmd); 
			LCD_WR_DATA16(sx);      //设置 X方向起点
			LCD_WR_DATA16(width);   //设置 X方向终点	
		
			LCD_WR_REG(lcddev.setycmd);
			LCD_WR_DATA16(sy);      //设置 Y方向起点
			LCD_WR_DATA16(height);  //设置 Y方向终点
		}		
	 else if(LCD_DIR_Mode==3)
		{	
			LCD_WR_REG(lcddev.setxcmd); 
			LCD_WR_DATA16(sx);      //设置 X方向起点
			LCD_WR_DATA16(width);   //设置 X方向终点	
		
			LCD_WR_REG(lcddev.setycmd);
			LCD_WR_DATA16(sy);      //设置 Y方向起点
			LCD_WR_DATA16(height);  //设置 Y方向终点
		}
} 


/*******************************************************************************/
//函数：void LCD_Init(void)
//函数功能：初始化lcd
//该初始化函数可以初始化ST7789V




/*******************************************************************************/

//void LCD_Init(void)
//{ 	

//	 
//   	LCD_GPIO_Init();        //初始化驱动 I/O接口	
//	
//	  LCD_SoftRest();        //软复位 
//	
//	  LCD_WR_REG(0x11);      //Sleep exit 
// 	  LCD_Delay_ms(120);      // delay 120 ms 


////ST7735R Frame Rate
//	  LCD_WR_REG(0xB1); 
//		LCD_WR_DATA8(0x05); 
//		LCD_WR_DATA8(0x3C); 
//		LCD_WR_DATA8(0x3C); 
//		LCD_WR_REG(0xB2); 
//		LCD_WR_DATA8(0x05); 
//		LCD_WR_DATA8(0x3C); 
//		LCD_WR_DATA8(0x3C); 
//		LCD_WR_REG(0xB3); 
//		LCD_WR_DATA8(0x05); 
//		LCD_WR_DATA8(0x3C); 
//		LCD_WR_DATA8(0x3C); 
//		LCD_WR_DATA8(0x05); 
//		LCD_WR_DATA8(0x3C); 
//		LCD_WR_DATA8(0x3C); 
//		//------------------------------------End ST7735S Frame Rate---------------------------------// 
//		LCD_WR_REG(0xB4); //Dot inversion 
//		LCD_WR_DATA8(0x03); 
//		//------------------------------------ST7735S Power Sequence---------------------------------// 
//		LCD_WR_REG(0xC0); 
//		LCD_WR_DATA8(0x28); 
//		LCD_WR_DATA8(0x08); 
//		LCD_WR_DATA8(0x04); 
//		LCD_WR_REG(0xC1); 
//		LCD_WR_DATA8(0XC0); 
//		LCD_WR_REG(0xC2); 
//		LCD_WR_DATA8(0x0D); 
//		LCD_WR_DATA8(0x00); 
//		LCD_WR_REG(0xC3); 
//		LCD_WR_DATA8(0x8D); 
//		LCD_WR_DATA8(0x2A); 
//		LCD_WR_REG(0xC4); 
//		LCD_WR_DATA8(0x8D); 
//		LCD_WR_DATA8(0xEE); 
//		//---------------------------------End ST7735S Power Sequence-------------------------------------// 
//		LCD_WR_REG(0xC5); //VCOM 
//		LCD_WR_DATA8(0x1A); 
//		LCD_WR_REG(0x36); //MX, MY, RGB mode 
//		LCD_WR_DATA8(0xC0); 
//		//------------------------------------ST7735S Gamma Sequence---------------------------------// 
//		LCD_WR_REG(0xE0); 
//		LCD_WR_DATA8(0x04); 
//		LCD_WR_DATA8(0x22); 
//		LCD_WR_DATA8(0x07); 
//		LCD_WR_DATA8(0x0A); 
//		LCD_WR_DATA8(0x2E); 
//		LCD_WR_DATA8(0x30); 
//		LCD_WR_DATA8(0x25); 
//		LCD_WR_DATA8(0x2A); 
//		LCD_WR_DATA8(0x28); 
//		LCD_WR_DATA8(0x26); 
//		LCD_WR_DATA8(0x2E); 
//		LCD_WR_DATA8(0x3A); 
//		LCD_WR_DATA8(0x00); 
//		LCD_WR_DATA8(0x01); 
//		LCD_WR_DATA8(0x03); 
//		LCD_WR_DATA8(0x13); 
//		LCD_WR_REG(0xE1); 
//		LCD_WR_DATA8(0x04); 
//		LCD_WR_DATA8(0x16); 
//		LCD_WR_DATA8(0x06); 
//		LCD_WR_DATA8(0x0D); 
//		LCD_WR_DATA8(0x2D); 
//		LCD_WR_DATA8(0x26); 
//		LCD_WR_DATA8(0x23); 
//		LCD_WR_DATA8(0x27); 
//		LCD_WR_DATA8(0x27); 
//		LCD_WR_DATA8(0x25); 
//		LCD_WR_DATA8(0x2D); 
//		LCD_WR_DATA8(0x3B); 
//		LCD_WR_DATA8(0x00); 
//		LCD_WR_DATA8(0x01); 
//		LCD_WR_DATA8(0x04); 
//		LCD_WR_DATA8(0x13); 
//		//------------------------------------End ST7735S Gamma Sequence-----------------------------// 
//		LCD_WR_REG(0x3A); //65k mode 
//		LCD_WR_DATA8(0x05); 
//		LCD_WR_REG(0x29); //Display on 

//	  LCD_Display_Dir(LCD_DIR_Mode);	//选择--屏幕显示方式
//		
//	  LCD_BLK_On;					//点亮背光
//		
//	  LCD_Clear(WHITE);
//	
//	
//	

//}
void LCD_Init(void)
{ 	

    LCD_GPIO_Init();        //初始化驱动 I/O接口	

    LCD_SoftRest();        //软复位 

    LCD_Delay_ms(50);      // delay 50 ms 

    //************* Start Initial Sequence **********// 

    LCD_WR_REG(0x36);//内存数据访问控制
    LCD_WR_DATA8(0x00);


    LCD_WR_REG(0x3A);//接口像素格式
    LCD_WR_DATA8(0x05);

    LCD_WR_REG(0xB2);     //门廊设置
    LCD_WR_DATA8(0x0C);
    LCD_WR_DATA8(0x0C);
    LCD_WR_DATA8(0x00);
    LCD_WR_DATA8(0x33);
    LCD_WR_DATA8(0x33); 

    LCD_WR_REG(0xB7);    //门控制
    LCD_WR_DATA8(0x35);  

    LCD_WR_REG(0xBB);    //VCOM Setting
    LCD_WR_DATA8(0x19);

    LCD_WR_REG(0xC0);    //LCM Control
    LCD_WR_DATA8(0x2C);

    LCD_WR_REG(0xC2);   //VDV and VRH Command Enable
    LCD_WR_DATA8(0x01);

    LCD_WR_REG(0xC3);  //VRH Set
    LCD_WR_DATA8(0x12);   

    LCD_WR_REG(0xC4);  //VDV Set
    LCD_WR_DATA8(0x20);  

    LCD_WR_REG(0xC6);   //正常模式下的帧率控制
    LCD_WR_DATA8(0x0F); //帧频率60帧   

    LCD_WR_REG(0xD0);   //Power Control
    LCD_WR_DATA8(0xA4);
    LCD_WR_DATA8(0xA1);

    LCD_WR_REG(0xE0);    //正电压伽玛控制
    LCD_WR_DATA8(0xD0);
    LCD_WR_DATA8(0x04);
    LCD_WR_DATA8(0x0D);
    LCD_WR_DATA8(0x11);
    LCD_WR_DATA8(0x13);
    LCD_WR_DATA8(0x2B);
    LCD_WR_DATA8(0x3F);
    LCD_WR_DATA8(0x54);
    LCD_WR_DATA8(0x4C);
    LCD_WR_DATA8(0x18);
    LCD_WR_DATA8(0x0D);
    LCD_WR_DATA8(0x0B);
    LCD_WR_DATA8(0x1F);
    LCD_WR_DATA8(0x23);

    LCD_WR_REG(0xE1);   //负电压伽玛控制
    LCD_WR_DATA8(0xD0);
    LCD_WR_DATA8(0x04);
    LCD_WR_DATA8(0x0C);
    LCD_WR_DATA8(0x11);
    LCD_WR_DATA8(0x13);
    LCD_WR_DATA8(0x2C);
    LCD_WR_DATA8(0x3F);
    LCD_WR_DATA8(0x44);
    LCD_WR_DATA8(0x51);
    LCD_WR_DATA8(0x2F);
    LCD_WR_DATA8(0x1F);
    LCD_WR_DATA8(0x1F);
    LCD_WR_DATA8(0x20);
    LCD_WR_DATA8(0x23);

    LCD_WR_REG(0x21); //显示翻转开

    LCD_WR_REG(0x11); //退出休眠

    delay_ms (120); 

    LCD_WR_REG(0x29); //开显示

    delay_ms (200); 

    LCD_Display_Dir(LCD_DIR_Mode);  //选择--屏幕显示方式

    LCD_Clear(BLACK);
    LCD_BLK_On;	      //点亮背光

}


//////////////////以下函数是屏幕显示图形的简单驱动函数////////////////////////////////////////////////



/*******************************************************************************/
//函数：void LCD_Clear(u16 color)
//函数功能：全屏清屏填充函数
//输入参数：
//color:要清屏的填充色




/*******************************************************************************/

void LCD_Clear(u16 color)
{
	u32 index=0;
	u32 totalpoint;
	
	LCD_Set_Window(0,0,lcddev.width,lcddev.height);//设置全屏窗口
	
	totalpoint=lcddev.width * lcddev.height; 			//得到总点数
 
	LCD_SetCursor(0x00,0x00);	//设置光标位置 
	
	LCD_WriteRAM_Prepare();     		//开始写入GRAM	
	
	for(index=0;index<totalpoint;index++)
	{
		LCD_WR_DATA16(color);	
	}
} 


/*******************************************************************************/
//函数：void LCD_Fill(u16 sx,u16 sy,u16 ex,u16 ey,u16 color)

//函数功能：区域填充函数--填充单个颜色
//输入参数：
//在指定区域内填充单个颜色
//(sx,sy),(ex,ey):填充矩形对角坐标,区域大小为:(ex-sx+1)*(ey-sy+1)   
//color:要填充的颜色




/*******************************************************************************/

void LCD_Fill(u16 sx,u16 sy,u16 ex,u16 ey,u16 color)
{          
		u32 i;
		u32 xlen=0;

 //设置窗口
    LCD_Set_Window(sx,sy,ex-sx+1,ey-sy+1);//设置窗口
		  	
	  LCD_WR_REG(lcddev.wramcmd);	  
	
 		xlen=(ex-sx+1)*(ey-sy+1);//计算出总共需要写入的点数
		
		LCD_WriteRAM_Prepare();     			//开始写入GRAM
		
		for(i=0;i<=xlen;i++)
		{
		 	LCD_WR_DATA16(color);	//显示颜色 	    
		}


//恢复全屏窗口
		
//以下设置，为窗口参数设置，设置了全屏的显示范围			
			
   LCD_Set_Window(0,0,lcddev.width,lcddev.height);//设置全屏窗口	
		 
}


/*******************************************************************************/
//函数：void LCD_Color_Fill(u16 sx,u16 sy,u16 ex,u16 ey,u16 *color)

//函数功能：区域填充函数
//输入参数：
//在指定区域内填充指定颜色块			 
//(sx,sy),(ex,ey):填充矩形对角坐标,区域大小为:(ex-sx+1)*(ey-sy+1)   
//color:要填充的颜色




/*******************************************************************************/

void LCD_Color_Fill(u16 sx,u16 sy,u16 ex,u16 ey,u16 *color)
{  

		u32 i;
		u32 xlen=0;

	 //设置窗口
	
	  LCD_Set_Window(sx,sy,ex-sx+1,ey-sy+1);//设置窗口
  	
	  LCD_WR_REG(lcddev.wramcmd);	  
	
 		xlen=(ex-sx+1)*(ey-sy+1);//计算出总共需要写入的点数
		
		LCD_WriteRAM_Prepare();     			//开始写入GRAM
		
		for(i=0;i<=xlen;i++)
		{
		 	LCD_WR_DATA16(*color);	//显示颜色 	    
		}


//恢复全屏窗口
		
//以下设置，为窗口参数设置，设置了全屏的显示范围			
			
  LCD_Set_Window(0,0,lcddev.width,lcddev.height);//设置全屏窗口	
		

}



/*******************************************************************************/
//函数：void LCD_DrawLine(u16 x1, u16 y1, u16 x2, u16 y2)
//函数功能：画线
//输入参数：
//x1,y1:起点坐标
//x2,y2:终点坐标
//Color;线条颜色




/*******************************************************************************/

void LCD_DrawLine(u16 x1, u16 y1, u16 x2, u16 y2, u16 Color)
{
	u16 t; 
	int xerr=0,yerr=0,delta_x,delta_y,distance; 
	int incx,incy,uRow,uCol; 
	
	delta_x=x2-x1; //计算坐标增量 
	delta_y=y2-y1; 
	uRow=x1; 
	uCol=y1; 
	
	
	if(delta_x>0)incx=1; //设置单步方向 
	else if(delta_x==0)incx=0;//垂直线 
	else
     {
		    incx=-1;
		    delta_x=-delta_x;
		 }

	
	if(delta_y>0)incy=1; 
	else if(delta_y==0)incy=0;//水平线 
	else
     { 
		   incy=-1;
		   delta_y=-delta_y;
		 }

	
	if( delta_x>delta_y)distance=delta_x; //选取基本增量坐标轴 
	else distance=delta_y; 
	
	for(t=0;t<=distance+1;t++ )//画线输出 
	{  
		LCD_Fast_DrawPoint(uRow,uCol,Color);//画点 --使用输入颜色参数 
		xerr+=delta_x ; 
		yerr+=delta_y ; 
		if(xerr>distance) 
		{ 
			xerr-=distance; 
			uRow+=incx; 
		} 
		if(yerr>distance) 
		{ 
			yerr-=distance; 
			uCol+=incy; 
		} 
	}  
} 

/*******************************************************************************/
//函数：void LCD_DrawLine(u16 x1, u16 y1, u16 x2, u16 y2)
//函数功能：画矩形	  
//输入参数：
//(x1,y1),(x2,y2):矩形的对角坐标
//Color;线条颜色




/*******************************************************************************/

void LCD_DrawRectangle(u16 x1, u16 y1, u16 x2, u16 y2, u16 Color)
{
	LCD_DrawLine(x1,y1,x2,y1,Color);
	LCD_DrawLine(x1,y1,x1,y2,Color);
	LCD_DrawLine(x1,y2,x2,y2,Color);
	LCD_DrawLine(x2,y1,x2,y2,Color);
}

/*******************************************************************************/
//函数：void LCD_Draw_Circle(u16 x0,u16 y0,u8 r)
//函数功能：在指定位置画一个指定大小的圆
//输入参数：
//(x,y):中心点
//r    :半径
//Color;线条颜色




/*******************************************************************************/

void LCD_Draw_Circle(u16 x0,u16 y0,u8 r, u16 Color)
{
	int a,b;
	int di;
	a=0;b=r;	  
	di=3-(r<<1);             //判断下个点位置的标志
	while(a<=b)
	{
		LCD_Fast_DrawPoint(x0+a,y0-b,Color);             //5
 		LCD_Fast_DrawPoint(x0+b,y0-a,Color);             //0           
		LCD_Fast_DrawPoint(x0+b,y0+a,Color);             //4               
		LCD_Fast_DrawPoint(x0+a,y0+b,Color);             //6 
		LCD_Fast_DrawPoint(x0-a,y0+b,Color);             //1       
 		LCD_Fast_DrawPoint(x0-b,y0+a,Color);             
		LCD_Fast_DrawPoint(x0-a,y0-b,Color);             //2             
  	LCD_Fast_DrawPoint(x0-b,y0-a,Color);             //7     	         
		a++;
		//使用Bresenham算法画圆     
		if(di<0)di +=4*a+6;	  
		else
		{
			di+=10+4*(a-b);   
			b--;
		} 						    
	}
}


///*******************************************************************************/
////函数：LCD_ShowChar(u16 x,u16 y,u8 num,u8 size,u16 color,u8 mode)
////函数功能：在指定位置显示一个字符
////输入参数：
////x,y:起始坐标
////num:要显示的字符:" "--->"~"
////size:字体大小 12/16/24
////color,字符颜色
////mode:叠加方式(1)还是非叠加方式(0)




///*******************************************************************************/

//void LCD_ShowChar(u16 x,u16 y,u8 num,u8 size,u16 color,u8 mode)
//{  							  
//  u8 temp,t1,t;
//	u16 y0=y;
//	u8 csize=(size/8+((size%8)?1:0))*(size/2);		//得到字体一个字符对应点阵集所占的字节数	
//	//设置窗口		   
//	num=num-' ';//得到偏移后的值
//	for(t=0;t<csize;t++)
//	{   
//		if(size==12)temp=asc2_1206[num][t]; 	 	//调用1206字体
//		else if(size==16)temp=asc2_1608[num][t];	//调用1608字体
//		else if(size==24)temp=asc2_2412[num][t];	//调用2412字体
//		else return;								//没有的字库
//		for(t1=0;t1<8;t1++)
//		{			    
//			if(temp&0x80)LCD_Fast_DrawPoint(x,y,color);
//			else if(mode==0)LCD_Fast_DrawPoint(x,y,BACK_COLOR);
//			
//			temp<<=1;
//			y++;
//			if(y>=lcddev.height)return;		//超区域了
//			if((y-y0)==size)
//			{
//				y=y0;
//				x++;
//				if(x>=lcddev.width)return;	//超区域了
//				break;
//			}
//		}  	 
//	}  	    	   	 	  
//}



///*******************************************************************************/
////函数：u32 LCD_Pow(u8 m,u8 n)
////函数功能：m^n函数
////输入参数：两个8位数据
////返回值:m^n次方.




///*******************************************************************************/
//u32 LCD_Pow(u8 m,u8 n)
//{
//	u32 result=1;	 
//	while(n--)result*=m;    
//	return result;
//}	


///*******************************************************************************/
////函数：void LCD_ShowNum(u16 x,u16 y,u32 num,u8 len,u8 size,u16 color)
////函数功能：显示数字,高位为0,则不显示
////输入参数：

////x,y :起点坐标	 
////num:数值(0~4294967295);	
////len :数字的位数
////size:字体大小
////color:颜色 


////返回值:无




///*******************************************************************************/

//void LCD_ShowNum(u16 x,u16 y,u32 num,u8 len,u8 size,u16 color)
//{         	
//	u8 t,temp;
//	u8 enshow=0;						   
//	for(t=0;t<len;t++)
//	{
//		temp=(num/LCD_Pow(10,len-t-1))%10;
//		if(enshow==0&&t<(len-1))
//		{
//			if(temp==0)
//			{
//				LCD_ShowChar(x+(size/2)*t,y,' ',size,color,0);
//				continue;
//			}else enshow=1; 
//		 	 
//		}
//	 	LCD_ShowChar(x+(size/2)*t,y,temp+'0',size,color,0); 
//	}
//} 



///*******************************************************************************/
////函数：void LCD_ShowxNum(u16 x,u16 y,u32 num,u8 len,u8 size,u16 color,u8 mode)
////函数功能：显示数字,高位为0,还是显示
////输入参数：
////显示数字,高位为0,还是显示
////x,y:起点坐标
////num:数值(0~999999999);	 
////len:长度(即要显示的位数)
////size:字体大小
////color:颜色 
////mode:
////[7]:0,不填充;1,填充0.
////[6:1]:保留
////[0]:0,非叠加显示;1,叠加显示.




///*******************************************************************************/

//void LCD_ShowxNum(u16 x,u16 y,u32 num,u8 len,u8 size,u16 color,u8 mode)
//{  
//	u8 t,temp;
//	u8 enshow=0;						   
//	for(t=0;t<len;t++)
//	{
//		temp=(num/LCD_Pow(10,len-t-1))%10;
//		if(enshow==0&&t<(len-1))
//		{
//			if(temp==0)
//			{
//				if(mode&0X80)LCD_ShowChar(x+(size/2)*t,y,'0',size,color,mode&0X01);  
//				else LCD_ShowChar(x+(size/2)*t,y,' ',size,color,mode&0X01);  
// 				continue;
//			}else enshow=1; 
//		 	 
//		}
//	 	LCD_ShowChar(x+(size/2)*t,y,temp+'0',size,color,mode&0X01); 
//	}
//} 

///*******************************************************************************/
////函数：void LCD_ShowString(u16 x,u16 y,u16 width,u16 height,u8 size,u16 color,u8 *p)
////函数功能：显示字符串
////输入参数：
////x,y:起点坐标
////width,height:区域大小  
////size:字体大小
////color:颜色 
////*p:字符串起始地址		  




///*******************************************************************************/


//void LCD_ShowString(u16 x,u16 y,u16 width,u16 height,u8 size,u16 color,u8 *p)
//{         
//	u8 x0=x;
//	width+=x;
//	height+=y;
//    while((*p<='~')&&(*p>=' '))//判断是不是非法字符!
//    {       
//        if(x>=width){x=x0;y+=size;}
//        if(y>=height)break;//退出
//        LCD_ShowChar(x,y,*p,size,color,0);
//        x+=size/2;
//        p++;
//    }  
//}







//****************************************************************************************/
//函数：void DisplayButtonDown(u16 x1,u16 y1,u16 x2,u16 y2)
//功能描述: 在屏幕显示一凸起的按钮框
//输    入: u16 x1,y1,x2,y2 按钮框左上角和右下角坐标
//输    出: 无





/****************************************************************************************/

void DisplayButtonDown(u16 x1,u16 y1,u16 x2,u16 y2)
{
	
	LCD_DrawLine(x1,y1, x2,y1,GRAY2);  //H
		
	LCD_DrawLine(x1+1,y1+1,x2,y1+1,GRAY1);  //H
		
	LCD_DrawLine(x1,  y1,  x1,y2,GRAY2);    //V
		
	LCD_DrawLine(x1+1,y1+1,x1+1,y2,GRAY1);  //V
	
	LCD_DrawLine(x1, y2, x2, y2,WHITE);     //H
	LCD_DrawLine(x2, y1, x2, y2,WHITE);     //V
}

//****************************************************************************************/
//函数：void DisplayButtonUp(u16 x1,u16 y1,u16 x2,u16 y2)
//功能描述: 在屏幕显示一凹下的按钮框
//输    入: u16 x1,y1,x2,y2 按钮框左上角和右下角坐标
//输    出: 无



/****************************************************************************************/
void DisplayButtonUp(u16 x1,u16 y1,u16 x2,u16 y2)
{
	
	LCD_DrawLine(x1,  y1,  x2,y1,WHITE);    //H
	LCD_DrawLine(x1,  y1,  x1,y2,WHITE);    //V
	
	LCD_DrawLine(x1+1,y2-1,x2,y2-1,GRAY1);  //H
		
	LCD_DrawLine(x1,  y2,  x2,y2,GRAY2);    //H
		
	LCD_DrawLine(x2-1,y1+1,x2-1,y2,GRAY1);  //V
		
  LCD_DrawLine(x2  ,y1  ,x2,y2,GRAY2);    //V
	

}


//****************************************************************************************/
//函数：void Draw_Test(void)
//功能描述: 绘制图形函数测试





/****************************************************************************************/

void Draw_Test(void)
{
	
		LCD_Clear(WHITE); //清屏
		
	  LCD_DrawLine(20,64, 220,128,RED);//划线函数
	  LCD_DrawLine(20,128, 220,64,RED);//划线函数
	
	  LCD_DrawRectangle(20,64,220,128, BLUE);//绘制方形状
	  
	  LCD_Draw_Circle(120,96,81, BRED);//绘制圆形
	  LCD_Draw_Circle(120,96,80, BRED);//绘制圆形
	  LCD_Draw_Circle(120,96,79, BRED);//绘制圆形
			
		delay_ms(2000);//延时

}


//****************************************************************************************/
//函数：void Color_Test(void)
//功能描述: 颜色填充显示测试





/****************************************************************************************/

void Color_Test(void)
{
	
		LCD_Clear(GRAY0); //清屏
		
//		Draw_Font16B(24,16,BLUE,"1: 颜色填充测试");
	
	  LCD_Fill(5,5,lcddev.width-5,lcddev.height-5,RED);//设置一个窗口，写入指定区域颜色
	
	  LCD_Fill(20,20,lcddev.width-20,lcddev.height-20,YELLOW);//设置一个窗口，写入指定区域颜色
	
	  LCD_Fill(50,50,lcddev.width-50,lcddev.height-50,BLUE);//设置一个窗口，写入指定区域颜色
	
	  LCD_Fill(80,100,lcddev.width-80,lcddev.height-80,MAGENTA);//设置一个窗口，写入指定区域颜色
	
		delay_ms(2000);

		LCD_Clear(WHITE);
	
	  Draw_Test();//绘图函数测试
	
		delay_ms(300);//延时
		LCD_Clear(BLACK);
		delay_ms(300);//延时
		LCD_Clear(RED);
		delay_ms(300);//延时
		LCD_Clear(YELLOW);
		delay_ms(300);//延时
		LCD_Clear(GREEN);
		delay_ms(300);//延时
		LCD_Clear(BLUE);
		delay_ms(300);//延时

}


uint16_t D_Color=BLUE; //点阵颜色
uint16_t B_Color=WHITE; //背景颜色

//****************************************************************************************/
//函数：void Font_Test(void)
//功能描述: 字体字形显示测试





/****************************************************************************************/

//void Font_Test(void)
//{
//	LCD_Clear(GRAY0);
//	Draw_Font16B(24,16,BLUE,"2: 文字显示测试");

//	delay_ms(1000);
//	LCD_Clear(GRAY0);
//	
//	Draw_Font24B(48,32,BLACK,"液晶屏测试");
//		Draw_Font16B(24,96, GREEN,"中文测试");
//	Draw_Font16B(24,112,BLUE,"专注液晶批发");
//	Draw_Font16B(24,128,RED, "全程技术支持");
//	Draw_Font16B(16,176,RED,"0123456789 ABCDEFGHIJKLMNOPQRSTUVWXYZ abcdefghijklmnopqrstuvwxyz");

//	delay_ms(3000);	
//	
//	LCD_Clear(GRAY0);


//}


//****************************************************************************************/
//函数：void Demo_Menu(void)
//功能描述: 演示程序菜单





/****************************************************************************************/

void Demo_Menu(void)
{
	Show_Picture();

}




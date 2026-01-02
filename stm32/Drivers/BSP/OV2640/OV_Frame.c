#include "OV_Frame.h"
#include "network.h"
#include "network_data.h"
#include "yoloface.h"

extern uint8_t Print_buf[32]; // 消息缓存区

extern uint8_t Key_Flag; // 键值

u8 OV_mode = 0; // bit0:0,RGB565模式;1,JPEG模式

u16 yoffset = 0; // y方向的偏移量


#if USE_HORIZONTAL

#define RGB_Width 240  // 根据屏幕方向，设置缓存大小和格式
#define RGB_Height 240 // 根据屏幕

#else

#define RGB_Width 112  // 根据屏幕方向，设置缓存大小和格式
#define RGB_Height 112 // 根据屏幕

#endif

#define RGB_Dot RGB_Height * RGB_Width * 2

uint8_t RGB_DATA[RGB_Dot];

u32 RGB_Line_Buf[2][RGB_Width * 2]; // RGB屏时,摄像头采用一行一行读取,定义行缓存

/************************************************************************************************/
// 用显示缓冲区数据进行刷屏操作

// STM32H7工程模板-HAL库函数版本

/************************************************************************************************/

u16 *R_Buf; // 刷屏数据指针

/************************************************************************************************/
// 处理帧数据
// 当采集完一帧数据后,调用此函数,帧计数器加一,开始下一帧采集.

// STM32H7工程模板-HAL库函数版本

/************************************************************************************************/

volatile uint16_t RGB_FrameNum = 0;

volatile uint16_t curline = 0; // 摄像头输出数据,当前行编号

void jpeg_data_process(void)
{
    curline = 0; // 行数复位
    RGB_FrameNum = 1; // 帧数计数
}


/************************************************************************************************/
// 数据转存函数

// STM32H7工程模板-HAL库函数版本

/************************************************************************************************/

static void Copy_RAM_Data(u8 *P1, u16 *P2, u16 Num)
{
    
    for(u16 i = 0; i < Num; i++)
    {
        *P1 = (*P2 >> 8)& (0xFF);
        P1++;
        *P1 = *P2 & (0xFF);;
        P1++;
        P2++;
    }
}

/************************************************************************************************/
// RGB屏数据接收回调函数

// STM32H7工程模板-HAL库函数版本

/************************************************************************************************/

void rgblcd_dcmi_rx_callback(void)
{

    if (DMA1_Stream1->CR & (1 << 19)) // DMA使用buf1,读取buf0
    {

        Copy_RAM_Data(&RGB_DATA[curline * RGB_Width * 2], (u16 *)RGB_Line_Buf[0], RGB_Width);

        if (curline < RGB_Height)
            ++curline;
    }
    else // DMA使用buf0,读取buf1
    {

        Copy_RAM_Data(&RGB_DATA[curline * RGB_Width * 2], (u16 *)RGB_Line_Buf[1], RGB_Width);

        if (curline < RGB_Height)
            ++curline;
    }
}


/************************************************************************************************/
// RGB565测试
// RGB数据直接显示在LCD上面

// STM32H7工程模板-HAL库函数版本

/************************************************************************************************/

void RGB565_mode(void)
{
    u8 contrast = 2;

    LCD_Clear(WHITE);

    OV2640_RGB565_Mode(); // RGB565模式

    DCMI_Init(); // DCMI配置

    dcmi_rx_callback = rgblcd_dcmi_rx_callback; // RGB屏接收数据回调函数

    DCMI_DMA_Init((u32)RGB_Line_Buf[0], (u32)RGB_Line_Buf[1], RGB_Width / 2, DMA_MDATAALIGN_HALFWORD, DMA_MINC_ENABLE); // DCMI DMA配置

    OV2640_OutSize_Set(RGB_Width, RGB_Height); // 满屏缩放显示
    
    OV2640_Contrast(contrast);//lcddev

}

void GetImage()
{
    DCMI_Start(); // 启动传输
    while(RGB_FrameNum == 0);
    RGB_FrameNum = 0;
}


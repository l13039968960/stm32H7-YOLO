#include "stm32h7xx_hal.h"
#include "./SYSTEM/usart/usart.h"
#include "./SYSTEM/delay/delay.h"
#include "./SYSTEM/sys/sys.h"
#include "yoloface.h"
#include <stdio.h>
#include "lcd.h"
#include "network.h"
#include "network_data.h"
#include "Picture.h"
#include "spi.h"
#include "OV_Frame.h"
#include "./BSP/MPU/mpu.h"


CRC_HandleTypeDef hcrc;

static void MX_CRC_Init(void);

uint32_t frame = 0;
uint8_t face_num = 0;


int main(void)
{

    sys_cache_enable();                         /* 打开L1-Cache */
    HAL_Init();                                 /* 初始化HAL库 */
    sys_stm32_clock_init(160, 5, 2, 4);         /* 设置时钟, 400Mhz */
//    sys_stm32_clock_init(200, 8, 4, 8);         /* 设置时钟, 400Mhz */
    delay_init(400);                            /* 延时初始化 */
    usart_init(115200);                         /* 初始化USART */
    MPU_Memory_Protection();
    MX_SPI2_Init();
    LCD_Init();
    OV2640_Init();
    MX_CRC_Init();
    aiInit();
    RGB565_mode();

    
    while (1)
    {
        face_num = 0;
        frame++;
        printf("=== Frame %d ===\r\n----------------------------------------\r\n",frame);
        GetImage();
        Show_Picture();
        resize_rgb565_uint8_112_to_56_direct();
        prepare_yolo_data();
        aiRun();
        post_process();
        printf("----------------------------------------\r\n[INFO] Total faces detected: %d\r\n", face_num);
    }

}




/**
  * @brief CRC Initialization Function
  * @param None
  * @retval None
  */
static void MX_CRC_Init(void)
{
    __HAL_RCC_CRC_CLK_ENABLE();
    
    hcrc.Instance = CRC;
    hcrc.Init.DefaultPolynomialUse = DEFAULT_POLYNOMIAL_ENABLE;
    hcrc.Init.DefaultInitValueUse = DEFAULT_INIT_VALUE_ENABLE;
    hcrc.Init.InputDataInversionMode = CRC_INPUTDATA_INVERSION_NONE;
    hcrc.Init.OutputDataInversionMode = CRC_OUTPUTDATA_INVERSION_DISABLE;
    hcrc.InputDataFormat = CRC_INPUTDATA_FORMAT_BYTES;
    HAL_CRC_Init(&hcrc);

}



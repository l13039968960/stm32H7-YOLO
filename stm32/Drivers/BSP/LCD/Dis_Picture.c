#include "Dis_Picture.h"
#include "Picture.h"
#include "spi.h"
#include "./SYSTEM/delay/delay.h"
#include "OV_Frame.h"
#include "./SYSTEM/usart/usart.h"

//uint8_t flag = 0;

volatile uint8_t LCD_TransmitSucc = 0;
//extern uint8_t Tmp_Data[56*56*2];

void Show_Picture(void)
{
    LCD_SetCursor(0, 0);
    LCD_WriteRAM_Prepare();
    LCD_CS_CLR;
    LCD_DC_SET;
    HAL_SPI_Transmit_DMA(&hspi2, (uint8_t *)&RGB_DATA, (uint16_t)25088);
    while(LCD_TransmitSucc == 0);
    LCD_TransmitSucc = 0;
    
}


void HAL_SPI_TxCpltCallback(SPI_HandleTypeDef *hspi)
{
    LCD_TransmitSucc = 1;
}

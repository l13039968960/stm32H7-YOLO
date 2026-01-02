#ifndef __SPI_H__
#define __SPI_H__

#include "stm32h7xx_hal.h"

extern DMA_HandleTypeDef hdma_spi2_tx;
extern SPI_HandleTypeDef hspi2;

void MX_SPI2_Init(void);

#endif


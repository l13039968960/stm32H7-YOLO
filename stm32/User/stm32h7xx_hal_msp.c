/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file         stm32h7xx_hal_msp.c
  * @brief        This file provides code for the MSP Initialization
  *               and de-Initialization codes.
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2025 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */

/* Includes ------------------------------------------------------------------*/
#include "stm32h7xx_hal.h"
#include "./SYSTEM/usart/usart.h"
/* USER CODE BEGIN Includes */

/* USER CODE END Includes */
extern DMA_HandleTypeDef hdma_spi2_tx;

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN TD */

/* USER CODE END TD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN Define */

/* USER CODE END Define */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN Macro */

/* USER CODE END Macro */

/* Private variables ---------------------------------------------------------*/
/* USER CODE BEGIN PV */

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* External functions --------------------------------------------------------*/
/* USER CODE BEGIN ExternalFunctions */

/* USER CODE END ExternalFunctions */

/* USER CODE BEGIN 0 */

/* USER CODE END 0 */
/**
  * Initializes the Global MSP.
  */
//void HAL_MspInit(void)
//{
//  /* USER CODE BEGIN MspInit 0 */

//  /* USER CODE END MspInit 0 */

//  __HAL_RCC_SYSCFG_CLK_ENABLE();

//  HAL_NVIC_SetPriorityGrouping(NVIC_PRIORITYGROUP_2);

//  /* System interrupt init*/

//  /* USER CODE BEGIN MspInit 1 */

//  /* USER CODE END MspInit 1 */
//}

///**
//* @brief SPI MSP Initialization
//* This function configures the hardware resources used in this example
//* @param hspi: SPI handle pointer
//* @retval None
//*/
//void HAL_SPI_MspInit(SPI_HandleTypeDef* hspi)
//{
//  GPIO_InitTypeDef GPIO_InitStruct = {0};
//  RCC_PeriphCLKInitTypeDef PeriphClkInitStruct = {0};
//  if(hspi->Instance==SPI2)
//  {
//  /* USER CODE BEGIN SPI2_MspInit 0 */

//  /* USER CODE END SPI2_MspInit 0 */

//  /** Initializes the peripherals clock
//  */
//    PeriphClkInitStruct.PeriphClockSelection = RCC_PERIPHCLK_SPI2;
//    PeriphClkInitStruct.Spi123ClockSelection = RCC_SPI123CLKSOURCE_PLL;
//    if (HAL_RCCEx_PeriphCLKConfig(&PeriphClkInitStruct) != HAL_OK)
//    {
//      Error_Handler();
//    }

//    /* Peripheral clock enable */
//    __HAL_RCC_SPI2_CLK_ENABLE();

//    __HAL_RCC_GPIOB_CLK_ENABLE();
//    /**SPI2 GPIO Configuration
//    PB13     ------> SPI2_SCK
//    PB14     ------> SPI2_MISO
//    PB15     ------> SPI2_MOSI
//    */
//    GPIO_InitStruct.Pin = GPIO_PIN_13|GPIO_PIN_14|GPIO_PIN_15;
//    GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
//    GPIO_InitStruct.Pull = GPIO_NOPULL;
//    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
//    GPIO_InitStruct.Alternate = GPIO_AF5_SPI2;
//    HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

//    /* SPI2 DMA Init */
//    /* SPI2_TX Init */
//    hdma_spi2_tx.Instance = DMA1_Stream0;
//    hdma_spi2_tx.Init.Request = DMA_REQUEST_SPI2_TX;
//    hdma_spi2_tx.Init.Direction = DMA_MEMORY_TO_PERIPH;
//    hdma_spi2_tx.Init.PeriphInc = DMA_PINC_DISABLE;
//    hdma_spi2_tx.Init.MemInc = DMA_MINC_ENABLE;
//    hdma_spi2_tx.Init.PeriphDataAlignment = DMA_PDATAALIGN_BYTE;
//    hdma_spi2_tx.Init.MemDataAlignment = DMA_MDATAALIGN_BYTE;
//    hdma_spi2_tx.Init.Mode = DMA_NORMAL;
//    hdma_spi2_tx.Init.Priority = DMA_PRIORITY_VERY_HIGH;
//    hdma_spi2_tx.Init.FIFOMode = DMA_FIFOMODE_DISABLE;
//    if (HAL_DMA_Init(&hdma_spi2_tx) != HAL_OK)
//    {
//      Error_Handler();
//    }

//    __HAL_LINKDMA(hspi,hdmatx,hdma_spi2_tx);

//    /* SPI2 interrupt Init */
//    HAL_NVIC_SetPriority(SPI2_IRQn, 2, 1);
//    HAL_NVIC_EnableIRQ(SPI2_IRQn);
//  /* USER CODE BEGIN SPI2_MspInit 1 */

//  /* USER CODE END SPI2_MspInit 1 */
//  }

//}

///**
//* @brief SPI MSP De-Initialization
//* This function freeze the hardware resources used in this example
//* @param hspi: SPI handle pointer
//* @retval None
//*/
//void HAL_SPI_MspDeInit(SPI_HandleTypeDef* hspi)
//{
//  if(hspi->Instance==SPI2)
//  {
//  /* USER CODE BEGIN SPI2_MspDeInit 0 */

//  /* USER CODE END SPI2_MspDeInit 0 */
//    /* Peripheral clock disable */
//    __HAL_RCC_SPI2_CLK_DISABLE();

//    /**SPI2 GPIO Configuration
//    PB13     ------> SPI2_SCK
//    PB14     ------> SPI2_MISO
//    PB15     ------> SPI2_MOSI
//    */
//    HAL_GPIO_DeInit(GPIOB, GPIO_PIN_13|GPIO_PIN_14|GPIO_PIN_15);

//    /* SPI2 DMA DeInit */
//    HAL_DMA_DeInit(hspi->hdmatx);

//    /* SPI2 interrupt DeInit */
//    HAL_NVIC_DisableIRQ(SPI2_IRQn);
//  /* USER CODE BEGIN SPI2_MspDeInit 1 */

//  /* USER CODE END SPI2_MspDeInit 1 */
//  }

//}

//void HAL_SPI_TxCpltCallback(SPI_HandleTypeDef *hspi)
//{
//    static uint8_t i = 0;
//    
//    printf("2.进入中断%d\r\n",i);
//    i++;
//    // 判断是不是我们关心的SPI2
//    if(hspi->Instance == SPI2)
//    {
//        
//        LCD_CS_SET; 
//        flag = 0;
//        if(flag_2 == 1)
//            flag_2 = 0;
//        else
//            flag_2 = 1;
//    }
//}

/* USER CODE BEGIN 1 */

/* USER CODE END 1 */

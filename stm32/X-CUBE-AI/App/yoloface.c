#include "yoloface.h"
#include "Picture.h"
#include "lcd.h"

#include <stdio.h>

#include "network.h"
#include "network_data.h"
#include "OV_Frame.h"
AI_ALIGNED(32)
ai_i8 in_data[AI_NETWORK_IN_1_SIZE];

// 定义网络输出数组
AI_ALIGNED(32)
ai_i8 out_data[AI_NETWORK_OUT_1_SIZE];

// 人脸方框的左上右下像素坐标
int x1, y1, x2, y2;
// yoloface的anchor尺寸
uint8_t anchors[3][2] = {{9, 14}, {12, 17}, {22, 21}};

uint8_t Tmp_Data[56*56*2];

extern uint8_t face_num;

void resize_rgb565_uint8_112_to_56_direct(void) 
{
    for (int y = 0; y < 56; y++) {
        for (int x = 0; x < 56; x++) {
            // 源图像中对应的2x2块起始位置（字节偏移）
            int src_base_y = y * 2;
            int src_base_x = x * 2;
            
            uint32_t sum_r = 0, sum_g = 0, sum_b = 0;
            
            // 遍历2x2区域
            for (int dy = 0; dy < 2; dy++) {
                for (int dx = 0; dx < 2; dx++) {
                    // 计算像素的字节偏移
                    int src_byte_offset = ((src_base_y + dy) * 112 + (src_base_x + dx)) * 2;
                    
                    // 组合两个字节为16位像素值（小端字节序：低位在前）
                    uint16_t pixel = ((uint16_t)RGB_DATA[src_byte_offset] << 8) | 
                                      RGB_DATA[src_byte_offset + 1];
                    
                    // 提取RGB565分量
                    sum_r += (pixel >> 11) & 0x1F;
                    sum_g += (pixel >> 5)  & 0x3F;
                    sum_b += (pixel)       & 0x1F;
                }
            }
            
            // 计算平均值
            uint8_t avg_r = (uint8_t)(sum_r >> 2);
            uint8_t avg_g = (uint8_t)(sum_g >> 2);
            uint8_t avg_b = (uint8_t)(sum_b >> 2);
            
            // 重新组合为RGB565
            uint16_t result = ((avg_r & 0x1F) << 11) | 
                              ((avg_g & 0x3F) << 5)  | 
                              (avg_b & 0x1F);
            
            // 计算目标字节偏移
            int dst_byte_offset = (y * 56 + x) * 2;
            
            // 存储为两个字节（小端字节序）
            Tmp_Data[dst_byte_offset + 1] = result & 0xFF;        // 低字节
            Tmp_Data[dst_byte_offset] = (result >> 8) & 0xFF; // 高字节
        }
    }
}

void prepare_yolo_data(void)
{
    ai_u8 r,g,b;
    for(int i = 0; i < 56; i++)
    {
      for(int j = 0; j < 56; j++)
      {
        uint16_t color = Tmp_Data[(i*56+j)*2];
        color = (color<<8)|Tmp_Data[(i*56+j)*2+1];

        r = ((color&0xF800)>>8);
        g = ((color&0x07E0)>>3);
        b = ((color&0x001F)<<3);

        in_data[(i*56+j)*3] = (int8_t)r - 128;
        in_data[(i*56+j)*3+1] = (int8_t)g - 128;
        in_data[(i*56+j)*3+2] = (int8_t)b - 128;
      }
    }

}



// 定义sigmoid函数
float sigmoid(float x)
{
	float y = 1/(1+expf(-x));
	return y;
}


void post_process(void)
{
    float max_conf = 0;
    int max_i, max_j = 0;
    int grid_x, grid_y;
    float x, y, w ,h;
    for(int i = 0; i < 49; i++)
    {
        for(int j = 0; j < 3; j++)
        {

            float conf = sigmoid((out_data[i*18+j*6+4]+15)*0.14218327403068542f);
//            if(conf > max_conf)
//            {
//                max_i = i;
//                max_j = j;
//                max_conf = conf;
//            }
            if(conf >= 0.7)
            {
                face_num++;
//                max_conf = conf;
//                max_i = i;
//                max_j = j;
                grid_x = i % 7;
                grid_y = (i - grid_x)/7;
                x = ((float)out_data[i*18+j*6]+15)*0.14218327403068542f;
                y = ((float)out_data[i*18+j*6+1]+15)*0.14218327403068542f;
                w = ((float)out_data[i*18+j*6+2]+15)*0.14218327403068542f;
                h = ((float)out_data[i*18+j*6+3]+15)*0.14218327403068542f;
                x = (sigmoid(x)+grid_x) * 8;
                y = (sigmoid(y)+grid_y) * 8;
                w = expf(w) * anchors[j][0];
                h = expf(h) * anchors[j][1];
                y2 = (x - w/2);
                y1 = (x + w/2);
                x1 = y - h/2;
                x2 = y + h/2;
                if(x1 < 0) x1 = 0;
                if(y1 < 0) y1 = 0;
                if(x2 > 55) x2 = 55;
                if(y2 > 55) y2 = 55;
                LCD_DrawRectangle(x1 * 2, y1 * 2, x2 * 2, y2 * 2, RED);
                printf("[Face %d] BBox: [%d, %d, %d, %d], Conf: %.2f\r\n", face_num, x1*2, y1*2, x2*2, y2*2, conf);
                
            }
        }
    }
//            if(max_conf > 0.7f)
//            {
//                grid_x = max_i % 7;
//                grid_y = (max_i - grid_x)/7;
//                x = ((float)out_data[max_i*18+max_j*6]+15)*0.14218327403068542f;
//                y = ((float)out_data[max_i*18+max_j*6+1]+15)*0.14218327403068542f;
//                w = ((float)out_data[max_i*18+max_j*6+2]+15)*0.14218327403068542f;
//                h = ((float)out_data[max_i*18+max_j*6+3]+15)*0.14218327403068542f;
//                x = (sigmoid(x)+grid_x) * 8;
//                y = (sigmoid(y)+grid_y) * 8;
//                w = expf(w) * anchors[max_j][0];
//                h = expf(h) * anchors[max_j][1];
//                y2 = (x - w/2);
//                y1 = (x + w/2);
//                x1 = y - h/2;
//                x2 = y + h/2;
//                if(x1 < 0) x1 = 0;
//                if(y1 < 0) y1 = 0;
//                if(x2 > 55) x2 = 55;
//                if(y2 > 55) y2 = 55;
//                LCD_DrawRectangle(x1 * 2, y1 * 2, x2 * 2, y2 * 2, RED);
//            }
}
/* Global handle to reference an instantiated C-model */
static ai_handle network = AI_HANDLE_NULL;

/* Global c-array to handle the activations buffer */
AI_ALIGNED(32)
static ai_u8 activations[AI_NETWORK_DATA_ACTIVATIONS_SIZE];

// 定义网络输入数组

/* 
 * Bootstrap code
 */
int aiInit(void) {
  ai_error err;
  
  /* 1 - Create an instance of the model */
  err = ai_network_create(&network, AI_NETWORK_DATA_CONFIG /* or NULL */);
  if (err.type != AI_ERROR_NONE) {
    printf("E: AI ai_network_create error - type=%d code=%d\r\n", err.type, err.code);
    return -1;
    };

  /* 2 - Initialize the instance */
  const ai_network_params params = AI_NETWORK_PARAMS_INIT(
      AI_NETWORK_DATA_WEIGHTS(ai_network_data_weights_get()),
      AI_NETWORK_DATA_ACTIVATIONS(activations)
  );

  if (!ai_network_init(network, &params)) {
      err = ai_network_get_error(network);
      printf("E: AI ai_network_init error - type=%d code=%d\r\n", err.type, err.code);
      return -1;
    }

  return 0;
}

/* 
 * Run inference code
 */
int aiRun(void)
{
  ai_i32 n_batch;
  ai_error err;

  /* 1 - Create the AI buffer IO handlers with the default definition */
  ai_buffer ai_input[AI_NETWORK_IN_NUM] = AI_NETWORK_IN ;
  ai_buffer ai_output[AI_NETWORK_OUT_NUM] = AI_NETWORK_OUT ;
  
  /* 2 - Update IO handlers with the data payload */
  ai_input[0].n_batches = 1;
  ai_input[0].data = AI_HANDLE_PTR(in_data);
  ai_output[0].n_batches = 1;
  ai_output[0].data = AI_HANDLE_PTR(out_data);

  /* 3 - Perform the inference */
  n_batch = ai_network_run(network, &ai_input[0], &ai_output[0]);
  if (n_batch != 1) {
      err = ai_network_get_error(network);
      printf("E: AI ai_network_run error - type=%d code=%d\r\n", err.type, err.code);
      return -1;
  };
  
  return 0;
}


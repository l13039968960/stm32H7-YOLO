#ifndef YOLO_FACE_H
#define YOLO_FACE_H

void resize_rgb565_uint8_112_to_56_direct(void);
void prepare_yolo_data(void);
void post_process(void);
int aiInit(void);
int aiRun(void);

#endif

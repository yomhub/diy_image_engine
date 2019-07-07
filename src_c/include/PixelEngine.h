#include <stdio.h>
#include <stdbool.h> // Bool
#include <stdint.h>  // For uint**_t
#include <math.h> // float_t
#include <malloc.h>

typedef enum EngineState
{
	ENG_SUCCESS = 0,
    ENG_RUNNING,
    ENG_SLEEP,
    ENG_ERR,
    ENG_BUSSY,
    ENG_MEMERY_INSUFFICIENT
} EngineState;

typedef struct Pixels
{
    uint16_t width;
    uint16_t height;
    uint8_t sizePerPixel;
    union{
        uint8_t *data;
        float_t *mask;
    };
} Pixels;

struct pixelEngine{

    EngineState (*smooth)
    (Pixels *src, Pixels *mask, float_t factor);
    EngineState (*smooth2D)
    (Pixels *src, Pixels *mask1, Pixels *mask2, float_t factor1, float_t factor2);
    EngineState (*resize)
    (Pixels *src, uint16_t newWidth, uint16_t newHeight, uint8_t mode);
    EngineState (*rotate)
    (Pixels *src, float_t angle, uint8_t mode);
    EngineState (*flip)
    (Pixels *src, uint8_t mode, uint16_t selectLine);
};

extern const struct pixelEngine PixelEngine;


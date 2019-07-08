#include "include/PixelEngine.h"

#define myassert(expression) \
    if (!(expression))       \
        return ENG_ERR;

#define ptrImgChk(img)                                               \
    myassert(img && img->height && img->width && img->sizePerPixel); \
    myassert(img->data &&malloc_usable_size(img->data) >= img->height * img->width * img->sizePerPixel);
#define ptrMskChk(ms)														\
	myassert(ms && ms->height && ms->width && ms->sizePerPixel && ms->mask);	
#define thhold(start, end, val) start > val ? start : end < val ? end : val;
#define abs(a) (a>0?a:-a)

EngineState pegmalloc(uint8_t **buff, size_t size)
{
    if (!size)
        return ENG_ERR;
    if (*buff)
    {
		if (malloc_usable_size(*buff) < size) {
			free(*buff);
			*buff = NULL;
		}
		else
		{
			return ENG_SUCCESS;
		}  
    }
    *buff = (uint8_t *)malloc(size);
    if (!*buff || malloc_usable_size(*buff) < size)
        return ENG_MEMERY_INSUFFICIENT;
    return ENG_SUCCESS;
}

EngineState smooth3x3(Pixels *src, Pixels *mask, float_t factor);
EngineState smooth2D3x3(Pixels *src, Pixels *mask1, Pixels *mask2, float_t factor1, float_t factor2);
EngineState resize(Pixels *src, uint16_t newWidth, uint16_t newHeight, uint8_t mode);
EngineState rotate(Pixels *src, float_t angle, uint8_t mode);
EngineState flip(Pixels *src, uint8_t mode, uint16_t selectLine);

/*

Will return A if B:

	ENG_READY		success
	ENG_RUNNING		aouther operation is running

Mask is:
				 [m  m  m]
		faceor * [m  m  m]
				 [m  m  m]

Return ENG_ERR if:
	pixels data size < height * width * sizePerPixel
	pixels sizePerPixel or height or width is 0
	Mask matrix buffer < n * n
*/
EngineState smooth3x3(Pixels *src, Pixels *mask, float_t factor)
{
    ptrImgChk(src);
	ptrMskChk(mask);

    uint8_t *buff = NULL;

	if (pegmalloc(&buff, src->height * src->width * src->sizePerPixel) != ENG_SUCCESS)
		return ENG_MEMERY_INSUFFICIENT;
    memcpy(buff, src->data, src->height * src->width * src->sizePerPixel);

    uint16_t baceAdrY[3] = {0};
    uint16_t boundX = src->width - mask->width / 2, boundY = src->height - mask->height / 2;
    float_t unsBuf;
    for (size_t row = mask->height / 2 + mask->height % 2; row < boundY; row += src->sizePerPixel)
    {
        baceAdrY[0] = (row - 1) * src->width;
        baceAdrY[1] = (row) * src->width;
        baceAdrY[2] = (row + 1) * src->width;

        for (size_t col = (mask->width / 2 + mask->width % 2); col < boundX; col += src->sizePerPixel)
        {
            unsBuf = (buff[col - 1 + baceAdrY[0]] * mask->mask[0] +
                      buff[col + baceAdrY[0]] * mask->mask[1] +
                      buff[col + 1 + baceAdrY[0]] * mask->mask[2] +
                      buff[col - 1 + baceAdrY[1]] * mask->mask[3] +
                      buff[col + baceAdrY[1]] * mask->mask[4] +
                      buff[col + 1 + baceAdrY[1]] * mask->mask[5] +
                      buff[col - 1 + baceAdrY[2]] * mask->mask[6] +
                      buff[col + baceAdrY[2]] * mask->mask[7] +
                      buff[col + 1 + baceAdrY[2]] * mask->mask[8]) *
                     factor;
            src->data[col + baceAdrY[1]] = thhold(0, 255, unsBuf);
        }
    }
    if (src->sizePerPixel > 1)
    {
        for (size_t row = mask->height / 2 + mask->height % 2; row < boundY; row += src->sizePerPixel)
        {
            baceAdrY[0] = (row - 1) * src->width;
            baceAdrY[1] = (row)* src->width;;
            baceAdrY[2] = (row + 1) * src->width;
            for (size_t col = (mask->width / 2 + mask->width % 2); col < boundX; col += src->sizePerPixel)
            {
                unsBuf = (buff[col - 1 + baceAdrY[0] + 1] * mask->mask[0] +
                          buff[col + baceAdrY[0] + 1] * mask->mask[1] +
                          buff[col + 1 + baceAdrY[0] + 1] * mask->mask[2] +
                          buff[col - 1 + baceAdrY[1] + 1] * mask->mask[3] +
                          buff[col + baceAdrY[1] + 1] * mask->mask[4] +
                          buff[col + 1 + baceAdrY[1] + 1] * mask->mask[5] +
                          buff[col - 1 + baceAdrY[2] + 1] * mask->mask[6] +
                          buff[col + baceAdrY[2] + 1] * mask->mask[7] +
                          buff[col + 1 + baceAdrY[2] + 1] * mask->mask[8]) *
                         factor;
                src->data[col + baceAdrY[1] + 1] = thhold(0, 255, unsBuf);
            }
        }
    }
    if (src->sizePerPixel > 2)
    {
        for (size_t row = mask->height / 2 + mask->height % 2; row < boundY; row += src->sizePerPixel)
        {
            baceAdrY[0] = (row - 1) * src->width;
            baceAdrY[1] = (row)* src->width;;
            baceAdrY[2] = (row + 1) * src->width;
            for (size_t col = (mask->width / 2 + mask->width % 2); col < boundX; col += src->sizePerPixel)
            {
                unsBuf = (buff[col - 1 + baceAdrY[0] + 2] * mask->mask[0] +
                          buff[col + baceAdrY[0] + 2] * mask->mask[1] +
                          buff[col + 1 + baceAdrY[0] + 2] * mask->mask[2] +
                          buff[col - 1 + baceAdrY[1] + 2] * mask->mask[3] +
                          buff[col + baceAdrY[1] + 2] * mask->mask[4] +
                          buff[col + 1 + baceAdrY[1] + 2] * mask->mask[5] +
                          buff[col - 1 + baceAdrY[2] + 2] * mask->mask[6] +
                          buff[col + baceAdrY[2] + 2] * mask->mask[7] +
                          buff[col + 1 + baceAdrY[2] + 2] * mask->mask[8]) *
                         factor;
                src->data[col + baceAdrY[1] + 2] = thhold(0, 255, unsBuf);
            }
        }
    }

    free(buff);
    return ENG_SUCCESS;
}

EngineState smooth2D3x3(Pixels *src, Pixels *mask1, Pixels *mask2, float_t factor1, float_t factor2)
{
    ptrImgChk(src);
	ptrMskChk(mask1);
	ptrMskChk(mask2);

    uint8_t *buff = NULL;

	if (pegmalloc(&buff, src->height * src->width * src->sizePerPixel) != ENG_SUCCESS)
		return ENG_MEMERY_INSUFFICIENT;

    memcpy(buff, src->data, src->height * src->width * src->sizePerPixel);

    float_t unsBuf1;
    float_t unsBuf2;
    uint16_t baceAdrY[3] = {0};
    uint16_t boundX = src->width - mask1->width / 2, boundY = src->height - mask1->height / 2;
    for (size_t row = (mask1->height / 2 + mask1->height % 2); row < boundY; row += src->sizePerPixel)
    {
        baceAdrY[0] = (row - 1) * src->width;
        baceAdrY[1] = (row)* src->width;;
        baceAdrY[2] = (row + 1) * src->width;
        for (size_t col = (mask1->width / 2 + mask1->width % 2); col < boundX; col += src->sizePerPixel)
        {
            unsBuf1 = buff[col - 1 + (row - 1) * src->width] * mask1->mask[0] +
                      buff[col + (row - 1) * src->width] * mask1->mask[1] +
                      buff[col + 1 + (row - 1) * src->width] * mask1->mask[2] +
                      buff[col - 1 + (row)*src->width] * mask1->mask[3] +
                      buff[col + (row)*src->width] * mask1->mask[4] +
                      buff[col + 1 + (row)*src->width] * mask1->mask[5] +
                      buff[col - 1 + (row + 1) * src->width] * mask1->mask[6] +
                      buff[col + (row + 1) * src->width] * mask1->mask[7] +
                      buff[col + 1 + (row + 1) * src->width] * mask1->mask[8];

            unsBuf2 = buff[col - 1 + (row - 1) * src->width] * mask2->mask[0] +
                      buff[col + (row - 1) * src->width] * mask2->mask[1] +
                      buff[col + 1 + (row - 1) * src->width] * mask2->mask[2] +
                      buff[col - 1 + (row)*src->width] * mask2->mask[3] +
                      buff[col + (row)*src->width] * mask2->mask[4] +
                      buff[col + 1 + (row)*src->width] * mask2->mask[5] +
                      buff[col - 1 + (row + 1) * src->width] * mask2->mask[6] +
                      buff[col + (row + 1) * src->width] * mask2->mask[7] +
                      buff[col + 1 + (row + 1) * src->width] * mask2->mask[8];
            unsBuf1 *= factor1;
            unsBuf2 *= factor2;
            unsBuf1 = sqrtf(unsBuf1 * unsBuf1 + unsBuf2 * unsBuf2);
            src->data[col + baceAdrY[1]] = thhold(0, 255, unsBuf1);
        }
    }
    if (src->sizePerPixel > 1)
    {
        for (size_t row = (mask1->height / 2 + mask1->height % 2); row < boundY; row += src->sizePerPixel)
        {
            baceAdrY[0] = (row - 1) * src->width;
            baceAdrY[1] = (row)* src->width;;
            baceAdrY[2] = (row + 1) * src->width;
            for (size_t col = (mask1->width / 2 + mask1->width % 2); col < boundX; col += src->sizePerPixel)
            {

                unsBuf1 = buff[col - 1 + (row - 1) * src->width + 1] * mask1->mask[0] +
                          buff[col + (row - 1) * src->width + 1] * mask1->mask[1] +
                          buff[col + 1 + (row - 1) * src->width + 1] * mask1->mask[2] +
                          buff[col - 1 + (row)*src->width + 1] * mask1->mask[3] +
                          buff[col + (row)*src->width + 1] * mask1->mask[4] +
                          buff[col + 1 + (row)*src->width + 1] * mask1->mask[5] +
                          buff[col - 1 + (row + 1) * src->width + 1] * mask1->mask[6] +
                          buff[col + (row + 1) * src->width + 1] * mask1->mask[7] +
                          buff[col + 1 + (row + 1) * src->width + 1] * mask1->mask[8];

                unsBuf2 = buff[col - 1 + (row - 1) * src->width + 1] * mask2->mask[0] +
                          buff[col + (row - 1) * src->width + 1] * mask2->mask[1] +
                          buff[col + 1 + (row - 1) * src->width + 1] * mask2->mask[2] +
                          buff[col - 1 + (row)*src->width + 1] * mask2->mask[3] +
                          buff[col + (row)*src->width + 1] * mask2->mask[4] +
                          buff[col + 1 + (row)*src->width + 1] * mask2->mask[5] +
                          buff[col - 1 + (row + 1) * src->width + 1] * mask2->mask[6] +
                          buff[col + (row + 1) * src->width + 1] * mask2->mask[7] +
                          buff[col + 1 + (row + 1) * src->width + 1] * mask2->mask[8];
                unsBuf1 *= factor1;
                unsBuf2 *= factor2;
                unsBuf1 = sqrtf(unsBuf1 * unsBuf1 + unsBuf2 * unsBuf2);
                src->data[col + baceAdrY[1] + 1] = thhold(0, 255, unsBuf1);
            }
        }
    }
    if (src->sizePerPixel > 2)
    {
        for (size_t row = (mask1->height / 2 + mask1->height % 2); row < boundY; row += src->sizePerPixel)
        {
            baceAdrY[0] = (row - 1) * src->width;
            baceAdrY[1] = (row)* src->width;;
            baceAdrY[2] = (row + 1) * src->width;
            for (size_t col = (mask1->width / 2 + mask1->width % 2); col < boundX; col += src->sizePerPixel)
            {
                unsBuf1 = buff[col - 1 + (row - 1) * src->width + 2] * mask1->mask[0] +
                          buff[col + (row - 1) * src->width + 2] * mask1->mask[1] +
                          buff[col + 1 + (row - 1) * src->width + 2] * mask1->mask[2] +
                          buff[col - 1 + (row)*src->width + 2] * mask1->mask[3] +
                          buff[col + (row)*src->width + 2] * mask1->mask[4] +
                          buff[col + 1 + (row)*src->width + 2] * mask1->mask[5] +
                          buff[col - 1 + (row + 1) * src->width + 2] * mask1->mask[6] +
                          buff[col + (row + 1) * src->width + 2] * mask1->mask[7] +
                          buff[col + 1 + (row + 1) * src->width + 2] * mask1->mask[8];
                unsBuf2 = buff[col - 1 + (row - 1) * src->width + 2] * mask2->mask[0] +
                          buff[col + (row - 1) * src->width + 2] * mask2->mask[1] +
                          buff[col + 1 + (row - 1) * src->width + 2] * mask2->mask[2] +
                          buff[col - 1 + (row)*src->width + 2] * mask2->mask[3] +
                          buff[col + (row)*src->width + 2] * mask2->mask[4] +
                          buff[col + 1 + (row)*src->width + 2] * mask2->mask[5] +
                          buff[col - 1 + (row + 1) * src->width + 2] * mask2->mask[6] +
                          buff[col + (row + 1) * src->width + 2] * mask2->mask[7] +
                          buff[col + 1 + (row + 1) * src->width + 2] * mask2->mask[8];
                unsBuf1 *= factor1;
                unsBuf2 *= factor2;
                unsBuf1 = sqrtf(unsBuf1 * unsBuf1 + unsBuf2 * unsBuf2);
                src->data[col + baceAdrY[1] + 2] = thhold(0, 255, unsBuf1);
            }
        }
    }

    free(buff);
    return ENG_SUCCESS;
}

EngineState resize(Pixels *src, uint16_t newWidth, uint16_t newHeight, uint8_t mode)
{
    ptrImgChk(src);

    bool b_x2Big = newWidth > src->width ? true : false;
    bool b_y2Big = newHeight > src->height ? true : false;

    uint16_t x_big = newWidth > src->width ? newWidth : src->width;
    uint16_t x_small = newWidth < src->width ? newWidth : src->width;

    uint16_t y_big = newHeight > src->height ? newHeight : src->height;
    uint16_t y_small = newHeight < src->height ? newHeight : src->height;

    uint16_t x_step = x_big / x_small + (x_big % x_small == 0 ? 0 : 1);
    uint16_t y_step = y_big / y_small + (y_big % y_small == 0 ? 0 : 1);

    uint8_t *buff = NULL;

	if (pegmalloc(&buff, src->height * src->width * src->sizePerPixel) != ENG_SUCCESS)
		return ENG_MEMERY_INSUFFICIENT;

    size_t detY = b_y2Big ? y_step * src->sizePerPixel : src->sizePerPixel;
	size_t detX = b_x2Big ? x_step * src->sizePerPixel : src->sizePerPixel;
    size_t lineSpace, bsAdrX, newBsAdrY;

    for (size_t row = 0; row < newHeight; row += detY)
    {
        // Line sampling
        lineSpace = b_y2Big ? row / y_step : row * y_step * src->width;
        newBsAdrY = row * newWidth;

        for (size_t col = 0; col < newWidth - 1; col += detX)
        {
            bsAdrX = col * x_step;
            if (b_x2Big)
            {
                // X amplification
                for (size_t x = 0; x < x_step; ++x)
                {
                    buff[newBsAdrY + col +x] = thhold(0, 255, src->data[bsAdrX + lineSpace] + (src->data[bsAdrX + lineSpace] - src->data[(col + 1) * x_step + lineSpace]) * x / x_step);
                    if (src->sizePerPixel > 1)
                        buff[newBsAdrY + col+x + 1] = thhold(0, 255, src->data[bsAdrX + lineSpace + 1] + (src->data[bsAdrX + lineSpace + 1] - src->data[(col + 1) * x_step + lineSpace + 1]) * x / x_step);
                    if (src->sizePerPixel > 2)
                        buff[newBsAdrY + col+x + 2] = thhold(0, 255, src->data[bsAdrX + lineSpace + 2] + (src->data[bsAdrX + lineSpace + 2] - src->data[(col + 1) * x_step + lineSpace + 2]) * x / x_step);
                }
            }
            else
            {
                // X narrow
                buff[newBsAdrY + col] = src->data[bsAdrX + lineSpace];
                if (src->sizePerPixel > 1)
                    buff[newBsAdrY + col + 1] = src->data[bsAdrX + lineSpace + 1];
                if (src->sizePerPixel > 2)
                    buff[newBsAdrY + col + 2] = src->data[bsAdrX + lineSpace + 2];
            }

            if (b_y2Big && row > 0)
            {
                // Y amplification
                for (size_t y = 1; y < y_step; ++y)
                {
                    buff[(row + y - y_step) * newWidth + col] =
                        thhold(0, 255, buff[(row - y_step) * newWidth + col] + (buff[(row - y_step) * newWidth + col] - buff[newBsAdrY + col]) * y / y_step);
                    if (src->sizePerPixel > 1)
                        buff[(row + y - y_step) * newWidth + col + 1] =
                            thhold(0, 255, buff[(row - y_step) * newWidth + col + 1] + (buff[(row - y_step) * newWidth + col + 1] - buff[newBsAdrY + col + 1]) * y / y_step);
                    if (src->sizePerPixel > 2)
                        buff[(row + y - y_step) * newWidth + col + 2] =
                            thhold(0, 255, buff[(row - y_step) * newWidth + col + 2] + (buff[(row - y_step) * newWidth + col + 2] - buff[newBsAdrY + col + 2]) * y / y_step);
                }
            }
        }
    }

    free(src->data);
    src->data = NULL;

	if (pegmalloc(&src->data, newHeight * newWidth * src->sizePerPixel) != ENG_SUCCESS)
		return ENG_MEMERY_INSUFFICIENT;
    memcpy(src->data, buff, newHeight * newWidth * src->sizePerPixel);

    src->height = newHeight;
    src->width = newWidth;
    free(buff);
    return ENG_SUCCESS;
}

/*
Assume the origin is in the upper left corner of the coordinate system

origin
	+--------------------. x+
	| \ [angle]
	|  \
	|   \
	|    \
	|     \
	|
	|
	y+

So, New X = height*cos(a+90)+width*cos(a)
	New Y = height*sin(a+90)+width*sin(a)

	Origin image's col row . new image's dx dy is:

	dX=(height - row)*cos(a+90)+col*cos(a)
	dY=row*sin(a+90)+col*sin(a)
	
An ENG_ERR is return if:
	pixels data size < height * width * sizePerPixel
	pixels sizePerPixel or height or width is 0

*/
EngineState rotate(Pixels *src, float_t angle, uint8_t mode)
{
    ptrImgChk(src);
	int8_t n90 = abs(angle / 90);
	n90 = abs(n90);
	float_t angleR = ((abs(angle) - n90 * 90) * 3.14f / 180);
    bool b_change;
    // Decide whether to exchange XY
    switch (((int)(angle) / 90) % 4)
    {
    case 1:
    case 3:
        b_change = true;
        break;
    case 0:
    case 2:
    default:
        b_change = false;
        break;
    }
    float_t cosR = cos(angleR), sinR = sin(angleR);

    uint16_t x = src->height * abs(sinR) + src->width * abs(cosR);
    uint16_t y = src->height * abs(cosR) + src->width * abs(sinR);
    uint16_t dx = 0, dy = 0;
    uint8_t *buff=NULL;

	if (pegmalloc(&buff, (size_t)(x * y * src->sizePerPixel)) != ENG_SUCCESS)
		return ENG_MEMERY_INSUFFICIENT;
	memset(buff,0, malloc_usable_size(buff));
    uint16_t newBsAdr,bsAdrY;

    for (size_t row = 0; row < src->height; row += src->sizePerPixel)
    {
		bsAdrY = row * src->width;
        for (size_t col = 0; col < src->width; col += src->sizePerPixel)
        {
            dy = row * cosR + col * sinR;
            dx = (src->height - row) * sinR + col * cosR;
            newBsAdr = b_change ? (dx * y + y - dy) : (dx + dy * x);
			/*
            buff[newBsAdr] = src->data[bsAdrY + col + 0];
            if (src->sizePerPixel > 1)
                buff[newBsAdr + 1] = src->data[bsAdrY + col + 1];
            if (src->sizePerPixel > 2)
                buff[newBsAdr + 2] = src->data[bsAdrY + col + 2];
				*/
			buff[b_change ? (dx * y + y - dy + 0) : (dx + dy * x + 0)] = src->data[row * src->width + col + 0];
			if (src->sizePerPixel > 1)buff[b_change ? (dx * y + y - dy + 1) : (dx + dy * x + 1)] = src->data[row * src->width + col + 1];
			if (src->sizePerPixel > 2)buff[b_change ? (dx * y + y - dy + 2) : (dx + dy * x + 2)] = src->data[row * src->width + col + 2];
        }
    }

    free(src->data);
    src->data = NULL;
	if (pegmalloc(&src->data, (size_t)(x * y * src->sizePerPixel)) != ENG_SUCCESS)
		return ENG_MEMERY_INSUFFICIENT;
	size_t tmp = malloc_usable_size(buff);
	tmp = malloc_usable_size(src->data);
    memcpy(src->data, buff, x * y * src->sizePerPixel);

    src->height = b_change ? x : y;
    src->width = b_change ? y : x;
    free(buff);
    return ENG_SUCCESS;
}

/*
Will flip pixels

Mode: 0: flip vertically
	  1: flip horizontally

selectLine: 0: Flip the entire image (Default)
			N: Flip the first N row/col


An peg::ENG_ERR is return if:
	pixels data size < height * width * sizePerPixel
	pixels sizePerPixel or height or width is 0
*/

EngineState flip(Pixels *src, uint8_t mode, uint16_t selectLine)
{
    ptrImgChk(src);

    uint8_t buff;
    uint16_t bsAdrY;
	if (selectLine == 0)selectLine = mode ? src->width / 2 : src->height / 2;
    switch (mode)
    {
    case 0:
        // vertically
        for (size_t row = 0; row < (src->height / 2) && row < selectLine; row += src->sizePerPixel){
            bsAdrY = row * src->width;
            for (size_t col = 0; col < src->width; col += src->sizePerPixel)
            {
                buff = src->data[bsAdrY + col + 0];
                src->data[bsAdrY + col + 0] = src->data[(src->height - row - 1) * src->width + col + 0];
                src->data[(src->height - row - 1) * src->width + col + 0] = buff;
                if (src->sizePerPixel > 1)
                {
                    buff = src->data[bsAdrY + col + 1];
                    src->data[bsAdrY + col + 1] = src->data[(src->height - row - 1) * src->width + col + 1];
                    src->data[(src->height - row - 1) * src->width + col + 1] = buff;
                }
                if (src->sizePerPixel > 2)
                {
                    buff = src->data[bsAdrY + col + 2];
                    src->data[bsAdrY + col + 2] = src->data[(src->height - row - 1) * src->width + col + 2];
                    src->data[(src->height - row - 1) * src->width + col + 2] = buff;
                }
            }
        }
        break;
    default:
        // horizontally
        for (size_t row = 0; row < src->height; row += src->sizePerPixel){
            bsAdrY = row * src->width;
            for (size_t col = 0; col < (src->width / 2) && col < selectLine; col += src->sizePerPixel)
            {
                buff = src->data[bsAdrY + col + 0];
                src->data[bsAdrY + col + 0] = src->data[bsAdrY + (src->width - col - 1) + 0];
                src->data[bsAdrY + (src->width - col - 1) + 0] = buff;
                if (src->sizePerPixel > 1)
                {
                    buff = src->data[bsAdrY + col + 1];
                    src->data[bsAdrY + col + 1] = src->data[bsAdrY + (src->width - col - 1) + 1];
                    src->data[bsAdrY + (src->width - col - 1) + 1] = buff;
                }
                if (src->sizePerPixel > 2)
                {
                    buff = src->data[bsAdrY + col + 2];
                    src->data[bsAdrY + col + 2] = src->data[bsAdrY + (src->width - col - 1) + 2];
                    src->data[bsAdrY + (src->width - col - 1) + 2] = buff;
                }
            }
        }
        break;
    }

    return ENG_SUCCESS;
}

const struct pixelEngine PixelEngine = {
	smooth : smooth3x3,
    smooth2D : smooth2D3x3,
    resize : resize,
    rotate : rotate,
    flip : flip,
};
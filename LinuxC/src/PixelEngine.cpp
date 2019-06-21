#include "include/PixelEngine.h"

namespace peg
{

namespace detail
{

template <typename T>
T clamp(const T min_value, const T max_value, const T value)
{
	return value < min_value ? min_value
							 : (value > max_value ? max_value : value);
}

} // namespace detail

PixelEngine::PixelEngine()
{

	f_state = ENG_READY;
}

PixelEngine::~PixelEngine()
{
	v_Pixels.clear();
}

/*

Will return A if B:

	ENG_READY		success
	ENG_RUNNING		aouther operation is running


An std::runtime_error is thrown if:
	pixels data size < height * width * sizePerPixel
	pixels sizePerPixel or height or width is 0
	Mask matrix buffer < n * n
	Mask order <=1
	Mask order > height or width
*/
EngineState PixelEngine::smooth(Pixels *src, const Matrix *mask, float factor)
{
	myassert(src->data.size() >= (src->height * src->width * src->sizePerPixel), "Bad pixels size. peg::PixelEngine::smooth");
	myassert(src->sizePerPixel && src->height && src->width, "Src pixels member sizePerPixel or height or width is 0. peg::PixelEngine::smooth");
	myassert(mask->x >= 1 && mask->y >= 1, "Bad matrix size. peg::PixelEngine::smooth");
	myassert((mask->y <= src->height && mask->x <= src->width), "Bad matrix size. peg::PixelEngine::smooth");
	myassert(mask->data.size() >= mask->x * mask->y, "Bad matrix buffer size. peg::PixelEngine::smooth");

	if (f_state != ENG_READY)
		return ENG_BUSSY;

	f_state = ENG_RUNNING;

	std::vector<std::uint8_t> buff(src->data.size() * src->sizePerPixel);

	std::copy(src->data.begin(), src->data.end(), buff.data());

	long double *pixelBuff = new long double[src->sizePerPixel];

	for (auto row = std::size_t{(std::size_t)(mask->y / 2 + mask->y % 2)}; row < (src->height - mask->y / 2); ++row)
	{
		for (auto col = std::size_t{(std::size_t)(mask->x / 2 + mask->x % 2)}; col < (src->width - mask->x / 2); ++col)
		{
			for (auto i = std::size_t{0}; i < src->sizePerPixel; ++i)
			{
				pixelBuff[i] = 0;
			}
			for (auto k = std::size_t{0}; k < mask->x * mask->y; ++k)
			{
				for (auto i = std::size_t{0}; i < src->sizePerPixel; ++i)
				{
					pixelBuff[i] += buff.data()[(col - mask->x / 2 + k % mask->x) + (row - mask->y / 2 + k / mask->y) * src->width] * mask->data.data()[k];
				}
			}
			for (auto i = std::size_t{0}; i < src->sizePerPixel; ++i)
			{
				src->data.data()[col + row * src->width + i] = pixelBuff[i] * factor;
			}
		}
	}

	f_state = ENG_READY;
	return ENG_READY;
}
/*

Will return A if B:

	ENG_READY		success
	ENG_RUNNING		aouther operation is running


An std::runtime_error is thrown if:
	pixels data size < height * width * sizePerPixel
	pixels sizePerPixel or height or width is 0
	Mask matrix* buffer < n * n
	Mask order <=1
	Mask order > height or width
*/
EngineState PixelEngine::smooth2D(Pixels *src, const Matrix *mask1, const Matrix *mask2, float factor1, float factor2)
{
	myassert(src->data.size() >= (src->height * src->width * src->sizePerPixel), "Bad pixels size. peg::PixelEngine::smooth2D");
	myassert(src->sizePerPixel && src->height && src->width, "Src pixels member sizePerPixel or height or width is 0. peg::PixelEngine::smooth2D");
	myassert(mask1->x >= 1 && mask1->y >= 1, "Bad matrix size. peg::PixelEngine::smooth2D");
	myassert((mask1->y <= src->height && mask1->x <= src->width), "Bad matrix size. peg::PixelEngine::smooth2D");
	myassert(mask1->data.size() >= mask1->x * mask1->y, "Bad matrix buffer size. peg::PixelEngine::smooth2D");

	if (f_state != ENG_READY)
		return ENG_BUSSY;

	f_state = ENG_RUNNING;

	std::vector<std::uint8_t> buff(src->data.size() * src->sizePerPixel);

	std::copy(src->data.begin(), src->data.end(), buff.data());

	long double *pixelBuff1 = new long double[src->sizePerPixel];
	long double *pixelBuff2 = new long double[src->sizePerPixel];
	for (auto row = std::size_t{(std::size_t)(mask1->y / 2 + mask1->y % 2)}; row < (src->height - mask1->y / 2); ++row)
	{
		for (auto col = std::size_t{(std::size_t)(mask1->x / 2 + mask1->x % 2)}; col < (src->width - mask1->x / 2); ++col)
		{
			for (auto i = std::size_t{0}; i < src->sizePerPixel; ++i)
			{
				pixelBuff1[i] = 0;
				pixelBuff2[i] = 0;
			}
			for (auto k = std::size_t{0}; k < mask1->x * mask1->y; ++k)
			{
				for (auto i = std::size_t{0}; i < src->sizePerPixel; ++i)
				{
					pixelBuff1[i] += buff.data()[(col - mask1->x / 2 + k % mask1->x) + (row - mask1->y / 2 + k / mask1->y) * src->width] * mask1->data.data()[k];
					pixelBuff2[i] += buff.data()[(col - mask2->x / 2 + k % mask2->x) + (row - mask2->y / 2 + k / mask2->y) * src->width] * mask2->data.data()[k];
				}
			}
			for (auto i = std::size_t{0}; i < src->sizePerPixel; ++i)
			{
				pixelBuff1[i] *= factor1;
				pixelBuff2[i] *= factor2;
				src->data.data()[col + row * src->width + i] = sqrtf(pixelBuff1[i] * pixelBuff1[i] + pixelBuff2[i] * pixelBuff2[i]);
			}
		}
	}

	f_state = ENG_READY;
	return ENG_READY;
}

/*
Will resize pixels

Attention:it will change the size of Pixels->data

Mode: 0: linear

An std::runtime_error is thrown if:
	pixels data size < height * width * sizePerPixel
	pixels sizePerPixel or height or width is 0
*/
EngineState PixelEngine::resize(Pixels *src, std::uint16_t newWidth, std::uint16_t newHeight, std::uint8_t mode)
{
	myassert(newWidth && newHeight, "New Width and Height should't be 0. peg::PixelEngine::resize");
	myassert(src->sizePerPixel && src->height && src->width, "Src pixels member sizePerPixel or height or width is 0. peg::PixelEngine::resize");
	myassert(src->data.size() >= (src->height * src->width * src->sizePerPixel), "Bad pixels size. peg::PixelEngine::resize");

	if (newWidth == src->width && newHeight == src->height)
		return ENG_READY;
	if (f_state != ENG_READY)
		return ENG_BUSSY;

	f_state = ENG_RUNNING;

	bool b_x2Big = newWidth > src->width ? true : false;
	bool b_y2Big = newHeight > src->height ? true : false;

	std::uint16_t x_big = newWidth > src->width ? newWidth : src->width;
	std::uint16_t x_small = newWidth < src->width ? newWidth : src->width;

	std::uint16_t y_big = newHeight > src->height ? newHeight : src->height;
	std::uint16_t y_small = newHeight < src->height ? newHeight : src->height;

	std::uint16_t x_step = x_big / x_small + (x_big % x_small == 0 ? 0 : 1);
	std::uint16_t y_step = y_big / y_small + (y_big % y_small == 0 ? 0 : 1);

	std::vector<std::uint8_t> buff(newHeight * newWidth * src->sizePerPixel);

	for (auto row = std::size_t{0}; row < newHeight; row += b_y2Big ? y_step : 1)
	{
		for (auto col = std::size_t{0}; col < newWidth - 1; ++col)
		{
			if (b_x2Big)
			{
				// amplification
				for (auto x = std::size_t{0}; x < x_step; ++x)
				{
					for (auto i = std::size_t{0}; i < src->sizePerPixel; ++i)
					{
						buff.data()[row * newWidth + col + i] =
							src->data.data()[col * x_step + (b_y2Big ? row / y_step : row * y_step) * src->width + i] +
							(src->data.data()[col * x_step + (b_y2Big ? row / y_step : row * y_step) * src->width + i] -
							 src->data.data()[(col + 1) * x_step + (b_y2Big ? row / y_step : row * y_step) * src->width + i]) *
								x / x_step;
					}
				}
			}
			else
			{
				//narrow
				for (auto i = std::size_t{0}; i < src->sizePerPixel; ++i)
				{
					buff.data()[row * newWidth + col + i] = src->data.data()[col * x_step + (b_y2Big ? row / y_step : row * y_step) * src->width + i];
				}
			}

			if (b_y2Big && row > 0)
			{
				// amplification
				for (auto y = std::size_t{1}; y < y_step; ++y)
				{
					for (auto i = std::size_t{0}; i < src->sizePerPixel; ++i)
					{
						buff.data()[(row + y - y_step) * newWidth + col + i] =
							buff.data()[(row - y_step) * newWidth + col + i] +
							(buff.data()[(row - y_step) * newWidth + col + i] -
							 buff.data()[(row)*newWidth + col + i]) *
								y / y_step;
					}
				}
			}
		}
	}

	src->data.resize(newHeight * newWidth * src->sizePerPixel);
	std::copy(buff.begin(), buff.end(), src->data.data());
	src->height = newHeight;
	src->width = newWidth;

	f_state = ENG_READY;
	return ENG_READY;
}

/*
Assume the origin is in the upper left corner of the coordinate system

origin
	+---------------------> x+
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

	Origin image's col row -> new image's dx dy is:

	dX=(height - row)*cos(a+90)+col*cos(a)
	dY=row*sin(a+90)+col*sin(a)
	
An std::runtime_error is thrown if:
	pixels data size < height * width * sizePerPixel
	pixels sizePerPixel or height or width is 0

*/
EngineState PixelEngine::rotate(Pixels *src, myfloat angle, std::uint8_t mode)
{
	myassert(src->sizePerPixel && src->height && src->width, "Src pixels member sizePerPixel or height or width is 0. peg::PixelEngine::rotate");
	myassert(src->data.size() >= (src->height * src->width * src->sizePerPixel), "Bad pixels size. peg::PixelEngine::rotate");

	if (f_state != ENG_READY)
		return ENG_BUSSY;

	f_state = ENG_RUNNING;
	double angleR = ((abs(angle) - int(abs(angle) / 90) * 90) * 3.14 / 180);
	//double angleR = angle  * 3.14 / 180;
	bool b_change;
	switch ((int(angle) / 90) % 4)
	{
	case 0:
	case 2:
	default:
		b_change = false;
		break;
	case 1:
	case 3:
		b_change = true;
		break;
	}
	//New x = height*cos(a+90)+width*cos(a)
	std::uint16_t x = src->height * abs(sin(angleR)) + src->width * abs(cos(angleR));
	//New y = height*sin(a+90)+width*sin(a)
	std::uint16_t y = src->height * abs(cos(angleR)) + src->width * abs(sin(angleR));

	std::uint16_t dx = 0, dy = 0;
	std::vector<std::uint8_t> buff(x * y * src->sizePerPixel);

	for (auto row = std::size_t{0}; row < src->height; ++row)
	{
		for (auto col = std::size_t{0}; col < src->width; ++col)
		{
			dy = row * cos(angleR) + col * sin(angleR);
			dx = (src->height - row) * sin(angleR) + col * cos(angleR);
			/*
			dx = b_change ? row * cos(angleR) + col * sin(angleR) :
				((src->height - row)*sin(angleR) + col * cos(angleR));
			dy = b_change ? ((src->height - row)*sin(angleR) + col * cos(angleR)):
				(row * cos(angleR) + col * sin(angleR));*/
			for (auto i = std::size_t{0}; i < src->sizePerPixel; ++i)
			{
				buff.data()[b_change ? (dx * y + y - dy + i) : (dx + dy * x + i)] = src->data.data()[row * src->width + col + i];
			}
		}
	}

	src->data.resize(x * y * src->sizePerPixel);
	std::copy(buff.begin(), buff.end(), src->data.data());
	src->height = b_change ? x : y;
	src->width = b_change ? y : x;

	return f_state = ENG_READY;
}

/*
Will flip pixels

Mode: 0: flip vertically
	  1: flip horizontally

selectLine: 0: Flip the entire image (Default)
			N: Flip the first N row/col


An std::runtime_error is thrown if:
	pixels data size < height * width * sizePerPixel
	pixels sizePerPixel or height or width is 0
*/

EngineState PixelEngine::flip(Pixels *src, std::uint8_t mode, std::uint16_t selectLine = 0)
{
	myassert(src->sizePerPixel && src->height && src->width, "Src pixels member sizePerPixel or height or width is 0. peg::PixelEngine::flip");
	myassert(src->data.size() >= (src->height * src->width * src->sizePerPixel), "Bad pixels size. peg::PixelEngine::flip");

	if (f_state != ENG_READY)
		return ENG_BUSSY;

	f_state = ENG_RUNNING;
	std::vector<std::uint8_t> buff(src->sizePerPixel);
	switch (mode)
	{
	case 0:
		// vertically
		for (auto row = std::size_t{0}; row < src->height / 2 && row < selectLine; ++row)
			for (auto col = std::size_t{0}; col < src->width; ++col)
				for (auto i = std::size_t{0}; i < src->sizePerPixel; ++i)
				{
					//std::swap(src->data.data()[(row)*src->width + col + i], src->data.data()[(src->height - row-1) * src->width + col + i]);
					buff.data()[i] = src->data.data()[row * src->width + col + i];
					src->data.data()[row * src->width + col + i] = src->data.data()[(src->height - row - 1) * src->width + col + i];
					src->data.data()[(src->height - row - 1) * src->width + col + i] = buff.data()[i];
				}
		break;
	default:
		// horizontally
		for (auto row = std::size_t{0}; row < src->height; ++row)
			for (auto col = std::size_t{0}; col < src->width / 2 && col < selectLine; ++col)
				for (auto i = std::size_t{0}; i < src->sizePerPixel; ++i)
				{
					//std::swap(src->data.data()[row * src->width + col + i], src->data.data()[row * src->width + (src->width - col) + i]);
					buff.data()[i] = src->data.data()[row * src->width + col + i];
					src->data.data()[row * src->width + col + i] = src->data.data()[row * src->width + (src->width - col) + i];
					src->data.data()[row * src->width + (src->width - col) + i] = buff.data()[i];
				}

		break;
	}

	return f_state = ENG_READY;
}

EngineState PixelEngine::HOG(
	Pixels const *src, 
	Pixels *hog, 
	Matrix *mX, Matrix *mY, 
	std::uint16_t startX, std::uint16_t startY, 
	std::uint16_t endX, std::uint16_t endY, 
	std::size_t particle, 
	bool isWeighted)
{
	myassert(src->sizePerPixel && src->height && src->width, "Src pixels member sizePerPixel or height or width is 0. peg::PixelEngine::HOG");
	myassert(src->data.size() >= (src->height * src->width * src->sizePerPixel), "Bad pixels size. peg::PixelEngine::HOG");
	myassert(mX && mX->x >= 1 && mX->y >= 1 && mX->y <= src->height && mX->x <= src->width, "Bad matrix size. peg::PixelEngine::HOG");
	myassert(mX->data.size() >= mX->x * mX->y, "Bad matrix buffer size. peg::PixelEngine::HOG");
	myassert(mY && mY->x >= 1 && mY->y >= 1 && mY->y <= src->height && mY->x <= src->width, "Bad matrix size. peg::PixelEngine::HOG");
	myassert(mY->data.size() >= mY->x * mY->y, "Bad matrix buffer size. peg::PixelEngine::HOG");
	myassert(mY->x == mX->y && mY->y == mX->x, "Matrix asymmetry. peg::PixelEngine::HOG");
	if (f_state != ENG_READY)
		return ENG_BUSSY;
	f_state = ENG_RUNNING;

	particle = particle < 2 ? 2 : particle;
	hog->width = particle;
	hog->height = 1;
	hog->sizePerPixel = 1;
	if (PixelsInit(hog) != ENG_SUCCESS)
	{
		f_state = ENG_READY;
		return ENG_MEMERY_INSUFFICIENT;
	}
	std::size_t edgeX = mX->x > mY->x ? mX->x : mY->x;
	std::size_t edgeY = mX->y > mY->y ? mX->y : mY->y;
	std::size_t hit, subhit;
	std::double_t cx = 0, cy = 0, ori = 0, g = 0;
	myfloat dorg = 360 / particle;
	myfloat d2org = dorg / 2;
	//std::vector<std::double_t> buffx, buffy;
	startX = startX < src->width - edgeX / 2 ? startX : 0;
	startY = startY < src->height - edgeY / 2 ? startX : 0;
	endX = endX < edgeX / 2 ? edgeX : src->width;
	endY = endY < edgeY / 2 ? edgeY : src->height;

	for (std::size_t y = startY + edgeY / 2; y < endY - edgeY / 2; y++)
	{
		for (std::size_t x = startX + edgeX / 2; x < endX - edgeX / 2; x++)
		{
			for (std::size_t y1 = 0; y1 < mX->y; y1++)
			{
				for (std::size_t x1 = 0; x1 < mX->x; x1++)
				{
					for (std::size_t k = 0; k < src->sizePerPixel; k++)
					{
						cx += src->data.data()[(x - mX->x / 2 + x1) + (y - mX->y / 2 + y1) * src->width + k] * mX->data.data()[y1 * mX->x + x1];
					}
				}
				cx /= src->sizePerPixel * mX->x;
			}
			cx /= mX->y;

			for (std::size_t y1 = 0; y1 < mY->y; y1++)
			{
				for (std::size_t x1 = 0; x1 < mY->x; x1++)
				{
					for (std::size_t k = 0; k < src->sizePerPixel; k++)
					{
						cy += src->data.data()[(x - mY->x / 2 + x1) + (y - mY->y / 2 + y1) * src->width + k] * mY->data.data()[y1 * mY->x + x1];
					}
				}
				cy /= src->sizePerPixel * mY->x;
			}
			cy /= mY->y;
			g = sqrt(cx * cx + cy * cy);
			ori = atan2(cy, cx) * 180 / 3.14159f + 180;
			hit = ((std::uint16_t)ori) / ((std::uint16_t)dorg);
			subhit = ((std::uint16_t)ori) / ((std::uint16_t)d2org);
			hit = hit == particle ? hit - 1 : hit;
			subhit = subhit == 2 * particle ? subhit - 1 : subhit;
			if (isWeighted)
			{
				if (subhit % 1)
				{
					//uper
					hog->data.data()[hit] += g * (ori - d2org * subhit) / d2org;
					hog->data.data()[hit+1 == particle ? 0: hit+1] += g * (1- (ori - d2org * subhit) / d2org);
				}
				else
				{
					//lower
					hog->data.data()[hit] += g * (ori - d2org * subhit) / d2org;
					hog->data.data()[hit == 0 ? particle-1 : hit-1] += g * (1- (ori - d2org * subhit) / d2org);
				}
				g *(ori - (360 / particle) * hit);
			}
			else
			{
				hog->data.data()[hit] += 1;
			}
		}
	}

	return f_state = ENG_READY;
}

} // namespace peg

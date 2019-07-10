#include "include/PixelEngine.h"

#ifdef NDEBUG
#define myassert(expression, mes) \
	if (!(expression))            \
	return peg::ENG_ERR
#else
#define myassert(expression, mes) assert(expression &&mes)
#endif // NDEBUG


namespace peg
{

namespace detail
{

template <typename T, typename B>
T clamp(const T min_value, const T max_value, const B value)
{
	return value < min_value ? min_value
							 : (value > max_value ? max_value : value);
}
template <typename T>
inline T abs(const T value)
{
	return value > 0 ? value : ((T)-1) * value;
}

} // namespace detail

PixelEngine::PixelEngine(EngineState init)
{
#ifdef CUDA_CODE_COMPILE
	if(init==ENG_CUDA_READY){
		device_ = -1;
		cudaGetDevice(&device_);
		cudaGetDeviceProperties(&prop_, device_);

		if (!(prop_.major > 3 || (prop_.major == 3 && prop_.minor >= 5)))
		{
			std::cout << "GPU"<< device_ << "-" << prop_.name << "does not support CUDA Dynamic Parallelism." << std::endl;
			device_ = -1;
			custate_ = ENG_CUDA_NOT_SUPPORT;
		}
		else if(init==ENG_CUDA_READY){
			std::cout << "Use GPU device:" << device_ << ": " << prop_.name << std::endl;
			std::cout << "SM Count:" << prop_.multiProcessorCount << std::endl;
			std::cout << "Shared Memery Per Block:" << (prop_.sharedMemPerBlock / 1024) << " KB "<< std::endl;
			std::cout << "Max Threads Per Block:" << prop_.maxThreadsPerBlock << std::endl;
			std::cout << "Max Threads Per Multi Processor:" << prop_.maxThreadsPerMultiProcessor << std::endl;
			std::cout << "Max Threads Bunch:" << prop_.maxThreadsPerMultiProcessor / 32 << std::endl;
			custate_ = ENG_CUDA_READY;
		}
	}

#endif // CUDA_CODE_COMPILE

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

Mask is:
				 [m  m  m]
		faceor * [m  m  m]
				 [m  m  m]

Return peg::ENG_ERR if:
	pixels data size < height * width * sizePerPixel
	pixels sizePerPixel or height or width is 0
	Mask matrix buffer < n * n
	Mask order <=1
	Mask order > height or width
*/
EngineState PixelEngine::smooth(Pixels & src, const Matrix & mask, float_t factor)
{
	myassert(src.sizePerPixel && src.sizePerPixel<=3 && src.height && src.width, "Src pixels member sizePerPixel or height or width is 0. peg::PixelEngine::smooth");
	myassert(src.data.size() >= (src.height * src.width * src.sizePerPixel), "Bad pixels size. peg::PixelEngine::smooth");
	myassert(mask.x >= 1 && mask.y >= 1, "Bad matrix size. peg::PixelEngine::smooth");
	myassert((mask.y <= src.height && mask.x <= src.width), "Bad matrix size. peg::PixelEngine::smooth");
	myassert(mask.data.size() >= mask.x * mask.y, "Bad matrix buffer size. peg::PixelEngine::smooth");

	if (f_state != ENG_READY)
		return ENG_BUSSY;

	f_state = ENG_RUNNING;


#ifdef CUDA_CODE_COMPILE
	if (device_ != -1 && custate_ == ENG_CUDA_READY) {
		std::uint8_t *buff,*out;

		cusmooth(src.data.data(), src.width, src.height, src.sizePerPixel, mask.x , mask.y, mask.data.data(), factor,device_,&prop_);

		f_state = ENG_READY;
		return ENG_READY;
	}
	
#endif // CUDA_CODE_COMPILE

	std::vector<std::uint8_t> buff = src.data;
	float_t pixelBuff[3] = {};

	// Only handle the central area
	for (size_t row = (mask.y / 2 + mask.y % 2); row < (src.height - mask.y / 2); row += src.sizePerPixel)
	{
		for (size_t col = (mask.x / 2 + mask.x % 2); col < (src.width - mask.x / 2); col += src.sizePerPixel)
		{
			pixelBuff[0] = pixelBuff[1] = pixelBuff[2] = 0;
			for (auto k = size_t{ 0 }; k < mask.x * mask.y; ++k)
			{
				pixelBuff[0] += buff[(col - mask.x / 2 + k % mask.x) + (row - mask.y / 2 + k / mask.y) * src.width + 0] * mask.data[k];
				if (src.sizePerPixel > 1)
					pixelBuff[1] += buff[(col - mask.x / 2 + k % mask.x) + (row - mask.y / 2 + k / mask.y) * src.width + 1] * mask.data[k];
				if (src.sizePerPixel > 2)
					pixelBuff[2] += buff[(col - mask.x / 2 + k % mask.x) + (row - mask.y / 2 + k / mask.y) * src.width + 2] * mask.data[k];
			}
			src.data[col + row * src.width + 0] = pixelBuff[0] * factor;
			if (src.sizePerPixel > 1)
				src.data[col + row * src.width + 1] = pixelBuff[1] * factor;
			if (src.sizePerPixel > 2)
				src.data[col + row * src.width + 2] = pixelBuff[2] * factor;
		}
	}

	f_state = ENG_READY;
	return ENG_READY;
}
/*

Will return A if B:

	ENG_READY		success
	ENG_RUNNING		aouther operation is running

Mask is:
				  [m  m  m]
		faceorX * [m  m  m]
				  [m  m  m]

An std::runtime_error is thrown if:
	pixels data size < height * width * sizePerPixel
	pixels sizePerPixel or height or width is 0
	Mask matrix* buffer < n * n
	Mask order <=1
	Mask order > height or width
*/
EngineState PixelEngine::smooth2D(Pixels & src, const Matrix & mask1, const Matrix & mask2, float_t factor1, float_t factor2)
{
	myassert(src.sizePerPixel && src.sizePerPixel <= 3 && src.height && src.width, "Src pixels member sizePerPixel or height or width is 0. peg::PixelEngine::smooth2D");
	myassert(src.data.size() >= (src.height * src.width * src.sizePerPixel), "Bad pixels size. peg::PixelEngine::smooth2D");
	myassert(mask1.x >= 1 && mask1.y >= 1, "Bad matrix size. peg::PixelEngine::smooth2D");
	myassert((mask1.y <= src.height && mask1.x <= src.width), "Bad matrix size. peg::PixelEngine::smooth2D");
	myassert(mask1.data.size() >= mask1.x * mask1.y, "Bad matrix buffer size. peg::PixelEngine::smooth2D");

	if (f_state != ENG_READY)
		return ENG_BUSSY;

	f_state = ENG_RUNNING;

	std::vector<std::uint8_t> buff = src.data;
	float_t pixelBuff1[3] = {};
	float_t pixelBuff2[3] = {};

	for (size_t row = (mask1.y / 2 + mask1.y % 2); row < (src.height - mask1.y / 2); row += src.sizePerPixel)
	{
		for (size_t col = (mask1.x / 2 + mask1.x % 2); col < (src.width - mask1.x / 2); col += src.sizePerPixel)
		{
			pixelBuff1[0] = pixelBuff1[1] = pixelBuff1[2] = 0;
			pixelBuff2[0] = pixelBuff2[1] = pixelBuff2[2] = 0;
			for (auto k = size_t{ 0 }; k < mask1.x * mask1.y; ++k)
			{
				pixelBuff1[0] += buff[(col - mask1.x / 2 + k % mask1.x) + (row - mask1.y / 2 + k / mask1.y) * src.width + 0] * mask1.data[k];
				pixelBuff2[0] += buff[(col - mask2.x / 2 + k % mask2.x) + (row - mask2.y / 2 + k / mask2.y) * src.width + 0] * mask2.data[k];
				if (src.sizePerPixel > 1) {
					pixelBuff1[1] += buff[(col - mask1.x / 2 + k % mask1.x) + (row - mask1.y / 2 + k / mask1.y) * src.width + 1] * mask1.data[k];
					pixelBuff2[1] += buff[(col - mask2.x / 2 + k % mask2.x) + (row - mask2.y / 2 + k / mask2.y) * src.width + 1] * mask2.data[k];
				}
				if (src.sizePerPixel > 2) {
					pixelBuff1[2] += buff[(col - mask1.x / 2 + k % mask1.x) + (row - mask1.y / 2 + k / mask1.y) * src.width + 2] * mask1.data[k];
					pixelBuff2[2] += buff[(col - mask2.x / 2 + k % mask2.x) + (row - mask2.y / 2 + k / mask2.y) * src.width + 2] * mask2.data[k];
				}
			}			
			pixelBuff1[0] *= factor1; pixelBuff1[1] *= factor1; pixelBuff1[2] *= factor1;
			pixelBuff2[0] *= factor2; pixelBuff2[1] *= factor2; pixelBuff2[2] *= factor2;
			src.data[col + row * src.width + 0] = detail::clamp(0, 255, sqrtf(pixelBuff1[0] * pixelBuff1[0] + pixelBuff2[0] * pixelBuff2[0]));
			if (src.sizePerPixel > 1)
				src.data[col + row * src.width + 1] = detail::clamp(0, 255, sqrtf(pixelBuff1[1] * pixelBuff1[1] + pixelBuff2[1] * pixelBuff2[1]));
			if (src.sizePerPixel > 2)
				src.data[col + row * src.width + 2] = detail::clamp(0, 255, sqrtf(pixelBuff1[2] * pixelBuff1[2] + pixelBuff2[2] * pixelBuff2[2]));
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
EngineState PixelEngine::resize(Pixels & src, std::uint16_t newWidth, std::uint16_t newHeight, std::uint8_t mode)
{
	myassert(src.sizePerPixel && src.sizePerPixel <= 3 && src.height && src.width, "Src pixels member sizePerPixel or height or width is 0. peg::PixelEngine::resize");
	myassert(newWidth && newHeight, "New Width and Height should't be 0. peg::PixelEngine::resize");
	myassert(src.data.size() >= (src.height * src.width * src.sizePerPixel), "Bad pixels size. peg::PixelEngine::resize");

	if (newWidth == src.width && newHeight == src.height)
		return ENG_READY;
	if (f_state != ENG_READY)
		return ENG_BUSSY;

	f_state = ENG_RUNNING;

	bool b_x2Big = newWidth > src.width ? true : false;
	bool b_y2Big = newHeight > src.height ? true : false;

	std::uint16_t x_big = newWidth > src.width ? newWidth : src.width;
	std::uint16_t x_small = newWidth < src.width ? newWidth : src.width;

	std::uint16_t y_big = newHeight > src.height ? newHeight : src.height;
	std::uint16_t y_small = newHeight < src.height ? newHeight : src.height;

	std::uint16_t x_step = x_big / x_small + (x_big % x_small == 0 ? 0 : 1);
	std::uint16_t y_step = y_big / y_small + (y_big % y_small == 0 ? 0 : 1);

	std::vector<std::uint8_t> buff(newHeight * newWidth * src.sizePerPixel);

	for (size_t row = 0; row < newHeight; row += (b_y2Big ? y_step  : 1 )*src.sizePerPixel)
	{
		for (size_t col = 0; col < newWidth - 1; col += (b_x2Big ? x_step : 1)*src.sizePerPixel)
		{
			if (b_x2Big)
			{
				// amplification
				for (size_t x = 0; x < x_step; ++x)
				{
					buff[row * newWidth + col + x + 0] = src.data[col * x_step + (b_y2Big ? row / y_step : row * y_step) * src.width + 0] +
													(src.data[col * x_step + (b_y2Big ? row / y_step : row * y_step) * src.width + 0] -
													src.data[(col + 1) * x_step + (b_y2Big ? row / y_step : row * y_step) * src.width + 0]) *
													x / x_step;
					if(src.sizePerPixel > 1)
						buff[row * newWidth + col + x + 1] = src.data[col * x_step + (b_y2Big ? row / y_step : row * y_step) * src.width + 1] +
														(src.data[col * x_step + (b_y2Big ? row / y_step : row * y_step) * src.width + 1] -
														src.data[(col + 1) * x_step + (b_y2Big ? row / y_step : row * y_step) * src.width + 1]) *
														x / x_step;
					if (src.sizePerPixel > 2)
						buff[row * newWidth + col + x + 2] = src.data[col * x_step + (b_y2Big ? row / y_step : row * y_step) * src.width + 2] +
														(src.data[col * x_step + (b_y2Big ? row / y_step : row * y_step) * src.width + 2] -
														src.data[(col + 1) * x_step + (b_y2Big ? row / y_step : row * y_step) * src.width + 2]) *
														x / x_step;
				}
			}
			else
			{
				//narrow
				buff[row * newWidth + col + 0] = src.data[col * x_step + (b_y2Big ? row / y_step : row * y_step) * src.width + 0];
				if (src.sizePerPixel > 1)buff[row * newWidth + col + 1] = src.data[col * x_step + (b_y2Big ? row / y_step : row * y_step) * src.width + 1];
				if (src.sizePerPixel > 2)buff[row * newWidth + col + 2] = src.data[col * x_step + (b_y2Big ? row / y_step : row * y_step) * src.width + 2];
			}

			if (b_y2Big && row > 0)
			{
				// amplification
				for (size_t y = 1; y < y_step; ++y)
				{
					buff[(row + y - y_step) * newWidth + col + 0] =
											buff[(row - y_step) * newWidth + col + 0] +
											(buff[(row - y_step) * newWidth + col + 0] -
											buff[(row)* newWidth + col + 0]) *
											y / y_step;
					if (src.sizePerPixel > 1)buff[(row + y - y_step) * newWidth + col + 1] =
											buff[(row - y_step) * newWidth + col + 1] +
											(buff[(row - y_step) * newWidth + col + 1] -
											buff[(row)* newWidth + col + 1]) *
											y / y_step;
					if (src.sizePerPixel > 2)buff[(row + y - y_step) * newWidth + col + 2] =
											buff[(row - y_step) * newWidth + col + 2] +
											(buff[(row - y_step) * newWidth + col + 2] -
											buff[(row)* newWidth + col + 2]) *
											y / y_step;
				}
			}
		}
	}

	src.data.clear();
	src.data = std::move(buff);

	src.height = newHeight;
	src.width = newWidth;

	f_state = ENG_READY;
	return ENG_READY;
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
	
An std::runtime_error is thrown if:
	pixels data size < height * width * sizePerPixel
	pixels sizePerPixel or height or width is 0

*/
EngineState PixelEngine::rotate(Pixels & src, float_t angle, std::uint8_t mode)
{
	myassert(src.sizePerPixel && src.sizePerPixel <= 3 && src.height && src.width, "Src pixels member sizePerPixel or height or width is 0. peg::PixelEngine::rotate");
	myassert(src.data.size() >= (src.height * src.width * src.sizePerPixel), "Bad pixels size. peg::PixelEngine::rotate");

	if (f_state != ENG_READY)
		return ENG_BUSSY;

	f_state = ENG_RUNNING;
	float_t angleR = ((detail::abs(angle) - (int)(detail::abs(angle) / 90) * 90) * 3.14 / 180);

	bool b_change;
	// Decide whether to exchange XY
	switch ((int(angle) / 90) % 4)
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
	float_t cosR=cos(angleR),sinR=sin(angleR);

	std::uint16_t x = src.height * detail::abs(sinR) + src.width * detail::abs(cosR);
	std::uint16_t y = src.height * detail::abs(cosR) + src.width * detail::abs(sinR);
	std::uint16_t dx = 0, dy = 0;
	std::vector<std::uint8_t> buff(x * y * src.sizePerPixel);

	for (size_t row = 0; row < src.height; row+=src.sizePerPixel)
	{
		for (size_t col = 0; col < src.width; col += src.sizePerPixel)
		{
			dy = row * cosR + col * sinR;
			dx = (src.height - row) * sinR + col * cosR;
			/*
			dx = b_change ? row * cos(angleR) + col * sin(angleR) :
				((src.height - row)*sin(angleR) + col * cos(angleR));
			dy = b_change ? ((src.height - row)*sin(angleR) + col * cos(angleR)):
				(row * cos(angleR) + col * sin(angleR));*/
			buff[b_change ? (dx * y + y - dy + 0) : (dx + dy * x + 0)] = src.data[row * src.width + col + 0];
			if (src.sizePerPixel > 1)buff[b_change ? (dx * y + y - dy + 1) : (dx + dy * x + 1)] = src.data[row * src.width + col + 1];
			if (src.sizePerPixel > 2)buff[b_change ? (dx * y + y - dy + 2) : (dx + dy * x + 2)] = src.data[row * src.width + col + 2];
		}
	}

	src.data.clear();
	src.data = std::move(buff);
	src.height = b_change ? x : y;
	src.width = b_change ? y : x;

	return f_state = ENG_READY;
}
EngineState PixelEngine::rotate2(Pixels& src, float_t angle, std::uint8_t mode) {
	float_t angleR = (angle * 3.14 / 180);
	float_t cosR = cos(angleR), sinR = sin(angleR);
	float_t offsetX = src.width / 2, offsetY = src.height / 2;
	std::uint16_t x = src.height * detail::abs(sinR) + src.width * detail::abs(cosR);
	std::uint16_t y = src.height * detail::abs(cosR) + src.width * detail::abs(sinR);

	float_t moveM[3][3] = { cosR ,sinR,(-1) * offsetX * cosR - offsetY * sinR + x/2*cosR + y/2*sinR
		,(-1) * sinR,cosR,offsetX * sinR - offsetY * cosR - x/2*sinR + y/2*cosR,
		0,0,1 };
	std::int16_t sx, sy;
	std::vector<std::uint8_t> buff(x * y * src.sizePerPixel);
	for (size_t row = 0; row < y; row += src.sizePerPixel)
	{
		for (size_t col = 0; col <x; col += src.sizePerPixel)
		{
			sx = (col) * moveM[0][0]+ (row) * moveM[0][1]+ moveM[0][2];
			sy = (col) * moveM[1][0] + (row) * moveM[1][1] + moveM[1][2];
			if (sx>=0&&sx<src.width&&sy>=0&&sy<src.height) {
				buff[row * y + col +0] = src.data[sy * src.width + sx + 0];
				if (src.sizePerPixel > 1)buff[row * y + col + 1] = src.data[sy * src.width + sx + 1];
				if (src.sizePerPixel > 2)buff[row * y + col + 2] = src.data[sy * src.width + sx + 2];
			}
			else {
				buff[row * y + col + 0] = 0;
			}
		}
	}
	src.data.clear();
	src.data = std::move(buff);
	src.height = x;
	src.width = y;
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

EngineState PixelEngine::flip(Pixels & src, std::uint8_t mode, std::uint16_t selectLine = 0)
{
	myassert(src.sizePerPixel && src.sizePerPixel <= 3 && src.height && src.width, "Src pixels member sizePerPixel or height or width is 0. peg::PixelEngine::flip");
	myassert(src.data.size() >= (src.height * src.width * src.sizePerPixel), "Bad pixels size. peg::PixelEngine::flip");

	if (f_state != ENG_READY)
		return ENG_BUSSY;

	f_state = ENG_RUNNING;
	std::uint8_t buff;
	if (selectLine == 0)selectLine = mode? src.width / 2:src.height / 2;
	switch (mode)
	{
	case 0:
		// vertically
		for (size_t row = 0; row < src.height / 2 && row < selectLine; row+=src.sizePerPixel)
			for (size_t col = 0; col < src.width; col += src.sizePerPixel)
			{
				buff = src.data[row * src.width + col + 0];
				src.data[row * src.width + col + 0] = src.data[(src.height - row - 1) * src.width + col + 0];
				src.data[(src.height - row - 1) * src.width + col + 0] = buff;
				if (src.sizePerPixel > 1) {
					buff = src.data[row * src.width + col + 1];
					src.data[row * src.width + col + 1] = src.data[(src.height - row - 1) * src.width + col + 1];
					src.data[(src.height - row - 1) * src.width + col + 1] = buff;
				}
				if (src.sizePerPixel > 2) {
					buff = src.data[row * src.width + col + 2];
					src.data[row * src.width + col + 2] = src.data[(src.height - row - 1) * src.width + col + 2];
					src.data[(src.height - row - 1) * src.width + col + 2] = buff;
				}
			}
				
		break;
	default:
		// horizontally
		for (size_t row = 0; row < src.height; row += src.sizePerPixel)
			for (size_t col = 0; col < src.width / 2 && col < selectLine; col += src.sizePerPixel)
			{
				buff = src.data[row * src.width + col + 0];
				src.data[row * src.width + col + 0] = src.data[row * src.width + (src.width - col-1) + 0];
				src.data[row * src.width + (src.width - col-1) + 0] = buff;
				if (src.sizePerPixel > 1) {
					buff = src.data[row * src.width + col + 1];
					src.data[row * src.width + col + 1] = src.data[row * src.width + (src.width - col-1) + 1];
					src.data[row * src.width + (src.width - col-1) + 1] = buff;
				}
				if (src.sizePerPixel > 2) {
					buff = src.data[row * src.width + col + 2];
					src.data[row * src.width + col + 2] = src.data[row * src.width + (src.width - col-1) + 2];
					src.data[row * src.width + (src.width - col-1) + 2] = buff;
				}
			}
		break;
	}

	return f_state = ENG_READY;
}

EngineState PixelEngine::HOG(
	Pixels const & src, Pixels & hog, Matrix & mX, Matrix & mY, 
	std::uint16_t startX,
	std::uint16_t startY, 
	std::uint16_t endX,
	std::uint16_t endY,
	size_t particle, 
	bool isWeighted)
{
	myassert(src.sizePerPixel && src.sizePerPixel <= 3 && src.height && src.width, "Src pixels member sizePerPixel or height or width is 0. peg::PixelEngine::HOG");
	myassert(src.data.size() >= (src.height * src.width * src.sizePerPixel), "Bad pixels size. peg::PixelEngine::HOG");
	myassert(mX.x >= 1 && mX.y >= 1 && mX.y <= src.height && mX.x <= src.width, "Bad matrix size. peg::PixelEngine::HOG");
	myassert(mX.data.size() >= mX.x * mX.y, "Bad matrix buffer size. peg::PixelEngine::HOG");
	myassert(mY.x >= 1 && mY.y >= 1 && mY.y <= src.height && mY.x <= src.width, "Bad matrix size. peg::PixelEngine::HOG");
	myassert(mY.data.size() >= mY.x * mY.y, "Bad matrix buffer size. peg::PixelEngine::HOG");
	myassert(mY.x == mX.y && mY.y == mX.x, "Matrix asymmetry. peg::PixelEngine::HOG");
	if (f_state != ENG_READY)
		return ENG_BUSSY;
	f_state = ENG_RUNNING;

	particle = particle < 2 ? 2 : particle;
	hog.width = particle;
	hog.height = 1;
	hog.sizePerPixel = 1;
	if (PixelsInit(hog) != ENG_SUCCESS)
	{
		f_state = ENG_READY;
		return ENG_MEMERY_INSUFFICIENT;
	}
	// Choose MAX edge in two matrices
	size_t edgeX = mX.x > mY.x ? mX.x : mY.x;
	size_t edgeY = mX.y > mY.y ? mX.y : mY.y;
	size_t hit, subhit;
	float_t cx = 0, cy = 0, ori = 0, g = 0;
	float_t dorg = 360 / particle;
	float_t d2org = dorg / 2;
	// Check if the selection can accommodate the window
	startX = startX < (src.width - edgeX / 2) ? startX : 0;
	startY = startY < (src.height - edgeY / 2) ? startX : 0;
	// Check if the selection can accommodate the window
	endX = endX < edgeX / 2 ? edgeX : src.width;
	endY = endY < edgeY / 2 ? edgeY : src.height;

	for (size_t y = startY + edgeY / 2; y < endY - edgeY / 2; y+=src.sizePerPixel)
	{
		for (size_t x = startX + edgeX / 2; x < endX - edgeX / 2; x += src.sizePerPixel)
		{
			// Calculate single point X gradient value
			for (size_t y1 = 0; y1 < mX.y; y1++)
			{
				for (size_t x1 = 0; x1 < mX.x; x1++)
				{
					cx += src.data[(x - mX.x / 2 + x1) + (y - mX.y / 2 + y1) * src.width + 0] * mX.data[y1 * mX.x + x1];
					if (src.sizePerPixel > 1)cx += src.data[(x - mX.x / 2 + x1) + (y - mX.y / 2 + y1) * src.width + 1] * mX.data[y1 * mX.x + x1];
					if (src.sizePerPixel > 2)cx += src.data[(x - mX.x / 2 + x1) + (y - mX.y / 2 + y1) * src.width + 2] * mX.data[y1 * mX.x + x1];
				}
				cx /= src.sizePerPixel * mX.x;
			}
			cx /= mX.y;
			// Calculate single point Y gradient value
			for (size_t y1 = 0; y1 < mY.y; y1++)
			{
				for (size_t x1 = 0; x1 < mY.x; x1++)
				{
					cy += src.data[(x - mY.x / 2 + x1) + (y - mY.y / 2 + y1) * src.width + 0] * mY.data[y1 * mY.x + x1];
					if (src.sizePerPixel > 1)cy += src.data[(x - mY.x / 2 + x1) + (y - mY.y / 2 + y1) * src.width + 1] * mY.data[y1 * mY.x + x1];
					if (src.sizePerPixel > 1)cy += src.data[(x - mY.x / 2 + x1) + (y - mY.y / 2 + y1) * src.width + 2] * mY.data[y1 * mY.x + x1];
				}
				cy /= src.sizePerPixel * mY.x;
			}
			cy /= mY.y;
			// Calculate gradient value
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
					hog.data[hit] += g * (ori - d2org * subhit) / d2org;
					hog.data[hit+1 == particle ? 0: hit+1] += g * (1- (ori - d2org * subhit) / d2org);
				}
				else
				{
					//lower
					hog.data[hit] += g * (ori - d2org * subhit) / d2org;
					hog.data[hit == 0 ? particle-1 : hit-1] += g * (1- (ori - d2org * subhit) / d2org);
				}
				g *(ori - (360 / particle) * hit);
			}
			else
			{
				hog.data[hit] += 1;
			}
		}
	}

	return f_state = ENG_READY;
}

} // namespace peg

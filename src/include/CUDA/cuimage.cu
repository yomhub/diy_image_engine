#include "CUDA/cuimage.h"


#if defined(_WIN32) || defined(_WIN64)
#define myfloat std::float_t
#define mydouble std::double_t
#else
#define myfloat float
#define mydouble double
#endif

__global__  void __cusmooth(std::uint8_t* dst,
	std::uint16_t width,
	std::uint16_t height,
	std::uint8_t pixels,
	std::uint8_t* src,
	std::uint8_t mx,
	std::uint8_t my,
	mydouble* m,
	float factor
) {
	mydouble* pixelBuff = new mydouble[pixels];
	//int row = blockIdx.y * blockDim.y + threadIdx.y;
	//int col = blockIdx.x * blockDim.x + threadIdx.x;
	size_t y, x, i, k, dx, dy;
	for (y = my / 2 + my % 2; y < (height - my / 2); y++)
	{
		for (x = mx / 2 + mx % 2; x <= (width - mx / 2); x++) {
			for (i = 0; i < pixels; i++)
			{
				pixelBuff[i] = 0;
			}
			for (k = 0; k < mx * my; k++)
			{
				for (i = 0; i < pixels; i++)
				{
					pixelBuff[i] += src[(x - mx / 2 + k % mx) + (y - my / 2 + k / my) * width] * m[k];
				}
			}
			for (i = 0; i < pixels; i++)
			{
				dst[x + y * width + i] = pixelBuff[i] * factor;
			}
		}
	}
}

__global__  void cutest() {}

void cusmooth(std::uint8_t* dst,
	std::uint16_t width,
	std::uint16_t height,
	std::uint8_t pixels,
	std::uint8_t* src,
	std::uint8_t mx,
	std::uint8_t my,
	mydouble* m,
	float factor
) {


	__cusmooth <<<prop_.multiProcessorCount, prop_.maxThreadsPerBlock > src->height ? src->height : prop_.maxThreadsPerBlock >>> (
		out,src->width, src->height, src->sizePerPixel,buff,mask->x, mask->y, m,factor);

}
#include "CUDA/cuimage.h"


#if defined(_WIN32) || defined(_WIN64)
#define myfloat std::float_t
#define mydouble std::double_t
#else
#define myfloat float
#define mydouble double
#endif

// threadIdx.x=N,threadIdx.y=1 col index
// blockIdx.x=N,blockIdx.y=1 Row index

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
	
	size_t x = (mx/2)+threadIdx.x * threadIdx.y * ((width-(mx-1))/blockDim.x);
	size_t y = (my/2)+blockIdx.x * blockIdx.y * ((height-(my-1))/gridDim.x);
	size_t i, k, dx, dy;

	for (dy=0; dy < ((height-(my-1))/gridDim.x); ++dy)
	{
		for (dx=0; dx < ((width-(mx-1))/blockDim.x); ++dx) {
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

CUresult cusmooth(std::uint8_t* dst,
	std::uint16_t width,
	std::uint16_t height,
	std::uint8_t pixels,
	std::uint8_t* src,
	std::uint8_t mx,
	std::uint8_t my,
	mydouble* msrc,
	float factor,
	int device,
	cudaDeviceProp *prop_
) {
	CUresult _ret;
	if(device<=0){
		_ret=cudaGetDevice(&device);
		if(device<=0)return _ret;
		cudaGetDeviceProperties(prop_, device);
	}
	cudaSetDevice(device);

	std::uint8_t *buff,*out;
	mydouble* m;

	cudaMalloc(&buff, height * width * pixels);
	cudaMalloc(&out, height * width * pixels);
	cudaMalloc(&m, mx * my * sizeof(mydouble));
	cudaMemcpy(buff, src, height * width * sizePerPixel, cudaMemcpyHostToDevice);
	cudaMemcpy(m, msrc, mx * my * sizeof(mydouble), cudaMemcpyHostToDevice);

	// Each block handles integer line 
	// Each thread handles integer window
	__cusmooth <<<prop_->multiProcessorCount >= height-(my-1) ? height-(my-1) : prop_->multiProcessorCount, 
	prop_->maxThreadsPerBlock >= width-(mx-1)  ? width-(mx-1) : prop_.maxThreadsPerBlock >>> (
		out,src->width, src->height, src->sizePerPixel,buff,mask->x, mask->y, m,factor);
	cudaMemcpy(src,out,height * width * sizePerPixel, cudaMemcpyDeviceToHost);
	cudeFree(buff);
	cudeFree(out);
	cudeFree(m);
}
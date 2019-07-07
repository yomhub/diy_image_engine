#include "CUDA/cuimage.h"

// threadIdx.x=N,threadIdx.y=0 col index
// blockIdx.x=N,blockIdx.y=0 Row index

__global__  void __cusmooth(std::uint8_t* dst,
	std::uint16_t width,
	std::uint16_t height,
	std::uint8_t pixels,
	std::uint8_t* src,
	std::uint8_t mx,
	std::uint8_t my,
	float_t* m,
	float factor
) {
	float_t* pixelBuff = new float_t[pixels];
	
	size_t x = threadIdx.x * width / blockDim.x;
	size_t y = blockIdx.x * height / gridDim.x;
	size_t i, k, dx, dy;
	

	for (dy=0; dy < height / gridDim.x; dy+=pixels)
	{
		for (dx=0; dx < width / blockDim.x; dx+=pixels) {
			if (((x + dx) < mx / 2)||((y + dy)< my/2) || ((x + dx) > mx / 2) || ((y + dy) > my / 2)) {
				for (i = 0; i < pixels; i++)
				{
					dst[x + dx + (y + dy) * width + i] = src[x + dx + (y + dy) * width + i];
				}
			}
			else {
				for (i = 0; i < pixels; i++)
				{
					pixelBuff[i] = 0;
				}
				for (k = 0; k < mx * my; k++)
				{
					for (i = 0; i < pixels; i++)
					{
						pixelBuff[i] += src[(x + dx - mx / 2 + (k / my) % mx) + (y + dy - my / 2 + k / my) * width] * m[k];
					}
				}
				for (i = 0; i < pixels; i++)
				{
					dst[x + dx + (y + dy) * width + i] = pixelBuff[i] * factor;
				}
			}
		}
	}
}

cudaError cusmooth(
	std::uint8_t* src,
	std::uint16_t width,
	std::uint16_t height,
	std::uint8_t pixels,
	std::uint8_t mx,
	std::uint8_t my,
	const float_t* msrc,
	float factor,
	int device,
	cudaDeviceProp *prop_
) {
	cudaError _ret;
	if(device<0){
		_ret=cudaGetDevice(&device);
		if(device<0)return _ret;
		cudaGetDeviceProperties(prop_, device);
	}
	cudaSetDevice(device);

	std::uint8_t *buff,*out;
	float_t* m;

	_ret = cudaMalloc(&buff, height * width * pixels);
	_ret = cudaMalloc(&out, height * width * pixels);
	_ret = cudaMalloc(&m, mx * my * sizeof(float_t));
	_ret = cudaMemcpy(buff, src, height * width * pixels, cudaMemcpyHostToDevice);
	_ret = cudaMemcpy(m, msrc, mx * my * sizeof(float_t), cudaMemcpyHostToDevice);

	// Each block handles integer line 
	// Each thread handles integer window
	__cusmooth << < prop_->multiProcessorCount >= height ? height : prop_->multiProcessorCount,
		prop_->maxThreadsPerBlock >= width ? width : prop_->maxThreadsPerBlock >> > (
			out, width, height, pixels, buff, mx, my, m, factor);
	_ret = cudaDeviceSynchronize();
	_ret = cudaMemcpy(src,out,height * width * pixels, cudaMemcpyDeviceToHost);
	cudaFree(buff);
	cudaFree(out);
	cudaFree(m);
	return cudaSuccess;
}


__global__  void __hog(
	std::uint16_t width, std::uint16_t height, std::uint8_t pixels,
	std::uint8_t* src,
	std::uint16_t startX, std::uint16_t startY, 
	std::uint16_t endX, std::uint16_t endY,
	std::uint8_t hogx, std::uint8_t hogy,
	std::uint8_t* hog,
	std::size_t particle
) {
	float_t* pixelBuff = new float_t[pixels];
	
	size_t x = startX + threadIdx.x * width / blockDim.x;
	size_t y = startY + blockIdx.x * height / gridDim.x;
	size_t i, k, dx, dy;
	
	for (dy=0; (dy + y)< endY; ++dy)
	{
		for (dx=0; (dx + x) < endX; ++dx) {
			
		}
	}
}
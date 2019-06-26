#pragma once

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#if defined(_WIN32) || defined(_WIN64)
#define myfloat std::float_t
#define mydouble std::double_t
#else
#define myfloat float
#define mydouble double
#endif

cudaError cusmooth(
	std::uint8_t* src,
	std::uint16_t width,
	std::uint16_t height,
	std::uint8_t pixels,
	std::uint8_t mx,
	std::uint8_t my,
	const mydouble* msrc,
	float factor,
	int device,
	cudaDeviceProp* prop_
);
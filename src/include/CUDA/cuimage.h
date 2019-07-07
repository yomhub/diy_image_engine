#pragma once

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

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
	cudaDeviceProp* prop_
);
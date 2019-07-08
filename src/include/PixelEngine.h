#pragma once

//#define USING_MULTI_THREAD
//#defing USING_AMP

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cassert>
#include <math.h>
#define _CRT_SECURE_DEPRECATE_MEMORY
#include <memory.h>		//memcpy
#include <algorithm>    // std::swap
#include <stdint.h>

#ifdef CUDA_CODE_COMPILE
#include "CUDA/cuimage.h"
#define CUCHECK checkCudaErrors
#endif // CUDA_CODE_COMPILE

namespace peg{

enum EngineState
{
	ENG_READY = 0,
	ENG_RUNNING,
	ENG_SLEEP,
	ENG_SUCCESS,
	ENG_ERR,
	ENG_BUSSY,
	ENG_MEMERY_INSUFFICIENT,

	// For CUDA
	ENG_CUDA_NOT_SUPPORT,
	ENG_CUDA_READY,
	ENG_CUDA_BUSY,
	ENG_CUDA_ERR,
	ENG_CUDA_MEMERY_INSUFFICIENT,

};

enum ColorSpace {
	P_GR=0,
	P_RGB,
	P_RGBA,
};

struct Pixels
{
	std::uint16_t width;
	std::uint16_t height;
	std::uint8_t sizePerPixel;
	std::vector<std::uint8_t> data;
};

// Suppose this is a square matrix of order n
struct Matrix {
	std::uint8_t x;
	std::uint8_t y;
	std::vector<float_t> data;
};

inline EngineState PixelsInit(Pixels &p) {
	if (p.data.size() < p.height * p.width * p.sizePerPixel)p.data.resize(p.height * p.width * p.sizePerPixel);
	if (p.data.size() < p.height * p.width * p.sizePerPixel) return ENG_MEMERY_INSUFFICIENT;
	return ENG_SUCCESS;
}

class PixelEngine
{
public:
	PixelEngine() :f_state(ENG_READY) {};
	PixelEngine(EngineState init);
	~PixelEngine();

	EngineState smooth(Pixels & src, const Matrix & mask, float factor);
	EngineState smooth2D(Pixels & src, const Matrix & mask1, const Matrix & mask2, float factor1, float factor2);
	EngineState resize(Pixels & src, std::uint16_t newWidth, std::uint16_t newHeight, std::uint8_t mode);
	EngineState rotate(Pixels & src, float_t angle, std::uint8_t mode);
	EngineState rotate2(Pixels& src, float_t angle, std::uint8_t mode);
	EngineState flip(Pixels & src, std::uint8_t mode, std::uint16_t selectLine=0);
	EngineState HOG(Pixels const & src, Pixels & hog, Matrix & mX, Matrix & mY, 
					std::uint16_t startX,
					std::uint16_t startY,
					std::uint16_t endX,
					std::uint16_t endY,
					std::size_t particle=9, 
					bool isWeighted =false);

private:
	std::vector<Pixels> v_Pixels;
	EngineState f_state;

#ifdef CUDA_CODE_COMPILE
	// Only for CUDA
	EngineState custate_;
	int device_;
	cudaDeviceProp prop_;
#endif // CUDA_CODE_COMPILE

};

}

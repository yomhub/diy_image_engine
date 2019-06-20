#pragma once

#include <stdio.h>
#include <string.h>
#include <stdint.h>		// For uint**_t
#include <assert.h>		// Assert
#include <stdbool.h>	// Bool
#include <stdlib.h>		// Malloc
#include <math.h>		// float_t

#define uchar unsigned char

#if defined(_WIN32) || defined(_WIN64)
#define myfloat float_t
#else
#define myfloat float
#endif


#define myassert(expression)	\
	if (!(expression))            \
	return PNM_RT_ERR;
/*
#ifdef NDEBUG
#else
#define myassert(expression) assert(expression)
#endif // NDEBUG
*/

#define clamp(min_value,max_value,value) 	\
		value < min_value ? min_value : 	\
		(value > max_value ? max_value : value);

#define getFileLen(size,pfile)		\
		fseek(pfile,0,SEEK_END);	\
		size=ftell(pfile);			\
		fseek(pfile,0,SEEK_SET);	\
		
typedef enum PNM_STATE
{
	PNM_SUCCESS = 0, //  SUCCESS
	PNM_RT_ERR,
	PNM_SETTING_ERR,
	PNM_MEMERY_INSUFFICIENT,
	PNM_FILE_FORMAT_ERR,
	// For thread
	PNM_UNINITIALIZED,
	PNM_THREAD_RUNNING,
	PNM_WAIT_QUIT,
	PNM_WAIT_DELETE,
	PNM_PAUSE,
	PNM_SLEEP
}PNM_STATE;

typedef enum PNMTYPE
{
	NO_TYPE = 0,   //  NONE
	PBM_ASCII,	 //  P1
	PGM_ASCII,	 //  P2
	PPM_ASCII,	 //  P3
	PBM_BINARY,	//  P4
	PGM_BINARY,	//  P5
	PPM_BINARY,	//  P6
	PAM,		   //  P7
	PFM_RGB,	   //  PF
	PFM_GREYSCALE //  Pf
}PNMTYPE;

const char c_NO_TYPE = "  ";
const char c_PBM_ASCII = "P1";
const char c_PGM_ASCII = "P2";
const char c_PPM_ASCII = "P3";
const char c_PBM_BINARY = "P4";
const char c_PGM_BINARY = "P5";
const char c_PPM_BINARY = "P6";
const char c_PAM = "P7";
const char c_PFM_RGB = "PF";
const char c_PFM_GREYSCALE = "pf";

#define PNMTYPE2MagNum(type)					\
		type == NO_TYPE ? c_NO_TYPE:			\
		type == PBM_ASCII ? c_PBM_ASCII:		\
		type == PGM_ASCII ? c_PGM_ASCII:		\
		type == PPM_ASCII ? c_PPM_ASCII:		\
		type == PBM_BINARY ? c_PBM_BINARY:		\
		type == PGM_BINARY ? c_PGM_BINARY:		\
		type == PPM_BINARY ? c_PPM_BINARY:		\
		type == PAM ? c_PAM:					\
		type == PFM_RGB ? c_PFM_RGB:			\
		c_PFM_GREYSCALE;								

#define MagNum2PNMTYPE(magicNum)				\
		magicNum[0] != 'P' ? NO_TYPE :			\
		magicNum[1] == '1' ? PBM_ASCII :		\
		magicNum[1] == '2' ? PGM_ASCII :		\
		magicNum[1] == '3' ? PPM_ASCII :		\
		magicNum[1] == '4' ? PBM_ASCII :		\
		magicNum[1] == '5' ? PGM_ASCII :		\
		magicNum[1] == '6' ? PPM_ASCII :		\
		magicNum[1] == '7' ? PAM :				\
		magicNum[1] == 'F' ? PFM_RGB :			\
		magicNum[1] == 'f' ? PFM_GREYSCALE :	\
		NO_TYPE;

typedef struct PNM
{
	char* filename;
	uint16_t width;
	uint16_t height;
	uint8_t threshold;
	uint16_t maxValue;
	enum PNMTYPE type;
	uchar magicNumber[2];
	uchar* data;
}PNM;

inline PNM_STATE mymalloc(void** buff, size_t size) {
	if (!(*buff)) {
		(*buff) = malloc(size);
		if (!(*buff)) {
			(*buff) = NULL;
			return PNM_MEMERY_INSUFFICIENT;
		}
	}
	else if (sizeof(*buff) < size) {
		uchar* tmp;
		tmp = realloc(*buff, size);
		if (!tmp)return PNM_MEMERY_INSUFFICIENT;
		*buff = tmp;
	}
	return PNM_SUCCESS;
}

struct PNM_IO
{
	// need PPM.filename
	PNM_STATE (*ReadPNMFile)(PNM *f, uint16_t const pa);
	PNM_STATE (*ReadPBMFile)(PNM *f, uint16_t const pa);
	PNM_STATE (*ReadPGMFile)(PNM *f);
	PNM_STATE (*ReadPPMFile)(PNM *f);

	PNM_STATE (*WritePNMFile)(PNM *f, uint16_t const pa);
	PNM_STATE (*WritePBMFile)(PNM *f, uint16_t const pa);
	PNM_STATE (*WritePGMFile)(PNM *f);
	PNM_STATE (*WritePPMFile)(PNM *f);

	PNM_STATE (*Greyscale2RGB)(size_t const width,
									 size_t const height,
									 myfloat const fr,
									 myfloat const fg,
									 myfloat const fb,
									 uchar **greyscale_pixel_data,
									 uchar **rgb_pixel_data);

	PNM_STATE (*RGB2Greyscale)(size_t const width,
									 size_t const height,
									 uchar **rgb_pixel_data,
									 uchar **greyscale_pixel_data);

	PNM_STATE (*BitMap2Greyscale)(size_t const width,
										size_t const height,
										uint8_t threshold,
										uchar **bit_map_data,
										uchar **greyscale_pixel_data);

	PNM_STATE (*Greyscale2BitMap)(size_t const width,
										size_t const height,
										uint16_t threshold,
										uchar **greyscale_pixel_data,
										uchar **bit_map_data);

	// thread *p_mainThread;
	// PNM_STATE n_pnmThreadState;
	// ifstream f_istream;
	// ofstream f_ostream;
	PNM_STATE n_pnmGlobalState;
};

extern const struct PNM_IO PNM_IO;


#pragma once
#define __STDC_WANT_LIB_EXT1__ 1
#include <stdio.h>
#include <string.h>
#include <stdint.h>  // For uint**_t
#include <assert.h>  // Assert
#include <stdbool.h> // Bool
#include <stdlib.h>  // Malloc
#include <math.h>	// float_t
#include <malloc.h>

#if defined(_WIN32) || defined(_WIN64)
#define myfloat std::float_t
#else
#define myfloat float_t
#endif
#define uchar unsigned char

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
} PNM_STATE;

typedef enum PNMTYPE
{
	NO_TYPE = 0,  //  NONE
	PBM_ASCII,	//  P1
	PGM_ASCII,	//  P2
	PPM_ASCII,	//  P3
	PBM_BINARY,   //  P4
	PGM_BINARY,   //  P5
	PPM_BINARY,   //  P6
	PAM,		  //  P7
	PFM_RGB,	  //  PF
	PFM_GREYSCALE //  Pf
} PNMTYPE;

typedef struct PNM
{
	char *filename;
	uint16_t width;
	uint16_t height;
	uint8_t threshold;
	uint8_t sizePrePixel;
	uint16_t maxValue;
	enum PNMTYPE type;
	uchar magicNumber[2];
	uchar* data;
} PNM;



struct pNM_IO
{
	// need PPM.filename
	PNM_STATE (*ReadPNMFile)
	(PNM* f, uint16_t const pa);
	PNM_STATE (*ReadPBMFile)
	(PNM* f, uint16_t const pa);
	PNM_STATE (*ReadPGMFile)
	(PNM* f);
	PNM_STATE (*ReadPPMFile)
	(PNM* f);

	PNM_STATE (*WritePNMFile)
	(PNM* f, uint16_t const pa);
	PNM_STATE (*WritePBMFile)
	(PNM* f, uint16_t const pa);
	PNM_STATE (*WritePGMFile)
	(PNM* f);
	PNM_STATE (*WritePPMFile)
	(PNM* f);

	PNM_STATE (*Greyscale2RGB)
	(size_t const width,
	 size_t const height,
	 myfloat const fr,
	 myfloat const fg,
	 myfloat const fb,
	 uchar **greyscale_pixel_data,
	 uchar **rgb_pixel_data);

	PNM_STATE (*RGB2Greyscale)
	(size_t const width,
	 size_t const height,
	 uchar **rgb_pixel_data,
	 uchar **greyscale_pixel_data);

	PNM_STATE (*BitMap2Greyscale)
	(size_t const width,
	 size_t const height,
	 uint8_t threshold,
	 uchar **bit_map_data,
	 uchar **greyscale_pixel_data);

	PNM_STATE (*Greyscale2BitMap)
	(size_t const width,
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

extern const struct pNM_IO PNM_IO;

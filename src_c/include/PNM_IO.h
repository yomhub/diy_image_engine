#pragma once

#include <stdio.h>
#include <string.h>
#include <stdint.h>		// For uint**_t
#include <assert.h>		// Assert
#include <stdbool.h>	// Bool

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

#define PNMTYPE2MagNum(type)				\
		type == NO_TYPE ? "  " : 			\
		type == PBM_ASCII ? "P1" : 			\
		type == PGM_ASCII ? "P2" : 			\
		type == PPM_ASCII ? "P3" : 			\
		type == PBM_BINARY ? "P4" : 		\
		type == PGM_BINARY ? "P5" : 		\
		type == PPM_BINARY ? "P6" : 		\
		type == PAM ? "P7" : 				\
		type == PFM_RGB ? "PF" : 			\
		"Pf";								

#define MagNum2PNMTYPE(magicNum)			\
		magicNum == "P1" ? 					\
		PBM_ASCII : magicNum == "P2" ? 		\
		PGM_ASCII : magicNum == "P3" ? 		\
		PPM_ASCII : magicNum == "P4" ? 		\
		PBM_BINARY : magicNum == "P5" ? 	\
		PGM_BINARY : magicNum == "P6" ? 	\
		PPM_BINARY : magicNum == "P7" ? 	\
		PAM : magicNum == "PF" ? 			\
		PFM_RGB : magicNum == "Pf" ? 		\
		PFM_GREYSCALE : 					\
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

struct PNM_IO
{
	// need PPM.filename
	PNM_STATE (*ReadPNMFile)(PNM *f, uint16_t pa = 128);
	PNM_STATE (*ReadPBMFile)(PNM *f, uint16_t pa = 128);
	PNM_STATE (*ReadPGMFile)(PNM *f);
	PNM_STATE (*ReadPPMFile)(PNM *f);

	PNM_STATE (*WritePNMFile)(PNM *f, uint16_t const pa = 128);
	PNM_STATE (*WritePBMFile)(PNM *f, uint16_t const pa = 128);
	PNM_STATE (*WritePGMFile)(PNM *f);
	PNM_STATE (*WritePPMFile)(PNM *f);

	
	PNM_STATE (*ConvertFormat)(PNM *src, PNM *dst, vector<myfloat> const *pa );
	PNM_STATE (*ConvertFormat)(PNM *f, PNMTYPE type, vector<myfloat> const *pa);

	PNM_STATE CreateTask(void (*cbfun)(PNM f), vector<string> const *s_list);
	PNM_STATE StartTask();
	PNM_STATE PauseTask();
	PNM_STATE DeleteTask();

	PNM_STATE (*ReadHeader)(PNM *f, istream &is);
	PNM_STATE ReadPixelData(PNM *f, istream &is);

	PNM_STATE WriteHeader(PNM *f, ostream &os);
	PNM_STATE WritePixelData(PNM *f, ostream &os);

	void ThreadMain(void(*cbfun)(PNM f),vector<string> const * s_list);
	PNM_STATE (*Greyscale2RGB)(size_t const width,
									 size_t const height,
									 myfloat const fr,
									 myfloat const fg,
									 myfloat const fb,
									 vector<uint8_t> *const greyscale_pixel_data,
									 vector<uint8_t> *const rgb_pixel_data);

	PNM_STATE (*RGB2Greyscale)(size_t const width,
									 size_t const height,
									 vector<uint8_t> *const rgb_pixel_data,
									 vector<uint8_t> *const greyscale_pixel_data);

	PNM_STATE (*BitMap2Greyscale)(size_t const width,
										size_t const height,
										uint8_t threshold,
										vector<uint8_t> *const bit_map_data,
										vector<uint8_t> *const greyscale_pixel_data);

	PNM_STATE (*Greyscale2BitMap)(size_t const width,
										size_t const height,
										uint16_t threshold,
										vector<uint8_t> *const greyscale_pixel_data,
										vector<uint8_t> *const bit_map_data);

	// thread *p_mainThread;
	// PNM_STATE n_pnmThreadState;
	// ifstream f_istream;
	// ofstream f_ostream;
	// PNM_STATE n_pnmGlobalState;
};

extern const struct PNM_IO PNM_IO;


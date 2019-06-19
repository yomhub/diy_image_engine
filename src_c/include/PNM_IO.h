#pragma once

#include <stdio.h>
#include <string.h>
#include <stdint.h>		// For uint**_t

#define uchar unsigned char

#if defined(_WIN32) || defined(_WIN64)
#define myfloat std::float_t
#else
#define myfloat float
#endif

#ifdef NDEBUG
#define myassert(expression, mes) \
	if (!(expression))            \
	return PNM_RT_ERR
#else
#define myassert(expression, mes) assert(expression &&mes)
#endif // NDEBUG

#define clamp(min_value,max_value,value) 	\
		value < min_value ? min_value : 	\
		(value > max_value ? max_value : value);

#define getFileLen(size,pfile)		\
		fseek(pfile,0,SEEK_END);	\
		size=ftell(pfile);			\
		fseek(pfile,0,SEEK_SET);	\


enum PNM_STATE
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
};

enum PNMTYPE
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
};

#define PNMTYPE2MagNum(type)				\
		type == NO_TYPE ? "" : 				\
		type == PBM_ASCII ? "P1" : 			\
		type == PGM_ASCII ? "P2" : 			\
		type == PPM_ASCII ? "P3" : 			\
		type == PBM_BINARY ? "P4" : 		\
		type == PGM_BINARY ? "P5" : 		\
		type == PPM_BINARY ? "P6" : 		\
		type == PAM ? "P7" : 				\
		type == PFM_RGB ? "PF" : 			\
		"Pf"								\

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
		NO_TYPE								\

struct PNM
{
	char* filename;
	uint16_t width;
	uint16_t height;
	uint8_t threshold = 128;
	uint32_t maxValue = 255;
	PNMTYPE type;
	uint16_t magicNumber = NULL;
	uchar* data;
};

struct PNM_IO
{
	// need PPM.filename
	PNM_STATE ReadPNMFile(PNM *f, uint16_t pa = 128);
	PNM_STATE ReadPBMFile(PNM *f, uint16_t pa = 128);
	PNM_STATE ReadPGMFile(PNM *f);
	PNM_STATE ReadPPMFile(PNM *f);

	PNM_STATE WritePNMFile(PNM *f, std::uint16_t const pa = 128);
	PNM_STATE WritePBMFile(PNM *f, std::uint16_t const pa = 128);
	PNM_STATE WritePGMFile(PNM *f);
	PNM_STATE WritePPMFile(PNM *f);

	
	PNM_STATE ConvertFormat(PNM *src, PNM *dst, std::vector<myfloat> const *pa );
	PNM_STATE ConvertFormat(PNM *f, PNMTYPE type, std::vector<myfloat> const *pa);

	PNM_STATE CreateTask(void (*cbfun)(PNM f), std::vector<std::string> const *s_list);
	PNM_STATE StartTask();
	PNM_STATE PauseTask();
	PNM_STATE DeleteTask();

	PNM_STATE ReadHeader(PNM *f, std::istream &is);
	PNM_STATE ReadPixelData(PNM *f, std::istream &is);

	PNM_STATE WriteHeader(PNM *f, std::ostream &os);
	PNM_STATE WritePixelData(PNM *f, std::ostream &os);

	void ThreadMain(void(*cbfun)(PNM f),std::vector<std::string> const * s_list);
	PNM_STATE Greyscale2RGB(std::size_t const width,
									 std::size_t const height,
									 myfloat const fr,
									 myfloat const fg,
									 myfloat const fb,
									 std::vector<std::uint8_t> *const greyscale_pixel_data,
									 std::vector<std::uint8_t> *const rgb_pixel_data);

	PNM_STATE RGB2Greyscale(std::size_t const width,
									 std::size_t const height,
									 std::vector<std::uint8_t> *const rgb_pixel_data,
									 std::vector<std::uint8_t> *const greyscale_pixel_data);

	PNM_STATE BitMap2Greyscale(std::size_t const width,
										std::size_t const height,
										std::uint8_t threshold,
										std::vector<std::uint8_t> *const bit_map_data,
										std::vector<std::uint8_t> *const greyscale_pixel_data);

	PNM_STATE Greyscale2BitMap(std::size_t const width,
										std::size_t const height,
										std::uint16_t threshold,
										std::vector<std::uint8_t> *const greyscale_pixel_data,
										std::vector<std::uint8_t> *const bit_map_data);

	// std::thread *p_mainThread;
	// PNM_STATE n_pnmThreadState;
	// std::ifstream f_istream;
	// std::ofstream f_ostream;
	// PNM_STATE n_pnmGlobalState;
};

extern const struct PNM_IO PNM_IO;


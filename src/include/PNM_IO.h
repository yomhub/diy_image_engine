#pragma once

#include <cassert>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>
#include <iostream>
#include <thread>
#include <chrono>

#if defined(_WIN32) || defined(_WIN64)
#define myfloat float//std::float_t
#else
#define myfloat float
#endif
namespace pnm_io
{

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
	PNM_SLEEP,
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
	PFM_GREYSCALE, //  Pf
};

struct PNM
{
	PNM(std::string name) { this->filename = name; };
	PNM(std::string name,PNM *copyPara) { 
		this->filename = name; 
		this->height = copyPara->height;
		this->width = copyPara->width;
		this->threshold = copyPara->threshold;
		this->type = copyPara->type;
		this->max_value = copyPara->max_value;
	};
	PNM() {
		this->threshold = 128;
		this->max_value = 255;
		this->magic_number = "";
	};
	std::string filename;
	std::size_t width;
	std::size_t height;
	std::uint8_t threshold = 128;
	std::uint32_t max_value = 255;
	PNMTYPE type;
	std::string magic_number = "";
	std::vector<std::uint8_t> data;
};

namespace detail
{

template <typename T>
inline T clamp(const T min_value, const T max_value, const T value)
{
	return value < min_value ? min_value
							 : (value > max_value ? max_value : value);
}

#ifdef NDEBUG
#define myassert(expression, mes) \
	if (!(expression))            \
	return PNM_RT_ERR
#else
#define myassert(expression, mes) assert(expression &&mes)
#endif // NDEBUG

template <typename T>
constexpr PNMTYPE MagNum2PNMTYPE(T magicNum)
{
	return magicNum == "P1" ? PBM_ASCII : magicNum == "P2" ? PGM_ASCII : magicNum == "P3" ? PPM_ASCII : magicNum == "P4" ? PBM_BINARY : magicNum == "P5" ? PGM_BINARY : magicNum == "P6" ? PPM_BINARY : magicNum == "P7" ? PAM : magicNum == "PF" ? PFM_RGB : magicNum == "Pf" ? PFM_GREYSCALE : NO_TYPE;
}

template <typename T>
constexpr std::string PNMTYPE2MagNum(T type)
{
	return type == NO_TYPE ? "" : type == PBM_ASCII ? "P1" : type == PGM_ASCII ? "P2" : type == PPM_ASCII ? "P3" : type == PBM_BINARY ? "P4" : type == PGM_BINARY ? "P5" : type == PPM_BINARY ? "P6" : type == PAM ? "P7" : type == PFM_RGB ? "PF" : "Pf";
}
} // namespace detail

class PNM_IO
{
public:
	PNM_IO();
	~PNM_IO();

	// need PPM.filename
	pnm_io::PNM_STATE ReadPNMFile(PNM *f, std::uint16_t const pa = 128);
	pnm_io::PNM_STATE ReadPBMFile(PNM *f, std::uint16_t const pa = 128);
	pnm_io::PNM_STATE ReadPGMFile(PNM *f);
	pnm_io::PNM_STATE ReadPPMFile(PNM *f);

	pnm_io::PNM_STATE WritePNMFile(PNM *f, std::uint16_t const pa = 128);
	pnm_io::PNM_STATE WritePBMFile(PNM *f, std::uint16_t const pa = 128);
	pnm_io::PNM_STATE WritePGMFile(PNM *f);
	pnm_io::PNM_STATE WritePPMFile(PNM *f);

	
	pnm_io::PNM_STATE ConvertFormat(PNM *src, PNM *dst, std::vector<myfloat> const *pa );
	pnm_io::PNM_STATE ConvertFormat(PNM *f, PNMTYPE type, std::vector<myfloat> const *pa);

	pnm_io::PNM_STATE CreateTask(void (*cbfun)(PNM f), std::vector<std::string> const *s_list);
	pnm_io::PNM_STATE StartTask();
	pnm_io::PNM_STATE PauseTask();
	pnm_io::PNM_STATE DeleteTask();

private:
	template <typename FileStreamT>
	inline pnm_io::PNM_STATE
	OpenFileStream(PNM const *f,
				   FileStreamT *f_stream,
				   std::ios_base::openmode const mode = std::ios_base::binary);

	pnm_io::PNM_STATE ReadHeader(PNM *f, std::istream &is);
	pnm_io::PNM_STATE ReadPixelData(PNM *f, std::istream &is);

	pnm_io::PNM_STATE WriteHeader(PNM *f, std::ostream &os);
	pnm_io::PNM_STATE WritePixelData(PNM *f, std::ostream &os);

	void ThreadMain(void(*cbfun)(PNM f),std::vector<std::string> const * s_list);
	pnm_io::PNM_STATE Greyscale2RGB(std::size_t const width,
									 std::size_t const height,
									 myfloat const fr,
									 myfloat const fg,
									 myfloat const fb,
									 std::vector<std::uint8_t> *const greyscale_pixel_data,
									 std::vector<std::uint8_t> *const rgb_pixel_data);

	pnm_io::PNM_STATE RGB2Greyscale(std::size_t const width,
									 std::size_t const height,
									 std::vector<std::uint8_t> *const rgb_pixel_data,
									 std::vector<std::uint8_t> *const greyscale_pixel_data);

	pnm_io::PNM_STATE BitMap2Greyscale(std::size_t const width,
										std::size_t const height,
										std::uint8_t threshold,
										std::vector<std::uint8_t> *const bit_map_data,
										std::vector<std::uint8_t> *const greyscale_pixel_data);

	pnm_io::PNM_STATE Greyscale2BitMap(std::size_t const width,
										std::size_t const height,
										std::uint16_t threshold,
										std::vector<std::uint8_t> *const greyscale_pixel_data,
										std::vector<std::uint8_t> *const bit_map_data);
private:
	std::thread *p_mainThread;
	pnm_io::PNM_STATE n_pnmThreadState;
	std::ifstream f_istream;
	std::ofstream f_ostream;
	pnm_io::PNM_STATE n_pnmGlobalState;
};

} // namespace pnm_io

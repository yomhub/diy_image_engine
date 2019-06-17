// Copyright(C) 2018 Tommy Hinks <tommy.hinks@gmail.com>
// This file is subject to the license terms in the LICENSE file
// found in the top-level directory of this distribution.

#ifndef THINKS_PNM_IO_PNM_IO_H_INCLUDED
#define THINKS_PNM_IO_PNM_IO_H_INCLUDED

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

namespace thinks
{
namespace pnm_io
{

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

struct PPM
{
	std::string filename;
	std::size_t width;
	std::size_t height;
	std::vector<std::uint8_t> data;
	PNMTYPE type;
};

namespace detail
{

template <typename T>
T clamp(const T min_value, const T max_value, const T value)
{
	return value < min_value ? min_value
							 : (value > max_value ? max_value : value);
}

template <typename FileStreamT>
void OpenFileStream(FileStreamT *const file_stream, std::string const &filename,
					std::ios_base::openmode const mode = std::ios_base::binary)
{
	assert(file_stream != nullptr && "file stream is null");

	file_stream->open(filename, mode);
	if (!(*file_stream))
	{
		constexpr auto kErrMsgLen = std::size_t{1024};
		char err_msg[kErrMsgLen];
		strerror_s(err_msg, kErrMsgLen, errno);
		auto oss = std::ostringstream{};
		oss << "cannot open file '" << filename << "', "
			<< "error: '" << err_msg << "'";
		throw std::runtime_error(oss.str());
	}
}

template <typename ExceptionT>
void ThrowIfInvalidMagicNumber(std::string const &magic_number,
							   std::string const &expected_magic_number)
{
	if (magic_number != expected_magic_number)
	{
		auto oss = std::ostringstream{};
		oss << "magic number must be '" << expected_magic_number << "', was '"
			<< magic_number << "'";
		throw ExceptionT(oss.str());
	}
}

template <typename ExceptionT>
void ThrowIfInvalidType(PNMTYPE type,
						PNMTYPE expected_type)
{
	if (type != expected_type)
	{
		auto oss = std::ostringstream{};
		oss << "magic number must be '" << expected_type << "', was '"
			<< type << "'";
		throw ExceptionT(oss.str());
	}
}

template <typename ExceptionT>
void ThrowIfInvalidWidth(std::size_t const width)
{
	if (width == 0)
	{
		throw ExceptionT("width must be non-zero");
	}
}

template <typename ExceptionT>
void ThrowIfInvalidHeight(std::size_t const height)
{
	if (height == 0)
	{
		throw ExceptionT("height must be non-zero");
	}
}

template <typename ExceptionT>
void ThrowIfInvalidMaxValue(std::uint32_t const max_value)
{
	constexpr auto expected_max_value =
		std::uint32_t{std::numeric_limits<std::uint8_t>::max()};
	if (max_value != expected_max_value)
	{
		auto oss = std::ostringstream{};
		oss << "max value must be " << expected_max_value << ", was " << max_value;
		throw ExceptionT(oss.str());
	}
}

template <typename ExceptionT>
void ThrowIfInvalidPixelData(std::vector<std::uint8_t> const &pixel_data,
							 std::size_t const expected_size)
{
	if (pixel_data.size() != expected_size)
	{
		throw ExceptionT("pixel data must match width and height");
	}
}

struct Header
{
	std::string magic_number = "";
	std::size_t width = 0;
	std::size_t height = 0;
	std::uint32_t max_value = 255;
	PNMTYPE type = NO_TYPE;
};
/*
Attention: Not support comment

*/
inline Header ReadHeader(std::istream &is)
{
	auto header = Header{};
	is.seekg(0, std::ios::beg);
	is >> header.magic_number >> header.width >> header.height >>
      header.max_value;

	header.type =	 header.magic_number == "P1" ? PBM_ASCII :
					 header.magic_number == "P2" ? PGM_ASCII : 
					 header.magic_number == "P3" ? PPM_ASCII : 
					 header.magic_number == "P4" ? PBM_BINARY : 
					 header.magic_number == "P5" ? PGM_BINARY : 
					 header.magic_number == "P6" ? PPM_BINARY : 
					 header.magic_number == "P7" ? PAM : 
					 header.magic_number == "PF" ? PFM_RGB : 
					 header.magic_number == "Pf" ? PFM_GREYSCALE : 
					 NO_TYPE;

	ThrowIfInvalidWidth<std::runtime_error>(header.width);
	ThrowIfInvalidHeight<std::runtime_error>(header.height);
	ThrowIfInvalidMaxValue<std::runtime_error>(header.max_value);

	// Skip ahead (an arbitrary number!) to the pixel data.
	is.ignore(256, '\n');

	return header;
}

inline void WriteHeader(std::ostream &os, Header const &header)
{
	ThrowIfInvalidWidth<std::invalid_argument>(header.width);
	ThrowIfInvalidHeight<std::invalid_argument>(header.height);
	ThrowIfInvalidMaxValue<std::invalid_argument>(header.max_value);

	os << header.magic_number << "\n"
	   << header.width << "\n"
	   << header.height << "\n"
	   << header.max_value << "\n"; // Marks beginning of pixel data.
}

inline void ReadPixelData(std::istream &is,
						  std::vector<std::uint8_t> *const pixel_data)
{
	is.read(reinterpret_cast<char *>(pixel_data->data()), pixel_data->size());

	if (!is)
	{
		auto oss = std::ostringstream();
		oss << "failed reading " << pixel_data->size() << " bytes";
		throw std::runtime_error(oss.str());
	}
}

inline void WritePixelData(std::ostream &os,
						   std::uint8_t const *const pixel_data,
						   std::size_t const size)
{
	os.write(reinterpret_cast<char const *>(pixel_data), size);
}

} // namespace detail

/*!
Read a PBM image from an input stream.

Pre-conditions:
  - the PBM header does not contain any comments.
  - the output pointers are non-null.

Pixel data is read as RGB triplets in row major order. For instance,
the pixel data for a 2x2 image is represented as follows:

		Column 0      Column 1
	  +-------------+-------------+
	  |             |             |
Row 0 | I: data[0]  | I: data[1]  |
	  |             |             |
	  +-------------+-------------+
	  |             |             |
Row 1 | I: data[2]  | I: data[3]  |
	  |             |             |
	  +-------------+-------------+

An std::runtime_error is thrown if:
  - the magic number is not 'P4'.
  - width or height is zero.
  - the pixel data cannot be read.
*/
inline void ReadPbmImage(std::istream &is, std::size_t *const width,
						 std::size_t *const height,
						 std::size_t threshold,
						 std::vector<std::uint8_t> *const pixel_data)
{
#ifdef NDEBUG
	if (!width || !height || !pixel_data)return;
#endif // NDEBUG
	assert(width != nullptr && "null width");
	assert(height != nullptr && "null height");
	assert(pixel_data != nullptr && "null pixel data");

	auto header = detail::ReadHeader(is);
	detail::ThrowIfInvalidType<std::runtime_error>(header.type, PBM_BINARY);

	*width = header.width;
	*height = header.height;

	if(pixel_data->size() < (*width) * (*height))pixel_data->resize((*width) * (*height));
	std::uint8_t buff;
	for (auto row = std::size_t{0}; row < *height; ++row)
	{
		for (auto col = std::size_t{0}; col < *width; ++col)
		{
			is.read(reinterpret_cast<char *>(&buff), 1);

			if (!is)
			{
				auto oss = std::ostringstream();
				oss << "failed reading at" << is.tellg() << " bytes";
				throw std::runtime_error(oss.str());
			}

			for (auto k = std::size_t{0}; k < 8; ++k)
			{
				pixel_data->data()[col + row * (*width)]=((buff >> (7-k)) & 0x1)? threshold : 0;
			}
		}
	}

}

/*!
Read a PGM (greyscale) image from an input stream.

Pre-conditions:
  - the PGM header does not contain any comments.
  - the output pointers are non-null.

Pixel data is read as RGB triplets in row major order. For instance,
the pixel data for a 2x2 image is represented as follows:

        Column 0      Column 1
      +-------------+-------------+
      |             |             |
Row 0 | I: data[0]  | I: data[1]  |
      |             |             |
      +-------------+-------------+
      |             |             |
Row 1 | I: data[2]  | I: data[3]  |
      |             |             |
      +-------------+-------------+

An std::runtime_error is thrown if:
  - the magic number is not 'P5'.
  - width or height is zero.
  - the max value is not '255'.
  - the pixel data cannot be read.
*/
inline void ReadPgmImage(std::istream &is, std::size_t *const width,
						 std::size_t *const height,
						 std::vector<std::uint8_t> *const pixel_data)
{
#ifdef NDEBUG
	if (!width || !height || !pixel_data)return;
#endif // NDEBUG
	assert(width != nullptr && "null width");
	assert(height != nullptr && "null height");
	assert(pixel_data != nullptr && "null pixel data");

	auto header = detail::ReadHeader(is);
	detail::ThrowIfInvalidMagicNumber<std::runtime_error>(header.magic_number,
														  "P5");

	*width = header.width;
	*height = header.height;

	if(pixel_data->size()< (*width) * (*height))pixel_data->resize((*width) * (*height));
	detail::ReadPixelData(is, pixel_data);
}


/*!
See std::istream overload version.

Throws an std::runtime_error if file cannot be opened.
*/
inline void ReadPgmImage(std::string const &filename, std::size_t *const width,
						 std::size_t *const height,
						 std::vector<std::uint8_t> *const pixel_data)
{
#ifdef NDEBUG
	if (!width || !height || !pixel_data)return;
#endif // NDEBUG
	assert(width != nullptr && "null width");
	assert(height != nullptr && "null height");
	assert(pixel_data != nullptr && "null pixel data");
	auto ifs = std::ifstream{};
	detail::OpenFileStream(&ifs, filename);
	ReadPgmImage(ifs, width, height, pixel_data);
	ifs.close();
}

/*!
Write a PBM (Bit) image to an output stream.

Pixel data is given a intensities in row major order. For instance,
the pixel data for a 2x2 image is represented as follows:

		Column 0      Column 1
	  +-------------+-------------+
	  |             |             |
Row 0 | I: data[0]  | I: data[1]  |
	  |             |             |
	  +-------------+-------------+
	  |             |             |
Row 1 | I: data[2]  | I: data[3]  |
	  |             |             |
	  +-------------+-------------+

And write as:
	0 0
	1 0
	whitch 1 means data[x]!=0 and 0 means data[x]==0
An std::invalid_argument is thrown if:
  - width or height is zero.
  - the size of the pixel data does not match the width and height.
*/

inline void WritePbmImage(std::ostream &os, std::size_t const width,
	std::size_t const height,
	std::uint8_t const threshold,
	std::uint8_t const *const pixel_data)
{
	auto header = detail::Header{};
	header.magic_number = "P4";
	header.width = width;
	header.height = height;
	os << header.magic_number << "\n"
	   << header.width << "\n"
	   << header.height << "\n";
	std::uint8_t buff=0x00,temp=0x00;


	for (auto row = std::size_t{ 0 }; row < height; ++row)
	{
		for (auto col = std::size_t{ 0 }; col < width; col +=8)
		{
			for (auto k = std::size_t{ 0 }; k < 8; ++k) {
				temp |= (pixel_data[col + row * (width)+k] > threshold ? 0x1 : 0x0) << (7 - k);
				buff |= pixel_data[col + row * (width)+k]  << (7 - k);
			}
			os << temp;
			temp = buff = 0x00;
			//if (((row*width + col) % 32) == (32 - 1))os << "\n";
			
		}
		os << "\n";
	}
}

/*!
Write a PGM (greyscale) image to an output stream.

Pixel data is given a intensities in row major order. For instance,
the pixel data for a 2x2 image is represented as follows:

        Column 0      Column 1
      +-------------+-------------+
      |             |             |
Row 0 | I: data[0]  | I: data[1]  |
      |             |             |
      +-------------+-------------+
      |             |             |
Row 1 | I: data[2]  | I: data[3]  |
      |             |             |
      +-------------+-------------+

An std::invalid_argument is thrown if:
  - width or height is zero.
  - the size of the pixel data does not match the width and height.
*/
inline void WritePgmImage(std::ostream &os, std::size_t const width,
						  std::size_t const height,
						  std::uint8_t const *const pixel_data)
{
	auto header = detail::Header{};
	header.magic_number = "P5";
	header.width = width;
	header.height = height;
	detail::WriteHeader(os, header);
	detail::WritePixelData(os, pixel_data, header.width * header.height);
}

/*!
See std::ostream overload version above.

Throws an std::runtime_error if file cannot be opened.
*/
inline void WritePgmImage(std::string const &filename, std::size_t const width,
						  std::size_t const height,
						  std::uint8_t const *const pixel_data)
{
	auto ofs = std::ofstream{};
	detail::OpenFileStream(&ofs, filename, std::ios_base::binary|std::ios_base::out);
	WritePgmImage(ofs, width, height, pixel_data);
	ofs.close();
}

/*!
Read a PPM (RGB) image from an input stream.

Pre-conditions:
  - the PPM header does not contain any comments.
  - the output pointers are non-null.

Pixel data is read as RGB triplets in row major order. For instance,
the pixel data for a 2x2 image is represented as follows:

        Column 0                           Column 1
      +----------------------------------+------------------------------------+
      |                                  |                                    |
Row 0 | RGB: {data[0], data[1], data[2]} | RGB: {data[3], data[4], data[5]}   |
      |                                  |                                    |
      +----------------------------------+------------------------------------+
      |                                  |                                    |
Row 1 | RGB: {data[6], data[7], data[8]} | RGB: {data[9], data[10], data[11]} |
      |                                  |                                    |
      +----------------------------------+------------------------------------+

An std::runtime_error is thrown if:
  - the magic number is not 'P6'.
  - width or height is zero.
  - the max value is not '255'.
  - the pixel data cannot be read.
*/
inline void ReadPpmImage(std::istream &is, std::size_t *const width,
						 std::size_t *const height,
						 std::vector<std::uint8_t> *const pixel_data)
{
#ifdef NDEBUG
	if (!width || !height || !pixel_data)return;
#endif // NDEBUG
	assert(width != nullptr && "null width");
	assert(height != nullptr && "null height");
	assert(pixel_data != nullptr && "null pixel data");

	auto header = detail::ReadHeader(is);
	detail::ThrowIfInvalidMagicNumber<std::runtime_error>(header.magic_number,
														  "P6");
	*width = header.width;
	*height = header.height;

	if(pixel_data->size()< (*width) * (*height) * 3)pixel_data->resize((*width) * (*height) * 3);
	detail::ReadPixelData(is, pixel_data);
}

/*!
See std::istream overload version.

Throws an std::runtime_error if file cannot be opened.
*/
inline void ReadPpmImage(std::string const &filename, std::size_t *const width,
						 std::size_t *const height,
						 std::vector<std::uint8_t> *const pixel_data)
{
#ifdef NDEBUG
	if (!width || !height || !pixel_data)return;
#endif // NDEBUG
	assert(width != nullptr && "null width");
	assert(height != nullptr && "null height");
	assert(pixel_data != nullptr && "null pixel data");

	auto ifs = std::ifstream{};
	detail::OpenFileStream(&ifs, filename);
	ReadPpmImage(ifs, width, height, pixel_data);
	ifs.close();
}

/*!
Write a PPM (RGB) image to an output stream.

Pixel data is given as RGB triplets in row major order. For instance,
the pixel data for a 2x2 image is represented as follows:

        Column 0                           Column 1
      +----------------------------------+------------------------------------+
      |                                  |                                    |
Row 0 | RGB: {data[0], data[1], data[2]} | RGB: {data[3], data[4], data[5]}   |
      |                                  |                                    |
      +----------------------------------+------------------------------------+
      |                                  |                                    |
Row 1 | RGB: {data[6], data[7], data[8]} | RGB: {data[9], data[10], data[11]} |
      |                                  |                                    |
      +----------------------------------+------------------------------------+

An std::invalid_argument is thrown if:
  - width or height is zero.
  - the size of the pixel data does not match the width and height.
*/
inline void WritePpmImage(std::ostream &os, std::size_t const width,
						  std::size_t const height,
						  std::uint8_t const *const pixel_data)
{
	auto header = detail::Header{};
	header.magic_number = "P6";
	header.width = width;
	header.height = height;
	detail::WriteHeader(os, header);
	detail::WritePixelData(os, pixel_data, header.width * header.height * 3);
}

/*!
See std::ostream overload version above.

Throws an std::runtime_error if file cannot be opened.
*/
inline void WritePpmImage(std::string const &filename, std::size_t const width,
						  std::size_t const height,
						  std::uint8_t const *const pixel_data)
{
	auto ofs = std::ofstream{};
	detail::OpenFileStream(&ofs, filename);
	WritePpmImage(ofs, width, height, pixel_data);
	ofs.close();
}

/*!
See std::ostream overload version above.

Throws an std::runtime_error if file cannot be opened.
*/
inline void WritePbmImage(std::string const &filename, std::size_t const width,
	std::size_t const height,
	std::size_t const threshold,
	std::uint8_t const *const pixel_data)
{
	auto ofs = std::ofstream{};
	detail::OpenFileStream(&ofs, filename);
	WritePbmImage(ofs, width, height, threshold, pixel_data);
	ofs.close();
}

/*!
See std::istream overload version.

Throws an std::runtime_error if file cannot be opened.
*/
inline void ReadPnmImage(std::string const &filename, std::size_t *const width,
	std::size_t *const height,
	std::vector<std::uint8_t> *const pixel_data)
{
#ifdef NDEBUG
	if (!width || !height || !pixel_data)return;
#endif // NDEBUG
	assert(width != nullptr && "null width");
	assert(height != nullptr && "null height");
	assert(pixel_data != nullptr && "null pixel data");
	auto ifs = std::ifstream{};
	detail::OpenFileStream(&ifs, filename);
	auto header = detail::ReadHeader(ifs);

	switch (header.type)
	{
	case PBM_BINARY:
		ReadPbmImage(ifs, width, height, 128, pixel_data);
		break;
	case PGM_BINARY:
		ReadPgmImage(ifs, width, height, pixel_data);
		break;
	case PPM_BINARY:
		ReadPpmImage(ifs, width, height, pixel_data);
		break;
	default:
		break;
	}

	ifs.close();
}

/*!
See std::istream overload version.

Throws an std::runtime_error if file cannot be opened.
*/
inline void WritePnmImage(std::string const &filename, std::size_t const width,
	std::size_t const height,
	std::vector<std::uint8_t> *const pixel_data,
	pnm_io::PNMTYPE pnmmtype)
{
#ifdef NDEBUG
	if ((pixel_data->size() < width * height) ||
		(pnmmtype == PPM_BINARY && pixel_data->size() < width * height * 3))return;
#endif // NDEBUG
	assert(pixel_data != nullptr && "null pixel data");
	auto ofs = std::ofstream{};
	detail::OpenFileStream(&ofs, filename);
	
	switch (pnmmtype)
	{
	case PBM_BINARY:
		assert(pixel_data->size()>= width * height && "input buffer size err in pnm_io::WritePnmImage");
		WritePbmImage(ofs, width, height, 128, pixel_data->data());
		break;
	case PGM_BINARY:
		assert(pixel_data->size() >= width * height &&"input buffer size err in pnm_io::WritePnmImage");
		WritePgmImage(ofs, width, height, pixel_data->data());
		break;
	case PPM_BINARY:
		assert(pixel_data->size() >= width * height * 3 &&"input buffer size err in pnm_io::WritePnmImage");
		WritePpmImage(ofs, width, height, pixel_data->data());
		break;
	default:
		break;
	}

	ofs.close();
}
/*!
Convert RGB to Greyscale while Gr=(R+G+B)/3

*/
inline void RGB2Greyscale(std::size_t const width,
						  std::size_t const height,
						  std::vector<std::uint8_t> *const rgb_pixel_data,
						  std::vector<std::uint8_t> *const greyscale_pixel_data)
{
#ifdef NDEBUG
	if (!rgb_pixel_data || rgb_pixel_data->size() < (width * height) * 3 ||
		!greyscale_pixel_data
		)return;
#endif // NDEBUG

	assert(rgb_pixel_data != nullptr && "null RGB pixel data");
	assert(rgb_pixel_data->size() <= (width * height) * 3 && "RGB pixel have a wrong size");
	assert(greyscale_pixel_data != nullptr && "null Greyscale pixel data");
	assert(greyscale_pixel_data->size() <= (width * height) && "Greyscale pixel have a wrong size");
	if (greyscale_pixel_data->size() < (width * height))
	{
		greyscale_pixel_data->resize(width * height, std::uint8_t{});
	}
	

	for (auto row = std::size_t{0}; row < height; ++row)
	{
		for (auto col = std::size_t{0}; col < width; ++col)
		{
			greyscale_pixel_data->data()[col + row * width] = (rgb_pixel_data->data()[3 * (col + row * width)] +
															   rgb_pixel_data->data()[3 * (col + row * width) + 1] +
															   rgb_pixel_data->data()[3 * (col + row * width) + 2]) /
															  3;
		}
	}
}

/*!
Convert Greyscale to RGB while :
	R = fr * Gr
	G = fg * Gr
	B = fb * Gr
*/
inline void Greyscale2RGB(std::size_t const width,
						  std::size_t const height,
						  std::float_t const fr,
						  std::float_t const fg,
						  std::float_t const fb,
						  std::vector<std::uint8_t> *const rgb_pixel_data,
						  std::vector<std::uint8_t> *const greyscale_pixel_data)
{
#ifdef NDEBUG
	if (!rgb_pixel_data || 
		!greyscale_pixel_data || greyscale_pixel_data->size() < (width * height)
		)return;
#endif // NDEBUG
	assert(greyscale_pixel_data != nullptr && "null Greyscale pixel data");
	assert(greyscale_pixel_data->size() <= (width * height) && "Greyscale pixel have a wrong size");
	assert(rgb_pixel_data != nullptr && "null RGB pixel data");

	if (rgb_pixel_data->size() < (width * height))
	{
		rgb_pixel_data->resize(width * height * 3, std::uint8_t{});
	}


	for (auto row = std::size_t{0}; row < height; ++row)
	{
		for (auto col = std::size_t{0}; col < width; ++col)
		{
			rgb_pixel_data->data()[3 * (col + row * width)] = detail::clamp(0.f, 1.f, fr) * greyscale_pixel_data->data()[col + row * width];
			rgb_pixel_data->data()[3 * (col + row * width) + 1] = detail::clamp(0.f, 1.f, fb) * greyscale_pixel_data->data()[col + row * width];
			rgb_pixel_data->data()[3 * (col + row * width) + 2] = detail::clamp(0.f, 1.f, fg) * greyscale_pixel_data->data()[col + row * width];
		}
	}
}

} // namespace pnm_io
} // namespace thinks

#endif // THINKS_PNM_IO_PNM_IO_H_INCLUDED

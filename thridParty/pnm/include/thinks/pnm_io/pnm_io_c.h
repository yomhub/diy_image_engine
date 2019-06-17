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
#include <stdio.h>

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

enum PNM_STATE
{
	SUCCESS = 0,   //  NONE
	
};

struct PPM
{
	std::string filename;
	std::size_t width;
	std::size_t height;
	std::vector<std::uint8_t> data;
	std::uint16_t threshold;
	PNMTYPE type;
};

namespace detail
{

inline void OpenFileStream(
	FILE *p_file,
	std::string const &filename,
	std::string const &openMode)
{
	p_file = fopen_s(filename.c_str(), openMode.c_str());
	assert(p_file && "Filed to open file. pnm_io::detail::OpenFileStream\n");
}

inline std::uint16_t FindCharInString(
	const char *p_char,
	const char c_char,
	std::size_t n_start,
	std::size_t n_end)
{
	for (std::size_t i = n_start; i < n_end; i++)
	{
		if (p_char[i] == c_char)
			return i;
	}
	return
}

#define READ_BUFFER_SIZE 20
#define MAX_NUM_DIGIT 5
inline void ReadAllData(FILE *p_file, PPM *p_pnm)
{
	assert(p_file && "Filed to open file. pnm_io::detail::ReadAllData\n");
	assert(p_pnm && "PPM pointer is null. pnm_io::detail::ReadAllData\n");
	std::size_t n_fileSize;
	std::uint16_t n_startSeparat=0, n_endSeparat=READ_BUFFER_SIZE, p_buff[READ_BUFFER_SIZE];
	std::uint8_t p_magicNum[2];

	fseek(p_file, 0, SEEK_END);
	n_fileSize = ftell(p_file);
	fseek(p_file, 0, SEEK_SET);

	fread(p_magicNum, 2, 1, p_file);

	header.type =	 header.magic_number == "P1" ? PBM_ASCII : \
					header.magic_number == "P2" ? PGM_ASCII : \
					header.magic_number == "P3" ? PPM_ASCII : \
					header.magic_number == "P4" ? PBM_BINARY : \
					header.magic_number == "P5" ? PGM_BINARY : \
					header.magic_number == "P6" ? PPM_BINARY : \
					header.magic_number == "P7" ? PAM : \
					header.magic_number == "PF" ? PFM_RGB : \
					header.magic_number == "Pf" ? PFM_GREYSCALE : \
					NO_TYPE;

	p_pnm->filename = std::string(p_magicNum);
	p_pnm->width=0;p_pnm->height=0;p_pnm->max_value=0;
	std::size_t i = 2;
	//Separator is 0x0A 0x20
	
	fgets(p_buff, READ_BUFFER_SIZE, p_file);

	while ((	p_pnm->width==0 	||
				p_pnm->height==0 	||
				p_pnm->max_value==0) &&
				i<n_fileSize
	)
	{
		if(i<(n_fileSize-2-READ_BUFFER_SIZE){
			fread(p_buff, READ_BUFFER_SIZE, 1, p_file);
			i+=READ_BUFFER_SIZE;
		}
		else{
			fread(p_buff, n_fileSize- 2 - i % READ_BUFFER_SIZE, 1, p_file);
			i=n_fileSize;
		}
		for (std::size_t j = n_startSeparat; j < READ_BUFFER_SIZE; j++)
		{
			if (p_buff[i] == "." || p_buff[i] == " "){
				n_startSeparat = j;
				break;
			}
		}
		for (std::size_t j = n_startSeparat; j < READ_BUFFER_SIZE; j++)
		{
			if (p_buff[i] == "." || p_buff[i] == " "){
				n_startSeparat = j;
				break;
			}
		}
		n_startSeparat=FindCharInString(p_buff,".",n_startSeparat,n_endSeparat-1);
		n_endSeparat=FindCharInString(p_buff,".",n_startSeparat,n_endSeparat-1);
	}
	
	n_startSeparat=FindCharInString(p_buff,".",0,1);
	for (std::size_t i = 2; i < n_fileSize; i += READ_BUFFER_SIZE)
	{
		fread(p_buff, READ_BUFFER_SIZE, 1, p_file);
		for (std::size_t j = n_startSeparat; j < count; j++)
		{
			/* code */
		}
		
		n_startSeparat=FindCharInString(p_buff,".",0,1);
		n_endSeparat
	}
};
} // namespace detail




} // namespace pnm_io
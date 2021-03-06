﻿/*

	All cheak use assert and should define NDEBUG to prohibit it.
	(Or use release config)
*/

#include <iostream>

#ifdef DEBUG

#include <utils/catch_utils.h>
#include <catch2/catch.hpp>

#endif // DEBUG

#include "include/PNM_IO.h"
#include "include/PixelEngine.h"

int main(int argc, char **argv)
{
	bool b_help = false;
	std::string s_inputFileName("input.pgm"), s_outputFileName("output.pgm");
	bool b_easypgm = true;
	bool b_mypgmtoppm = false;
	myfloat f_R, f_B, f_G;
	f_R = f_B = f_G = 1.0f;
	bool b_mypgmflipV = false;
	bool b_myppmflipH = false;
	bool b_mypgmscale = false;
	myfloat f_scaleFactor = 0.5;
	bool b_mypgmrotate = false;
	myfloat f_angle = 30.0f;
	bool b_mypgmsmooth = false;
	bool b_mysobel = false;
	/*
	CmdArgsMap cmdArgs = CmdArgsMap(argc, argv, "--")("help", "Produce help message", &b_help)\
	("input", "--input InputFileName: Need to specify an input file.(Default is input.pgm).", &s_inputFileName, s_inputFileName)\
	("easypgm", "--easypgm OutputFileName: Print image size and save as OutputFileName PGM file.", &s_outputFileName, s_outputFileName, &b_easypgm)\
	("mypgmtoppm", "--mypgmtoppm OutputFileName: Convert the image into an RGB image and produces a PPM file.", &s_outputFileName, s_outputFileName, &b_mypgmtoppm)\
	("R", "--R float: R channel conversion parameters. (Default 1.0f)", &f_R, f_R)\
	("B", "--B float: B channel conversion parameters. (Default 1.0f)", &f_B, f_B)\
	("G", "--G float: G channel conversion parameters. (Default 1.0f)", &f_G, f_G)\
	("mypgmflipV", "--mypgmflipV OutputFileName: Flip the image vertically and save as OutputFileName PGM file.", &s_outputFileName, s_outputFileName, &b_mypgmflipV)\
	("myppmflipH", "--mypgmflipH OutputFileName: Flip the image horizontally and save as OutputFileName PGM file.", &s_outputFileName, s_outputFileName, &b_myppmflipH)\
	("mypgmscale", "--mypgmscale OutputFileName: Shrinks or enlarges the image and save as OutputFileName PGM file.", &s_outputFileName, s_outputFileName, &b_mypgmscale)\
	("factor", "--factor float: Mypgmscale scale factor. (Default 0.5f)", &f_scaleFactor, f_scaleFactor)\
	("mypgmrotate", "--mypgmrotate OutputFileName: Rotate the image and save as OutputFileName PGM file.", &s_outputFileName, s_outputFileName, &b_mypgmrotate)\
	("angle", "--angle float: Mypgmrotate angle. (Default 30.0f)", &f_angle, f_angle)\
	("mypgmsmooth", "--mypgmrotate OutputFileName: Apply smoothing and save as OutputFileName PGM file.", &s_outputFileName, s_outputFileName, &b_mypgmsmooth)\
	("mysobel", "--mysobel OutputFileName: Apply Sobel and Laplacian edge detector and save as OutputFileName PGM file.", &s_outputFileName, s_outputFileName, &b_mysobel)
	;*/

	peg::PixelEngine n_PixelEngine;
	PNM m_org, m_out;
	m_org.filename = (char *)s_inputFileName.c_str();
	m_org.data = NULL;
	PNM_IO.ReadPNMFile(&m_org,128);
	m_out.filename = (char *)s_outputFileName.c_str();
	m_out.width = m_org.width;
	m_out.height = m_org.height;
	m_out.maxValue = m_org.maxValue;
	m_out.threshold = m_org.threshold;
	m_out.type = m_org.type;
	m_out.data = NULL;
	std::vector<std::string> slist = { s_inputFileName ,s_outputFileName };

	if (b_easypgm)
	{
		// Print image size and save as OutputFileName PGM file
		std::cout << "File :" << m_org.filename << std::endl
				  << "Width is :" << m_org.width << std::endl
				  << "Height is :" << m_org.height << std::endl;
		m_out.data = (uchar*)malloc(m_org.height*m_org.width*m_org.sizePrePixel);

		memcpy(m_out.data,m_org.data, m_org.height * m_org.width * m_org.sizePrePixel);
		
		PNM_IO.WritePGMFile(&m_out);
	}
	else if (b_mypgmtoppm)
	{
		std::vector<myfloat> pa = { 1.0f, 1.0f, 1.0f };
		// Convert the image into an RGB image and produces a PPM file
		PNM_IO.Greyscale2RGB(m_org.width,m_org.height,f_R,f_G,f_B, &m_org.data,&m_out.data);
		PNM_IO.WritePNMFile(&m_out,0);
		//thinks::pnm_io::Greyscale2RGB(m_org.width, m_org.height, f_R, f_G, f_B, &m_out.data, &m_org.data);
		//thinks::pnm_io::WritePpmImage(m_out.filename, m_out.width, m_out.height, m_out.data.data());
	}
	else if (b_mypgmflipV)
	{
		// Flip the image vertically and save as OutputFileName PGM file
		peg::Pixels n_Pixels = {m_org.width, m_org.height, 1, std::vector<std::uint8_t>(m_org.data, m_org.data + sizeof(m_org.data)) };
		n_PixelEngine.flip(&n_Pixels, 0, 1);
		m_out.data = n_Pixels.data.data();
		PNM_IO.WritePGMFile(&m_out);
		//thinks::pnm_io::WritePgmImage(m_out.filename, n_Pixels.width, n_Pixels.height, n_Pixels.data.data());
		//thinks::pnm_io::WritePgmImage(m_out.filename, n_Pixels.width, n_Pixels.height, n_Pixels.data.data());
	}
	else if (b_myppmflipH)
	{
		// Flip the image horizontally and save as OutputFileName PGM file
		peg::Pixels n_Pixels = {m_org.width, m_org.height, 1, std::vector<std::uint8_t>(m_org.data, m_org.data + sizeof(m_org.data)) };
		n_PixelEngine.flip(&n_Pixels, 1, 1);
		m_out.data = n_Pixels.data.data();
		PNM_IO.WritePGMFile(&m_out);
		//thinks::pnm_io::WritePgmImage(m_out.filename, n_Pixels.width, n_Pixels.height, n_Pixels.data.data());
	}
	else if (b_mypgmscale)
	{
		// Shrinks or enlarges the image and save as OutputFileName PGM file
		peg::Pixels n_Pixels = {m_org.width, m_org.height, 1, std::vector<std::uint8_t>(m_org.data, m_org.data + sizeof(m_org.data)) };
		n_PixelEngine.resize(&n_Pixels, n_Pixels.width * f_scaleFactor, n_Pixels.height * f_scaleFactor, 0);
		m_out.data = n_Pixels.data.data();
		m_out.width = n_Pixels.width;
		m_out.height = n_Pixels.height;
		PNM_IO.WritePGMFile(&m_out);
		//thinks::pnm_io::WritePgmImage(m_out.filename, n_Pixels.width, n_Pixels.height, n_Pixels.data.data());
	}
	else if (b_mypgmrotate)
	{
		// Rotate the image and save as OutputFileName PGM file.
		peg::Pixels n_Pixels = {m_org.width, m_org.height, 1, std::vector<std::uint8_t>(m_org.data, m_org.data + sizeof(m_org.data)) };
		n_PixelEngine.rotate(&n_Pixels, f_angle, 0);
		m_out.data = n_Pixels.data.data();
		m_out.width = n_Pixels.width;
		m_out.height = n_Pixels.height;
		PNM_IO.WritePGMFile(&m_out);
		//thinks::pnm_io::WritePgmImage(m_out.filename, n_Pixels.width, n_Pixels.height, n_Pixels.data.data());
	}
	else if (b_mypgmsmooth)
	{
		std::string s_name(m_out.filename);
		std::string s_finName;
		// Flip the image vertically and save as OutputFileName PGM file
		peg::Pixels n_Pixels = {m_org.width, m_org.height, 1, std::vector<std::uint8_t>(m_org.data, m_org.data + sizeof(m_org.data)) };
		peg::Matrix n_Matrix = {3,3,
								{1.0f, 1.0f, 1.0f,
								 1.0f, 1.0f, 1.0f,
								 1.0f, 1.0f, 1.0f}};
		n_PixelEngine.smooth(&n_Pixels, &n_Matrix, 1 / 9.0f);
		m_out.data = n_Pixels.data.data();
		m_out.width = n_Pixels.width;
		m_out.height = n_Pixels.height;
		s_finName = "Averaging_" + s_name;
		m_out.filename = (char*)s_finName.c_str();
		PNM_IO.WritePGMFile(&m_out);
		//thinks::pnm_io::WritePgmImage("Averaging_" + m_out.filename, n_Pixels.width, n_Pixels.height, n_Pixels.data.data());
		n_Pixels.data = std::vector<std::uint8_t>(m_org.data, m_org.data + sizeof(m_org.data));
		n_Matrix = {3,3,
					{1.0f, 1.0f, 1.0f,
					 1.0f, 2.0f, 1.0f,
					 1.0f, 1.0f, 1.0f}};
		n_PixelEngine.smooth(&n_Pixels, &n_Matrix, 1 / 10.0f);
		m_out.data = n_Pixels.data.data();
		m_out.width = n_Pixels.width;
		m_out.height = n_Pixels.height;
		s_finName = "Averaging_centerEmphasize_" + s_name;
		m_out.filename = (char*)s_finName.c_str();
		PNM_IO.WritePGMFile(&m_out);
		m_out.filename = (char*)s_name.c_str();
		//thinks::pnm_io::WritePgmImage("Averaging_centerEmphasize_" + m_out.filename, n_Pixels.width, n_Pixels.height, n_Pixels.data.data());
	}
	else if (b_mysobel)
	{
		// Apply Sobel and Laplacian edge detector and save as OutputFileName PGM file.
		std::string s_name(m_out.filename);
		std::string s_finName;
		peg::Pixels n_Pixels = {m_org.width, m_org.height, 1, std::vector<std::uint8_t>(m_org.data, m_org.data + sizeof(m_org.data)) };
		peg::Matrix n_Matrix1 = {3,3,
								 {1.0f, 0.0f, -1.0f,
								  2.0f, 0.0f, -2.0f,
								  1.0f, 0.0f, -1.0f}};
		peg::Matrix n_Matrix2 = {3,3,
								 {1.0f, 2.0f, 1.0f,
								  0.0f, 0.0f, 0.0f,
								  -1.0f, -2.0f, -1.0f}};
		n_PixelEngine.smooth2D(&n_Pixels, &n_Matrix1, &n_Matrix2, 1.0f, 1.0f);
		//thinks::pnm_io::WritePgmImage("Sobel_" + m_out.filename, n_Pixels.width, n_Pixels.height, n_Pixels.data.data());
		m_out.data = n_Pixels.data.data();
		m_out.width = n_Pixels.width;
		m_out.height = n_Pixels.height;
		s_finName = "Sobel_" + s_name;
		m_out.filename = (char*)s_finName.c_str();
		PNM_IO.WritePGMFile(&m_out);

		n_Pixels.data = std::vector<std::uint8_t>(m_org.data, m_org.data + sizeof(m_org.data));
		n_Matrix1 = {3,3,
					 {-1.0f, -1.0f, -1.0f,
					  -1.0f, 8.0f, -1.0f,
					  -1.0f, -1.0f, -1.0f}};
		n_PixelEngine.smooth(&n_Pixels, &n_Matrix1, 1 / 10.0f);
		m_out.data = n_Pixels.data.data();
		m_out.width = n_Pixels.width;
		m_out.height = n_Pixels.height;
		s_finName = "Laplacian_" + s_name;
		m_out.filename = (char*)s_finName.c_str();
		PNM_IO.WritePGMFile(&m_out);
		m_out.filename = (char*)s_name.c_str();
		//thinks::pnm_io::WritePgmImage("Laplacian_" + m_out.filename, n_Pixels.width, n_Pixels.height, n_Pixels.data.data());
	}
	else
	{
		//std::cout << cmdArgs.help() << std::endl;
	}
}

#include "include/PNM_IO.h"

#define mymalloc(buff, size){		\
	if (!buff)						\
	{								\
		buff = (u_char*)malloc(size);\
		if (!buff)					\
		{							\
			buff = NULL;			\
			return PNM_MEMERY_INSUFFICIENT;\
		}							\
	}								\
	else if (sizeof(buff) < size)	\
	{								\
		if (sizeof(buff))			\
			free(buff);				\
		buff = (u_char*)malloc(size);\
		if (!buff)					\
			return PNM_MEMERY_INSUFFICIENT;\
	}}								\

void PNMTYPE2MagNum(u_char* dst, PNMTYPE type) {
	dst[0] = 'P';
	switch (type)
	{
	case PBM_ASCII:
		dst[1] = '1';
		break;
	case PGM_ASCII:
		dst[1] = '2';
		break;
	case PPM_ASCII:
		dst[1] = '3';
		break;
	case PBM_BINARY:
		dst[1] = '4';
		break;
	case PGM_BINARY:
		dst[1] = '5';
		break;
	case PPM_BINARY:
		dst[1] = '6';
		break;
	case PAM:
		dst[1] = '7';
		break;
	case PFM_RGB:
		dst[1] = 'F';
		break;
	case PFM_GREYSCALE:
		dst[1] = 'f';
		break;
	default:
		break;
	}
}
#define MagNum2PNMTYPE(magicNum) \
	magicNum[0] != 'P' ? NO_TYPE : magicNum[1] == '1' ? PBM_ASCII : magicNum[1] == '2' ? PGM_ASCII : magicNum[1] == '3' ? PPM_ASCII : magicNum[1] == '4' ? PBM_ASCII : magicNum[1] == '5' ? PGM_ASCII : magicNum[1] == '6' ? PPM_ASCII : magicNum[1] == '7' ? PAM : magicNum[1] == 'F' ? PFM_RGB : magicNum[1] == 'f' ? PFM_GREYSCALE : NO_TYPE;

#define myassert(expression) \
	if (!(expression))       \
		return PNM_RT_ERR;

/*
#ifdef NDEBUG
#else
#define myassert(expression) assert(expression)
#endif // NDEBUG
*/

#define clamp(min_value, max_value, value) \
	value < min_value ? min_value : (value > max_value ? max_value : value);

#define getFileLen(size, pfile) \
	fseek(pfile, 0, SEEK_END);  \
	size = ftell(pfile);        \
	fseek(pfile, 0, SEEK_SET);

PNM_STATE ReadPNMFile(PNM* f, uint16_t const pa);
PNM_STATE ReadPBMFile(PNM* f, uint16_t const pa);
PNM_STATE ReadPGMFile(PNM* f);
PNM_STATE ReadPPMFile(PNM* f);

PNM_STATE WritePNMFile(PNM* f, uint16_t const pa);
PNM_STATE WritePBMFile(PNM* f, uint16_t const pa);
PNM_STATE WritePGMFile(PNM* f);
PNM_STATE WritePPMFile(PNM* f);

PNM_STATE ReadHeader(PNM* f, FILE* is);
PNM_STATE ReadPixelData(PNM* f, FILE* is);
PNM_STATE WriteHeader(PNM* f, FILE* os);
PNM_STATE Greyscale2RGB(size_t const width,
	size_t const height,
	myfloat const fr,
	myfloat const fg,
	myfloat const fb,
	u_char** greyscale_pixel_data,
	u_char** rgb_pixel_data);

PNM_STATE RGB2Greyscale(size_t const width,
	size_t const height,
	u_char** rgb_pixel_data,
	u_char** greyscale_pixel_data);

PNM_STATE BitMap2Greyscale(size_t const width,
	size_t const height,
	uint8_t threshold,
	u_char** bit_map_data,
	u_char** greyscale_pixel_data);

PNM_STATE Greyscale2BitMap(size_t const width,
	size_t const height,
	uint16_t threshold,
	u_char** greyscale_pixel_data,
	u_char** bit_map_data);

PNM_STATE ReadPNMFile(PNM* f, uint16_t const pa)
{
	myassert(f->filename);
	PNM_STATE s_this;
	FILE* pf_read;
	pf_read = fopen(f->filename, "rb");
	if (!pf_read)return PNM_RT_ERR;

	s_this = ReadHeader(f, pf_read);
	if (s_this != PNM_SUCCESS)
		return s_this;

	if (f->type == NO_TYPE)
		return PNM_FILE_FORMAT_ERR;
	if (f->type == PBM_ASCII || f->type == PBM_BINARY) {
		ReadPBMFile(f, pa);
	}
	else {
		ReadPixelData(f, pf_read);
	}
	fclose(pf_read);
	pf_read = NULL;
	return PNM_SUCCESS;
}

PNM_STATE ReadPBMFile(PNM* f, uint16_t const pa)
{
	myassert(f->data);
	u_char* buff;
	FILE* pf_read;

	pf_read = fopen(f->filename, "rb");
	if (!pf_read)return PNM_RT_ERR;
	ReadHeader(f, pf_read);

	if (f->type == NO_TYPE || f->type != PBM_ASCII || f->type != PBM_BINARY) {
		f->width = f->height = f->maxValue = 0;
		f->type = NO_TYPE;
		return PNM_FILE_FORMAT_ERR;
	}

	size_t n_start, n_size;
	getFileLen(n_size, pf_read);
	n_start = n_size - f->height * (f->width / 8 + f->width % 8);

	mymalloc(buff, n_size - n_start);

	if (fread(buff, n_size - n_start, 1, pf_read) < (n_size - n_start)) {
		free(buff);
		f->width = f->height = f->maxValue = 0;
		f->type = NO_TYPE;
		return PNM_RT_ERR;
	}

	BitMap2Greyscale(f->width, f->height, pa, &buff, &f->data);
	f->threshold = pa;
	fclose(pf_read);
	free(buff);
	return PNM_SUCCESS;
}

PNM_STATE ReadPGMFile(PNM* f)
{
	FILE* pf_read;

	pf_read = fopen(f->filename, "rb");
	if (!pf_read)return PNM_RT_ERR;
	ReadHeader(f, pf_read);

	if (f->type == NO_TYPE)
		return PNM_FILE_FORMAT_ERR;
	ReadPixelData(f, pf_read);

	fclose(pf_read);
	return PNM_SUCCESS;
}

PNM_STATE ReadPPMFile(PNM* f)
{
	FILE* pf_read;

	pf_read = fopen(f->filename, "rb");
	if (!pf_read)return PNM_RT_ERR;
	ReadHeader(f, pf_read);

	if (f->type == NO_TYPE)
		return PNM_FILE_FORMAT_ERR;

	ReadPixelData(f, pf_read);
	fclose(pf_read);
	return PNM_SUCCESS;
}

PNM_STATE WritePNMFile(PNM* f, uint16_t const pa)
{
	myassert(f->type != NO_TYPE);
	if (f->type == PBM_ASCII || f->type == PBM_BINARY)return WritePBMFile(f, pa);

	FILE* pf_read;
	pf_read = fopen(f->filename, "wb");
	if (!pf_read)return PNM_RT_ERR;

	WriteHeader(f, pf_read);
	fwrite(f->data, sizeof(f->data), 1, pf_read);
	fclose(pf_read);
	return PNM_SUCCESS;
}

PNM_STATE WritePBMFile(PNM* f, uint16_t const pa)
{
	myassert(f->type == PBM_ASCII || f->type == PBM_BINARY);
	FILE* pf_read;
	pf_read = fopen(f->filename, "wb");
	if (!pf_read)return PNM_RT_ERR;

	if (WriteHeader(f, pf_read) != PNM_SUCCESS) {
		fclose(pf_read);
		return PNM_RT_ERR;
	}
	u_char* buff;
	if (PNM_SUCCESS != Greyscale2BitMap(f->width, f->height, pa, &f->data, &buff)) { 
		fclose(pf_read);
		return PNM_MEMERY_INSUFFICIENT; 
	}

	fwrite(buff,sizeof(buff),1,pf_read);

	fclose(pf_read);
	return PNM_SUCCESS;
}

PNM_STATE WritePGMFile(PNM* f)
{
	myassert(f->type == PGM_ASCII || f->type == PGM_BINARY);
	return WritePNMFile(f, 0);
}

PNM_STATE WritePPMFile(PNM* f)
{
	myassert(f->type == PPM_ASCII || f->type == PPM_BINARY);
	return WritePNMFile(f , 0);
}


/*
	Will return PNM_RT_ERR when read error value
*/
PNM_STATE ReadHeader(PNM* f, FILE* is)
{
	u_char buff[20];
	uint16_t n_size;
	getFileLen(n_size, is);

	// Frist 2 Byte magic number
	fread(f->magicNumber, 2, 1, is);
	n_size -= 2;
	if (f->magicNumber[0] != 'P')return PNM_RT_ERR;
	f->type = MagNum2PNMTYPE(f->magicNumber);
	if (f->type == NO_TYPE)return PNM_RT_ERR;

	uint8_t s = 0, e = 0, i = 0;
	bool ishit = false;
	while (n_size > (0 + sizeof(buff))) {

		fread(buff, sizeof(buff), 1, is);
		sscanf((const char*)buff, "%d.%d.%d", &f->width, &f->height, &f->maxValue);
		if (f->width == 0 || f->height == 0 || f->maxValue == 0) {
			sscanf((const char*)buff, "%d %d %d", &f->width, &f->height, &f->maxValue);
		}
		if (f->width == 0 || f->height == 0 || f->maxValue == 0) {
			n_size -= sizeof(buff);
			continue;
		}
		else break;
	}
	if (n_size <= (0 + sizeof(buff))) {
		f->width = f->height = f->maxValue = 0;
		f->type = NO_TYPE;
		return PNM_FILE_FORMAT_ERR;
	}

	f->threshold = f->maxValue / 2;

#ifdef NDEBUG
	if (!(header.width && header.height && header.max_value))
	{
		header.type = NO_TYPE;
		return PNM_RT_ERR;
	}
#endif // NDEBUG
	myassert(f->width != 0 || f->height != 0 || f->maxValue != 0);
	f->type = MagNum2PNMTYPE(f->magicNumber);
	return PNM_SUCCESS;
}

PNM_STATE ReadPixelData(PNM* f, FILE* is)
{
	myassert(is && f->height && f->width);
	myassert(f->type != NO_TYPE);

	size_t n_start, n_size;
	getFileLen(n_size, is);
	if (f->type == PPM_ASCII || f->type == PPM_BINARY) {
		n_start = n_size - 3 * f->height * f->width;
	}
	else if (f->type == PBM_ASCII || f->type == PBM_BINARY) {
		n_start = n_size - f->height * (f->width / 8 + f->width % 8);
	}
	else {
		n_start = n_size - f->height * f->width;
	}
	fseek(is, n_start-1, SEEK_SET);

	if (!f->data || sizeof(f->data) < (n_size - n_start))
		mymalloc(f->data, n_size - n_start);

	if (fread(f->data, n_size - n_start, 1, is) < (n_size - n_start))
		return PNM_RT_ERR;
	return PNM_SUCCESS;
}

PNM_STATE WriteHeader(PNM* f, FILE* os)
{
	myassert(f->width && f->height && f->maxValue);
	if (f->magicNumber)
	{
		if (f->type == NO_TYPE)
			return PNM_RT_ERR;
		//strcpy_s(f->magicNumber, 2, PNMTYPE2MagNum(f->type));
		PNMTYPE2MagNum(f->magicNumber, f->type);
		
	}
	u_char po[1] = { '.' };
	fwrite(f->magicNumber,2,1,os);
	fwrite(po, 1, 1, os);
	fwrite(&f->width, 2, 1, os);
	fwrite(po, 1, 1, os);
	fwrite(&f->height, 2, 1, os);
	fwrite(po, 1, 1, os);
	fwrite(&f->maxValue, 2, 1, os);
	fwrite(po, 1, 1, os);

	return PNM_SUCCESS;
}

/*!
Convert Greyscale to RGB while :
	R = fr * Gr
	G = fg * Gr
	B = fb * Gr


Will return PNM_MEMERY_INSUFFICIENT: Not enough memery
			PNM_RT_ERR             : Parameter error
			PNM_SUCCESS            : Success
*/
PNM_STATE Greyscale2RGB(
	size_t const width,
	size_t const height,
	myfloat const fr,
	myfloat const fg,
	myfloat const fb,
	u_char** const greyscale_pixel_data,
	u_char** const rgb_pixel_data)
{
	myassert((*greyscale_pixel_data) != NULL);
	myassert(sizeof((*greyscale_pixel_data)) >= (width * height));
	mymalloc(*rgb_pixel_data, width * height * 3);

	for (size_t row = 0; row < height; ++row)
	{
		for (size_t col = 0; col < width; ++col)
		{
			(*rgb_pixel_data)[3 * (col + row * width)] = (*greyscale_pixel_data)[col + row * width] * clamp(0.0f, 1.0f, fr);
			(*rgb_pixel_data)[3 * (col + row * width) + 1] = (*greyscale_pixel_data)[col + row * width] * clamp(0.0f, 1.0f, fb);
			(*rgb_pixel_data)[3 * (col + row * width) + 2] = (*greyscale_pixel_data)[col + row * width] * clamp(0.0f, 1.0f, fg);
		}
	}
	return PNM_SUCCESS;
}

/*!
Convert RGB to Greyscale while Gr=(R+G+B)/3

Will return PNM_MEMERY_INSUFFICIENT: Not enough memery
			PNM_RT_ERR             : Parameter error
			PNM_SUCCESS            : Success
*/
PNM_STATE RGB2Greyscale(
	size_t const width,
	size_t const height,
	u_char** const rgb_pixel_data,
	u_char** const greyscale_pixel_data)
{
	myassert((*rgb_pixel_data) != NULL);
	myassert(sizeof((*rgb_pixel_data)) >= (width * height) * 3);
	mymalloc(*greyscale_pixel_data, width * height);

	for (size_t row = 0; row < height; ++row)
	{
		for (size_t col = 0; col < width; ++col)
		{
			(*greyscale_pixel_data)[col + row * width] = ((*rgb_pixel_data)[3 * (col + row * width)] +
				(*rgb_pixel_data)[3 * (col + row * width) + 1] +
				(*rgb_pixel_data)[3 * (col + row * width) + 2]) / 3;
		}
	}
	return PNM_SUCCESS;
}

PNM_STATE BitMap2Greyscale(
	size_t const width,
	size_t const height,
	uint8_t threshold,
	u_char** const bit_map_data,
	u_char** const greyscale_pixel_data)
{
	myassert(width && height && threshold);
	myassert((*bit_map_data) != NULL);
	myassert(sizeof((*bit_map_data)) >= (width * height) / 8);
	threshold = threshold ? threshold : 128;
	mymalloc(*greyscale_pixel_data, width * height);

	for (size_t row = 0; row < height; ++row)
	{
		for (size_t col = 0; col < width; ++col)
		{

			for (size_t k = 0; k < 8; ++k)
			{
				(*greyscale_pixel_data)[col + row * width] =
					(((*bit_map_data)[(col + row * width) / 8] >> (7 - k)) & 0x1) ? threshold : 0;
			}
		}
	}
	return PNM_SUCCESS;
}

PNM_STATE Greyscale2BitMap(
	size_t const width,
	size_t const height,
	uint16_t threshold,
	u_char** const greyscale_pixel_data,
	u_char** const bit_map_data)
{
	myassert(width && height && threshold);
	myassert((*greyscale_pixel_data) != NULL);
	myassert(sizeof((*greyscale_pixel_data)) >= (width * height) / 8);
	threshold = threshold ? threshold : 128;
	mymalloc(*bit_map_data, width * height);

	for (size_t row = 0; row < height; ++row)
	{
		for (size_t col = 0; col < width; ++col)
		{
			for (size_t k = 0; k < 8; ++k)
			{
				(*bit_map_data)[col + row * width / 8] =
					((*greyscale_pixel_data)[(col + row * width)] > threshold ? 0x1 : 0x0) << (7 - k);
			}
		}
	}

	return PNM_SUCCESS;
}

const struct pNM_IO PNM_IO = {
	ReadPNMFile : ReadPNMFile,
	ReadPBMFile : ReadPBMFile,
    ReadPGMFile : ReadPGMFile,
    ReadPPMFile : ReadPPMFile,
    WritePNMFile : WritePNMFile,
    WritePBMFile : WritePBMFile,
    WritePGMFile : WritePGMFile,
    WritePPMFile : WritePPMFile,
    RGB2Greyscale : RGB2Greyscale,
    Greyscale2RGB : Greyscale2RGB,
    BitMap2Greyscale : BitMap2Greyscale,
    Greyscale2BitMap : Greyscale2BitMap,
	n_pnmGlobalState : PNM_SUCCESS,
};


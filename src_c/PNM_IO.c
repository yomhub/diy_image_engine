#include "include/PNM_IO.h"

PNM_STATE ReadPNMFile(PNM *f, uint16_t pa)
{
	myassert(f->filename);
    PNM_STATE s_this;
    FILE *pf_read;
    pf_read = fopen(f->filename,"rb");
    if(!pf_read)return PNM_RT_ERR;

    s_this = ReadHeader(f, pf_read);
    if (s_this != PNM_SUCCESS)
        return s_this;

    if (f->type == NO_TYPE)
        return PNM_FILE_FORMAT_ERR;
    if (f->type == PBM_ASCII || f->type == PBM_BINARY){
        ReadPBMFile(f, pf_read, pa);
    }else{
        ReadPixelData(f, pf_read);
    }
	fclose(pf_read);
	pf_read = NULL;
    return PNM_SUCCESS;
}

PNM_STATE ReadPBMFile(PNM *f, FILE* is, uint16_t const pa)
{
    myassert(f->data);
	uchar *buff;
    if (f->type == NO_TYPE)
    {
        OpenFileStream(f, is);
        ReadHeader(f, is);
    }
    if (f->type == NO_TYPE)
        return PNM_FILE_FORMAT_ERR;
    ReadPixelData(f, is);

	size_t n_start, n_size;
	n_start = f_istream.tellg();
	f_istream.seekg(0, f_istream.end);
	n_size = (std::size_t)f_istream.tellg() - n_start;
	f_istream.seekg(n_start, f_istream.beg);

	if (buff.size() < n_size)
		buff.resize(n_size);
	if (buff.size() < n_size) {
		f->type = NO_TYPE;
		return PNM_MEMERY_INSUFFICIENT;
	}

	f_istream.read(reinterpret_cast<char *>(buff.data()), n_size);

	if (!f_istream) {
		f->type = NO_TYPE;
		return PNM_RT_ERR;
	}

	BitMap2Greyscale(f->width,f->height,pa,&buff,&f->data);
	f->threshold = pa;
	f_istream.close();
    return PNM_SUCCESS;
}

PNM_STATE ReadPGMFile(PNM *f)
{
    if (f->type == NO_TYPE)
    {
        OpenFileStream(f, &f_istream);
        ReadHeader(f, f_istream);
    }
    if (f->type == NO_TYPE)
        return PNM_FILE_FORMAT_ERR;
    ReadPixelData(f, f_istream);
	f_istream.close();
    return PNM_SUCCESS;
}

PNM_STATE ReadPPMFile(PNM *f)
{
    if (f->type == NO_TYPE)
    {
        OpenFileStream(f, &f_istream);
        ReadHeader(f, f_istream);
    }
    if (f->type == NO_TYPE)
        return PNM_FILE_FORMAT_ERR;
    ReadPixelData(f, f_istream);
	f_istream.close();
    return PNM_SUCCESS;
}

PNM_STATE WritePNMFile(PNM *f, uint16_t const pa)
{
    myassert(f->type != NO_TYPE, "Write err: PPM type err");
	if (f->type == PBM_ASCII || f->type == PBM_BINARY)return WritePBMFile(f,pa);
    OpenFileStream(f, &f_ostream);
    WriteHeader(f, f_ostream);
    WritePixelData(f, f_ostream);
	f_ostream.close();
    return PNM_SUCCESS;
}

PNM_STATE WritePBMFile(PNM *f, uint16_t const pa)
{
    myassert(f->type == PBM_ASCII || f->type == PBM_BINARY, "Write err: PPM type err");
	OpenFileStream(f, &f_ostream);
	WriteHeader(f, f_ostream);
	std::vector<std::uint8_t>buff;
	Greyscale2BitMap(f->width,f->height,pa,&f->data,&buff);

	f_ostream.write(reinterpret_cast<char const *>(buff.data()), buff.size());
	f_ostream.close();
	return PNM_SUCCESS;
}

PNM_STATE WritePGMFile(PNM *f)
{
    myassert(f->type == PGM_ASCII || f->type == PGM_BINARY, "Write err: PPM type err");
    return WritePNMFile(f);
}

PNM_STATE WritePPMFile(PNM *f)
{
    myassert(f->type == PPM_ASCII || f->type == PPM_BINARY, "Write err: PPM type err");
    return WritePNMFile(f);
}

PNM_STATE ConvertFormat(PNM *src, PNM *dst, std::vector<myfloat> const *pa)
{
    myassert(src->type != NO_TYPE && dst->type != NO_TYPE, "Type err at ConvertFormat");
	myassert(src->data.data() != nullptr , "NULL ptr at ConvertFormat");

	if ((src->type == PGM_ASCII || src->type == PGM_BINARY) &&
		(dst->type == PPM_ASCII || dst->type == PPM_BINARY)) {
		dst->width = src->width;
		dst->height = src->height;
		if (pa->size() > 2) {
			n_pnmGlobalState = Greyscale2RGB(src->width, src->height, pa->data()[0], pa->data()[1], pa->data()[2], &src->data, &dst->data);
		}
		else {
			n_pnmGlobalState = Greyscale2RGB(src->width, src->height, 1.0f, 1.0f, 1.0f, &src->data, &dst->data);
		}
		if (n_pnmGlobalState != PNM_SUCCESS) {
			dst->data.clear();
			dst->data.shrink_to_fit();
		}
	}
	if ((src->type == PPM_ASCII || src->type == PPM_BINARY) &&
		(dst->type == PGM_ASCII || dst->type == PGM_BINARY)) {

		dst->width = src->width;
		dst->height = src->height;
		dst->magic_number = PNMTYPE2MagNum(dst->type);
		RGB2Greyscale(src->width, src->height, &src->data, &dst->data);

		if (n_pnmGlobalState != PNM_SUCCESS) {
			//std::vector<std::uint8_t>().swap(dst->data);
			dst->data.clear();
			dst->data.shrink_to_fit();
		}
	}
    return PNM_SUCCESS;
}

PNM_STATE ConvertFormat(PNM *f, PNMTYPE type, std::vector<myfloat> const *pa)
{
	myassert(f->type != NO_TYPE , "Type err at ConvertFormat");
	myassert(f->data.data() != nullptr , "NULL ptr at ConvertFormat");
	std::vector<std::uint8_t> buff;

	if ((f->type == PGM_ASCII || f->type == PGM_BINARY) &&
		(type == PPM_ASCII || type == PPM_BINARY)) {
		buff = f->data;
		if (pa->size() > 2) {
			n_pnmGlobalState = Greyscale2RGB(f->width, f->height, pa->data()[0], pa->data()[1], pa->data()[2], &buff, &f->data);
		}
		else {
			n_pnmGlobalState = Greyscale2RGB(f->width, f->height, 1.0f, 1.0f, 1.0f, &buff, &f->data);
		}
		if (n_pnmGlobalState != PNM_SUCCESS) {
			buff.swap(f->data);	
		}
		f->magic_number = PNMTYPE2MagNum(type);
		
	}
	if ((f->type == PPM_ASCII || f->type == PPM_BINARY) &&
		(type == PGM_ASCII || type == PGM_BINARY)) {
		buff = f->data;
		RGB2Greyscale(f->width, f->height, &buff, &f->data);
		if (n_pnmGlobalState != PNM_SUCCESS) {
			buff.swap(f->data);
		}
		f->magic_number = PNMTYPE2MagNum(type);
	}
    return PNM_SUCCESS;
}

void ThreadMain(void(*cbfun)(PNM f), std::vector<std::string> const * s_list)
{
	PNM n_pnm;
	for (auto s_fName : *s_list) {
		if (n_pnmThreadState == PNM_PAUSE) {
			while (true)
			{
				if (n_pnmThreadState != PNM_PAUSE)break;
				std::this_thread::sleep_for(std::chrono::seconds(1));
			}
		}
			
		if (n_pnmThreadState == PNM_WAIT_QUIT)break;
		if (s_fName.empty())continue;
		n_pnm.filename = s_fName;
		ReadPNMFile(&n_pnm);
		cbfun(n_pnm);
	}
	delete(&n_pnm);
	n_pnmThreadState = PNM_WAIT_DELETE;
}

PNM_STATE CreateTask(void (*cbfun)(PNM f), std::vector<std::string> const * s_list)
{
	
	if (n_pnmThreadState == PNM_WAIT_DELETE) {
		delete(p_mainThread);
		p_mainThread = NULL;
		return PNM_SUCCESS;
	}
	if (n_pnmThreadState != PNM_UNINITIALIZED)return PNM_THREAD_RUNNING;
	;
	p_mainThread = &std::thread(&PNM_IO::ThreadMain, this, (void (*)(PNM))cbfun, s_list);
	n_pnmThreadState = PNM_THREAD_RUNNING;
	return PNM_SUCCESS;
}

PNM_STATE StartTask() {
	if (n_pnmThreadState != PNM_THREAD_RUNNING)return PNM_UNINITIALIZED;
	n_pnmThreadState = PNM_THREAD_RUNNING;
	return PNM_SUCCESS;
}
PNM_STATE PauseTask()
{
	if (n_pnmThreadState != PNM_THREAD_RUNNING)return PNM_UNINITIALIZED;
	n_pnmThreadState = PNM_PAUSE;
	return PNM_SUCCESS;
}

PNM_STATE DeleteTask()
{
	if (p_mainThread && n_pnmThreadState == PNM_WAIT_DELETE) {
		delete(p_mainThread);
		p_mainThread = NULL;
		return PNM_SUCCESS;
	}
	if (n_pnmThreadState != PNM_THREAD_RUNNING)return PNM_UNINITIALIZED;
	n_pnmThreadState = PNM_WAIT_QUIT;
	p_mainThread->join();
	n_pnmThreadState = PNM_UNINITIALIZED;
	delete(p_mainThread);
	p_mainThread = NULL;
	return PNM_SUCCESS;
}

/*
	Will return PNM_RT_ERR when read error value
*/
PNM_STATE ReadHeader(PNM *f, FILE *is)
{
    uchar buff[20];
    uint16_t n_size;
    getFileLen(n_size,is);

	// Frist 2 Byte magic number
	fread(f->magicNumber,2,1,is);
	n_size -= 2;
	if (f->magicNumber[0] != 'P')return PNM_RT_ERR;
	f->type = MagNum2PNMTYPE(f->magicNumber);
	if (f->type == NO_TYPE)return PNM_RT_ERR;
					
	uint8_t s = 0, e = 0,i=0;
	bool ishit=false;
	while (n_size > (0+ sizeof(buff))) {

		fread(buff, sizeof(buff), 1, is);
		sscanf_s(buff, "%d.%d.%d", &f->width, &f->height, &f->maxValue);
		if (f->width == 0 || f->height == 0 || f->maxValue == 0) {
			sscanf_s(buff, "%d %d %d", &f->width, &f->height, &f->maxValue);
		}
		if (f->width == 0 || f->height == 0 || f->maxValue == 0) {
			n_size -= sizeof(buff);
			continue;
		}
		else break;
	}
	if (n_size > (0 + sizeof(buff))) {
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
    myassert(f->width != 0 || f->height != 0 || f->max_value != 0);
    return PNM_SUCCESS;
}

PNM_STATE ReadPixelData(PNM *f, FILE *is)
{
	myassert(is && f->height && f->width && f->data);
	myassert(f->type != NO_TYPE);

    size_t n_start, n_size;
	getFileLen(n_size, is);
	if (f->type == PPM_ASCII || f->type == PPM_BINARY) {
		n_start = n_size - 3 * f->height * f->width;
	}
	else if (f->type == PBM_ASCII || f->type == PBM_BINARY) {
		n_start = n_size - f->height * (f->width / 8+ f->width % 8);
	}
	else {
		n_start = n_size - f->height * f->width;
	}
	fseek(n_start, 0, SEEK_SET);

	if (sizeof(f->data) < (n_size - n_start))
		if(mymalloc(&f->data, n_size - n_start)== PNM_MEMERY_INSUFFICIENT)
			return PNM_MEMERY_INSUFFICIENT;

    if (fread(f->data, n_size - n_start, 1, is)< (n_size - n_start))
        return PNM_RT_ERR;
    return PNM_SUCCESS;
}

PNM_STATE WriteHeader(PNM *f, std::ostream &os)
{
    myassert(f->width && f->height && f->max_value, "Write data err");
    if (f->magic_number.empty())
    {
        if (f->type == NO_TYPE)
            return PNM_RT_ERR;
        f->magic_number = PNMTYPE2MagNum(f->type);
    }
    os << f->magic_number << "\n"
       << f->width << "\n"
       << f->height << "\n"
       << f->max_value << "\n"; // Marks beginning of pixel data.
    return PNM_SUCCESS;
}

PNM_STATE WritePixelData(PNM *f, std::ostream &os)
{
    myassert(f->data.data() != nullptr && f->data.size() > 0, "Write err");
    os.write(reinterpret_cast<char const *>(f->data.data()), f->data.size());
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
    std::size_t const width,
    std::size_t const height,
    myfloat const fr,
    myfloat const fg,
    myfloat const fb,
    std::vector<std::uint8_t> *const greyscale_pixel_data,
    std::vector<std::uint8_t> *const rgb_pixel_data)
{
    myassert(greyscale_pixel_data != nullptr, "null Greyscale pixel data");
    myassert(greyscale_pixel_data->size() >= (width * height), "Greyscale pixel have a wrong size");
    myassert(rgb_pixel_data != nullptr, "null RGB pixel data");

    if (rgb_pixel_data->size() < (width * height))
    {
        rgb_pixel_data->resize(width * height * 3, std::uint8_t{});
    }
    if (rgb_pixel_data->size() < (width * height))
        return PNM_MEMERY_INSUFFICIENT;

    for (auto row = std::size_t{0}; row < height; ++row)
    {
        for (auto col = std::size_t{0}; col < width; ++col)
        {
            rgb_pixel_data->data()[3 * (col + row * width)] = clamp(0.0f, 1.0f, fr) * greyscale_pixel_data->data()[col + row * width];
            rgb_pixel_data->data()[3 * (col + row * width) + 1] = clamp(0.0f, 1.0f, fb) * greyscale_pixel_data->data()[col + row * width];
            rgb_pixel_data->data()[3 * (col + row * width) + 2] = clamp(0.0f, 1.0f, fg) * greyscale_pixel_data->data()[col + row * width];
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
    std::size_t const width,
    std::size_t const height,
    std::vector<std::uint8_t> *const rgb_pixel_data,
    std::vector<std::uint8_t> *const greyscale_pixel_data)
{
    myassert(rgb_pixel_data != nullptr, "null RGB pixel data");
    myassert(rgb_pixel_data->size() >= (width * height) * 3, "RGB pixel have a wrong size");
    myassert(greyscale_pixel_data != nullptr, "null Greyscale pixel data");
    myassert(greyscale_pixel_data->size() >= (width * height), "Greyscale pixel have a wrong size");
    if (greyscale_pixel_data->size() < (width * height))
    {
        greyscale_pixel_data->resize(width * height, std::uint8_t{});
    }
    if (greyscale_pixel_data->size() < (width * height))
        return PNM_MEMERY_INSUFFICIENT;
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
    return PNM_SUCCESS;
}

PNM_STATE BitMap2Greyscale(
    std::size_t const width,
    std::size_t const height,
    std::uint8_t threshold,
    std::vector<std::uint8_t> *const bit_map_data,
    std::vector<std::uint8_t> *const greyscale_pixel_data)
{
    myassert(width && height && threshold, "null RGB pixel data");
    myassert(bit_map_data != nullptr, "null Greyscale pixel data");
    myassert(greyscale_pixel_data != nullptr, "Greyscale pixel have a wrong size");
    myassert(bit_map_data->size() >= (width * height) / 8, "RGB pixel have a wrong size");

    if (greyscale_pixel_data->size() < (width * height))
        greyscale_pixel_data->resize(width * height);
    if (greyscale_pixel_data->size() < (width * height))
        return PNM_MEMERY_INSUFFICIENT;

    for (auto row = std::size_t{0}; row < height; ++row)
    {
        for (auto col = std::size_t{0}; col < width; ++col)
        {

            for (auto k = std::size_t{0}; k < 8; ++k)
            {
                greyscale_pixel_data->data()[col + row * width] =
                    ((bit_map_data->data()[(col + row * width) / 8] >> (7 - k)) & 0x1) ? threshold : 0;
            }
        }
    }
    return PNM_SUCCESS;
}

PNM_STATE Greyscale2BitMap(
    std::size_t const width,
    std::size_t const height,
    uint16_t threshold,
    std::vector<std::uint8_t> *const greyscale_pixel_data,
    std::vector<std::uint8_t> *const bit_map_data)
{
    myassert(width && height && threshold, "null RGB pixel data");
    myassert(greyscale_pixel_data != nullptr, "null Greyscale pixel data");
    myassert(bit_map_data != nullptr, "Greyscale pixel have a wrong size");
    myassert(greyscale_pixel_data->size() >= (width * height), "RGB pixel have a wrong size");

    if (bit_map_data->size() < (width * height) / 8)
        bit_map_data->resize(width * height / 8);
    if (bit_map_data->size() < (width * height) / 8)
        return PNM_MEMERY_INSUFFICIENT;

    for (auto row = std::size_t{0}; row < height; ++row)
    {
        for (auto col = std::size_t{0}; col < width; ++col)
        {
            for (auto k = std::size_t{0}; k < 8; ++k)
            {
                bit_map_data->data()[col + row * width / 8] =
                    (greyscale_pixel_data->data()[(col + row * width)] > threshold ? 0x1 : 0x0) << (7 - k);
            }
        }
    }

    return PNM_SUCCESS;
}

const struct PNM_IO PNM_IO = {
    .ReadPNMFile = ReadPNMFile,
    .ReadPBMFile = ReadPBMFile,
    .n_pnmGlobalState = PNM_SUCCESS
};
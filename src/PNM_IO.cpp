#include "include/PNM_IO.h"
//static void 
//using namespace pnm_io;

pnm_io::PNM_IO::PNM_IO()
{
    p_mainThread = NULL;
    state_ = PNM_UNINITIALIZED;
    n_pnmGlobalState = PNM_SUCCESS;
	n_remainTask = 0;
}

pnm_io::PNM_IO::~PNM_IO()
{

	if (p_mainThread || state_ != PNM_UNINITIALIZED) {
		DeleteTask();
	}
    if (f_istream.is_open())
        f_istream.close();
    if (f_ostream.is_open())
        f_ostream.close();
}

template <typename FileStreamT>
pnm_io::PNM_STATE pnm_io::PNM_IO::OpenFileStream(PNM const *f, FileStreamT *f_stream, std::ios_base::openmode const mode)
{
    myassert(f_stream != nullptr, "FileStream is null");
    myassert(!f->filename.empty(), "File name is empty");

    if (f_stream->is_open())
        f_stream->close();
    f_stream->open(f->filename, mode);

    if (!f_stream->is_open())
        return PNM_RT_ERR;
    return PNM_SUCCESS;
}

pnm_io::PNM_STATE pnm_io::PNM_IO::ReadPNMFile(PNM *f, std::uint16_t const pa)
{
    n_pnmGlobalState = OpenFileStream(f, &f_istream);
    if (n_pnmGlobalState != PNM_SUCCESS)
        return n_pnmGlobalState;

    n_pnmGlobalState = ReadHeader(f, f_istream);
    if (n_pnmGlobalState != PNM_SUCCESS)
        return n_pnmGlobalState;

    if (f->type == NO_TYPE)
        return PNM_FILE_FORMAT_ERR;
    if (f->type == PBM_ASCII || f->type == PBM_BINARY){
        ReadPBMFile(f,pa);
    }else{
        ReadPixelData(f, f_istream);
    }
	f_istream.close();
    return PNM_SUCCESS;
}

pnm_io::PNM_STATE pnm_io::PNM_IO::ReadPBMFile(PNM *f, std::uint16_t const pa)
{
    myassert(f->data.data() != nullptr, "Read err");
	std::vector<std::uint8_t> buff;
    if (f->type == NO_TYPE)
    {
        OpenFileStream(f, &f_istream);
        ReadHeader(f, f_istream);
    }
    if (f->type == NO_TYPE)
        return PNM_FILE_FORMAT_ERR;
    ReadPixelData(f, f_istream);

	std::size_t n_start, n_size;
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

pnm_io::PNM_STATE pnm_io::PNM_IO::ReadPGMFile(PNM *f)
{
    if (f->type == NO_TYPE)
    {
        OpenFileStream(f, &f_istream);
        ReadHeader(f, f_istream);
    }
    if (f->type == NO_TYPE || f->type!= PGM_ASCII || f->type != PGM_BINARY)
        return PNM_FILE_FORMAT_ERR;
    ReadPixelData(f, f_istream);
	f_istream.close();
    return PNM_SUCCESS;
}

pnm_io::PNM_STATE pnm_io::PNM_IO::ReadPPMFile(PNM *f)
{
    if (f->type == NO_TYPE)
    {
        OpenFileStream(f, &f_istream);
        ReadHeader(f, f_istream);
    }
	if (f->type == NO_TYPE || f->type != PPM_ASCII || f->type != PPM_BINARY)
        return PNM_FILE_FORMAT_ERR;
    ReadPixelData(f, f_istream);
	f_istream.close();
    return PNM_SUCCESS;
}

pnm_io::PNM_STATE pnm_io::PNM_IO::WritePNMFile(PNM *f, std::uint16_t const pa)
{
    myassert(f->type != NO_TYPE, "Write err: PPM type err");
	if (f->type == PBM_ASCII || f->type == PBM_BINARY)return WritePBMFile(f,pa);
    OpenFileStream(f, &f_ostream);
    WriteHeader(f, f_ostream);
    WritePixelData(f, f_ostream);
	f_ostream.close();
    return PNM_SUCCESS;
}

pnm_io::PNM_STATE pnm_io::PNM_IO::WritePBMFile(PNM *f, std::uint16_t const pa)
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

pnm_io::PNM_STATE pnm_io::PNM_IO::WritePGMFile(PNM *f)
{
    myassert(f->type == PGM_ASCII || f->type == PGM_BINARY, "Write err: PPM type err");
    return WritePNMFile(f);
}

pnm_io::PNM_STATE pnm_io::PNM_IO::WritePPMFile(PNM *f)
{
    myassert(f->type == PPM_ASCII || f->type == PPM_BINARY, "Write err: PPM type err");
    return WritePNMFile(f);
}

pnm_io::PNM_STATE pnm_io::PNM_IO::ConvertFormat(PNM *src, PNM *dst, std::vector<myfloat> const *pa)
{
    myassert(src->type != NO_TYPE && dst->type != NO_TYPE, "Type err at pnm_io::PNM_IO::ConvertFormat");
	myassert(src->data.data() != nullptr , "NULL ptr at pnm_io::PNM_IO::ConvertFormat");

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
		dst->magicNumber = detail::PNMTYPE2MagNum(dst->type);
		RGB2Greyscale(src->width, src->height, &src->data, &dst->data);

		if (n_pnmGlobalState != PNM_SUCCESS) {
			//std::vector<std::uint8_t>().swap(dst->data);
			dst->data.clear();
			dst->data.shrink_to_fit();
		}
	}
    return PNM_SUCCESS;
}

pnm_io::PNM_STATE pnm_io::PNM_IO::ConvertFormat(PNM *f, PNMTYPE type, std::vector<myfloat> const *pa)
{
	myassert(f->type != NO_TYPE , "Type err at pnm_io::PNM_IO::ConvertFormat");
	myassert(f->data.data() != nullptr , "NULL ptr at pnm_io::PNM_IO::ConvertFormat");
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
		f->magicNumber = detail::PNMTYPE2MagNum(type);
		
	}
	if ((f->type == PPM_ASCII || f->type == PPM_BINARY) &&
		(type == PGM_ASCII || type == PGM_BINARY)) {
		buff = f->data;
		RGB2Greyscale(f->width, f->height, &buff, &f->data);
		if (n_pnmGlobalState != PNM_SUCCESS) {
			buff.swap(f->data);
		}
		f->magicNumber = detail::PNMTYPE2MagNum(type);
	}
    return PNM_SUCCESS;
}

void pnm_io::PNM_IO::ThreadMain(void(*cbfun)(PNM f), std::vector<std::string> * s_list)
{
	PNM n_pnm;
	for (std::string s_fName : *s_list) {
		if (state_ == PNM_PAUSE) {
			while (true)
			{
				if (state_ != PNM_PAUSE)break;
				std::this_thread::sleep_for(std::chrono::seconds(1));
			}
		}
			
		if (state_ == PNM_WAIT_QUIT)break;
		if (s_fName.empty())continue;
		n_pnm.filename = s_fName;
		if(ReadPNMFile(&n_pnm)==PNM_SUCCESS)
			cbfun(n_pnm);
	}

	state_ = PNM_WAIT_DELETE;
}

pnm_io::PNM_STATE pnm_io::PNM_IO::CreateTask(void (*cbfun)(PNM f), std::vector<std::string> * s_list)
{
	if (state_ == PNM_WAIT_DELETE) {
		delete(p_mainThread);
		p_mainThread = NULL;
		return PNM_SUCCESS;
	}
	if (state_ != PNM_UNINITIALIZED)return PNM_THREAD_RUNNING;

	//p_mainThread = new std::thread(&PNM_IO::ThreadMain, this, cbfun, s_list);

	p_mainThread = new std::thread([this, cbfun, s_list] {
		PNM n_pnm;
		for (auto s_fName : *s_list) {
			if (state_ == PNM_PAUSE) {
				while (true)
				{
					if (state_ != PNM_PAUSE)break;
					std::this_thread::sleep_for(std::chrono::seconds(1));
				}
			}

			if (state_ == PNM_WAIT_QUIT)break;
			if (s_fName.empty())continue;
			n_pnm.filename = s_fName;
			ReadPNMFile(&n_pnm);
			cbfun(n_pnm);
		}
		state_ = PNM_WAIT_DELETE;
	});

	//p_mainThread->join();
	state_ = PNM_THREAD_RUNNING;

	return PNM_SUCCESS;
}

pnm_io::PNM_STATE pnm_io::PNM_IO::StartTask() {
	if (state_ != PNM_THREAD_RUNNING)return PNM_UNINITIALIZED;
	state_ = PNM_THREAD_RUNNING;
	return PNM_SUCCESS;
}
pnm_io::PNM_STATE pnm_io::PNM_IO::PauseTask()
{
	if (state_ != PNM_THREAD_RUNNING)return PNM_UNINITIALIZED;
	state_ = PNM_PAUSE;
	return PNM_SUCCESS;
}

pnm_io::PNM_STATE pnm_io::PNM_IO::DeleteTask()
{
	// PNM_WAIT_QUIT
	if (p_mainThread->joinable() && state_ == PNM_WAIT_DELETE) {
		p_mainThread->join();
		delete(p_mainThread);
		p_mainThread = NULL;
		return PNM_SUCCESS;
	}
	// PNM_UNINITIALIZED
	else if (state_ != PNM_THREAD_RUNNING)return PNM_UNINITIALIZED;
	// PNM_THREAD_RUNNING
	else {
		state_ = PNM_WAIT_QUIT;
		if (p_mainThread->joinable())p_mainThread->join();
		state_ = PNM_UNINITIALIZED;
		delete(p_mainThread);
		p_mainThread = NULL;
	}
	
	return PNM_SUCCESS;
}

/*
	Will return PNM_RT_ERR when read error value
*/
pnm_io::PNM_STATE pnm_io::PNM_IO::ReadHeader(PNM *f, std::istream &is)
{
    is.seekg(0, std::ios::beg);
    is >> f->magicNumber >> f->width >> f->height >> f->maxValue;
	f->threshold = f->maxValue / 2;
	f->type = detail::MagNum2PNMTYPE(f->magicNumber);
    
#ifdef NDEBUG
    if (!(f->width && f->height && f->maxValue))
    {
        f->type = NO_TYPE;
        return PNM_RT_ERR;
    }
#endif // NDEBUG
    myassert(f->width != 0 || f->height != 0 || f->maxValue != 0, "Error reading value at pnm_io::PNM_IO::ReadHeader");
    return PNM_SUCCESS;
}

pnm_io::PNM_STATE pnm_io::PNM_IO::ReadPixelData(PNM *f, std::istream &is)
{
    if (!is)
        return PNM_RT_ERR;
    std::size_t n_start, n_size;
    n_start = is.tellg();
    is.seekg(0, is.end);
    n_size = (std::size_t)is.tellg() - n_start;
    is.seekg(n_start, is.beg);

    if (f->data.size() < n_size)
        f->data.resize(n_size);
    if (f->data.size() < n_size)
        return PNM_MEMERY_INSUFFICIENT;

    is.read(reinterpret_cast<char *>(f->data.data()), n_size);

    if (!is)
        return PNM_RT_ERR;
    return PNM_SUCCESS;
}

pnm_io::PNM_STATE pnm_io::PNM_IO::WriteHeader(PNM *f, std::ostream &os)
{
    myassert(f->width && f->height && f->maxValue, "Write data err");
    if (f->magicNumber.empty())
    {
        if (f->type == NO_TYPE)
            return PNM_RT_ERR;
        f->magicNumber = detail::PNMTYPE2MagNum(f->type);
    }
    os << f->magicNumber << "\n"
       << f->width << "\n"
       << f->height << "\n"
       << f->maxValue << "\n"; // Marks beginning of pixel data.
    return PNM_SUCCESS;
}

pnm_io::PNM_STATE pnm_io::PNM_IO::WritePixelData(PNM *f, std::ostream &os)
{
    myassert( f->data.size() > 0, "Write err");
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
pnm_io::PNM_STATE pnm_io::PNM_IO::Greyscale2RGB(
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
            rgb_pixel_data->data()[3 * (col + row * width)] = detail::clamp(0.0f, 1.0f, fr) * greyscale_pixel_data->data()[col + row * width];
            rgb_pixel_data->data()[3 * (col + row * width) + 1] = detail::clamp(0.0f, 1.0f, fb) * greyscale_pixel_data->data()[col + row * width];
            rgb_pixel_data->data()[3 * (col + row * width) + 2] = detail::clamp(0.0f, 1.0f, fg) * greyscale_pixel_data->data()[col + row * width];
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
pnm_io::PNM_STATE pnm_io::PNM_IO::RGB2Greyscale(
    std::size_t const width,
    std::size_t const height,
    std::vector<std::uint8_t> *const rgb_pixel_data,
    std::vector<std::uint8_t> *const greyscale_pixel_data)
{
    myassert(rgb_pixel_data != nullptr, "null RGB pixel data");
    myassert(rgb_pixel_data->size() >= (width * height) * 3, "RGB pixel have a wrong size");
    myassert(greyscale_pixel_data != nullptr, "null Greyscale pixel data");

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

pnm_io::PNM_STATE pnm_io::PNM_IO::BitMap2Greyscale(
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

pnm_io::PNM_STATE pnm_io::PNM_IO::Greyscale2BitMap(
    std::size_t const width,
    std::size_t const height,
    std::uint16_t threshold,
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

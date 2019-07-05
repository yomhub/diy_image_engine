#include "include/PixelEngine.h"

#ifdef NDEBUG
#define myassert(expression, mes) \
	if (!(expression))            \
	return peg::ENG_ERR
#else
#define myassert(expression, mes) assert(expression &&mes)
#endif // NDEBUG

#ifndef LOW_ACCURACY
const myfloat sinLookup[360] = {
0.0f,
0.0174524f, 0.0348995f, 0.052336f, 0.0697565f, 0.0871557f, 0.104528f, 0.121869f, 0.139173f, 0.156434f, 0.173648f,
0.190809f, 0.207912f, 0.224951f, 0.241922f, 0.258819f, 0.275637f, 0.292372f, 0.309017f, 0.325568f, 0.34202f,
0.358368f, 0.374607f, 0.390731f, 0.406737f, 0.422618f, 0.438371f, 0.45399f, 0.469472f, 0.48481f, 0.5f,
0.515038f, 0.529919f, 0.544639f, 0.559193f, 0.573576f, 0.587785f, 0.601815f, 0.615662f, 0.62932f, 0.642788f,
0.656059f, 0.669131f, 0.681998f, 0.694658f, 0.707107f, 0.71934f, 0.731354f, 0.743145f, 0.75471f, 0.766044f,
0.777146f, 0.788011f, 0.798635f, 0.809017f, 0.819152f, 0.829038f, 0.838671f, 0.848048f, 0.857167f, 0.866025f,
0.87462f, 0.882948f, 0.891007f, 0.898794f, 0.906308f, 0.913545f, 0.920505f, 0.927184f, 0.93358f, 0.939693f,
0.945519f, 0.951057f, 0.956305f, 0.961262f, 0.965926f, 0.970296f, 0.97437f, 0.978148f, 0.981627f, 0.984808f,
0.987688f, 0.990268f, 0.992546f, 0.994522f, 0.996195f, 0.997564f, 0.99863f, 0.999391f, 0.999848f, 1.0f,
0.999848f, 0.999391f, 0.99863f, 0.997564f, 0.996195f, 0.994522f, 0.992546f, 0.990268f, 0.987688f, 0.984808f,
0.981627f, 0.978148f, 0.97437f, 0.970296f, 0.965926f, 0.961262f, 0.956305f, 0.951057f, 0.945519f, 0.939693f,
0.93358f, 0.927184f, 0.920505f, 0.913545f, 0.906308f, 0.898794f, 0.891007f, 0.882948f, 0.87462f, 0.866025f,
0.857167f, 0.848048f, 0.838671f, 0.829038f, 0.819152f, 0.809017f, 0.798635f, 0.788011f, 0.777146f, 0.766044f,
0.75471f, 0.743145f, 0.731354f, 0.71934f, 0.707107f, 0.694658f, 0.681998f, 0.669131f, 0.656059f, 0.642788f,
0.62932f, 0.615662f, 0.601815f, 0.587785f, 0.573576f, 0.559193f, 0.544639f, 0.529919f, 0.515038f, 0.5f,
0.48481f, 0.469472f, 0.45399f, 0.438371f, 0.422618f, 0.406737f, 0.390731f, 0.374607f, 0.358368f, 0.34202f,
0.325568f, 0.309017f, 0.292372f, 0.275637f, 0.258819f, 0.241922f, 0.224951f, 0.207912f, 0.190809f, 0.173648f,
0.156434f, 0.139173f, 0.121869f, 0.104528f, 0.0871557f, 0.0697565f, 0.052336f, 0.0348995f, 0.0174524f, 3.58979e-09f,
-0.0174524f, -0.0348995f, -0.052336f, -0.0697565f, -0.0871557f, -0.104528f, -0.121869f, -0.139173f, -0.156434f, -0.173648f,
-0.190809f, -0.207912f, -0.224951f, -0.241922f, -0.258819f, -0.275637f, -0.292372f, -0.309017f, -0.325568f, -0.34202f,
-0.358368f, -0.374607f, -0.390731f, -0.406737f, -0.422618f, -0.438371f, -0.45399f, -0.469472f, -0.48481f, -0.5f,
-0.515038f, -0.529919f, -0.544639f, -0.559193f, -0.573576f, -0.587785f, -0.601815f, -0.615661f, -0.62932f, -0.642788f,
-0.656059f, -0.669131f, -0.681998f, -0.694658f, -0.707107f, -0.71934f, -0.731354f, -0.743145f, -0.75471f, -0.766044f,
-0.777146f, -0.788011f, -0.798635f, -0.809017f, -0.819152f, -0.829038f, -0.838671f, -0.848048f, -0.857167f, -0.866025f,
-0.87462f, -0.882948f, -0.891007f, -0.898794f, -0.906308f, -0.913545f, -0.920505f, -0.927184f, -0.93358f, -0.939693f,
-0.945519f, -0.951057f, -0.956305f, -0.961262f, -0.965926f, -0.970296f, -0.97437f, -0.978148f, -0.981627f, -0.984808f,
-0.987688f, -0.990268f, -0.992546f, -0.994522f, -0.996195f, -0.997564f, -0.99863f, -0.999391f, -0.999848f, -1.0f,
-0.999848f, -0.999391f, -0.99863f, -0.997564f, -0.996195f, -0.994522f, -0.992546f, -0.990268f, -0.987688f, -0.984808f,
-0.981627f, -0.978148f, -0.97437f, -0.970296f, -0.965926f, -0.961262f, -0.956305f, -0.951057f, -0.945519f, -0.939693f,
-0.93358f, -0.927184f, -0.920505f, -0.913545f, -0.906308f, -0.898794f, -0.891007f, -0.882948f, -0.87462f, -0.866025f,
-0.857167f, -0.848048f, -0.838671f, -0.829038f, -0.819152f, -0.809017f, -0.798636f, -0.788011f, -0.777146f, -0.766044f,
-0.75471f, -0.743145f, -0.731354f, -0.71934f, -0.707107f, -0.694658f, -0.681998f, -0.669131f, -0.656059f, -0.642788f,
-0.62932f, -0.615662f, -0.601815f, -0.587785f, -0.573576f, -0.559193f, -0.544639f, -0.529919f, -0.515038f, -0.5f,
-0.48481f, -0.469472f, -0.453991f, -0.438371f, -0.422618f, -0.406737f, -0.390731f, -0.374607f, -0.358368f, -0.34202f,
-0.325568f, -0.309017f, -0.292372f, -0.275637f, -0.258819f, -0.241922f, -0.224951f, -0.207912f, -0.190809f, -0.173648f,
-0.156434f, -0.139173f, -0.121869f, -0.104528f, -0.0871558f, -0.0697565f, -0.052336f, -0.0348995f, -0.0174524f,
};
const myfloat cosLookup[360] = {
1.0f,
0.999848f, 0.999391f, 0.99863f, 0.997564f, 0.996195f, 0.994522f, 0.992546f, 0.990268f, 0.987688f, 0.984808f,
0.981627f, 0.978148f, 0.97437f, 0.970296f, 0.965926f, 0.961262f, 0.956305f, 0.951057f, 0.945519f, 0.939693f,
0.93358f, 0.927184f, 0.920505f, 0.913545f, 0.906308f, 0.898794f, 0.891007f, 0.882948f, 0.87462f, 0.866025f,
0.857167f, 0.848048f, 0.838671f, 0.829038f, 0.819152f, 0.809017f, 0.798635f, 0.788011f, 0.777146f, 0.766044f,
0.75471f, 0.743145f, 0.731354f, 0.71934f, 0.707107f, 0.694658f, 0.681998f, 0.669131f, 0.656059f, 0.642788f,
0.62932f, 0.615662f, 0.601815f, 0.587785f, 0.573576f, 0.559193f, 0.544639f, 0.529919f, 0.515038f, 0.5f,
0.48481f, 0.469472f, 0.45399f, 0.438371f, 0.422618f, 0.406737f, 0.390731f, 0.374607f, 0.358368f, 0.34202f,
0.325568f, 0.309017f, 0.292372f, 0.275637f, 0.258819f, 0.241922f, 0.224951f, 0.207912f, 0.190809f, 0.173648f,
0.156434f, 0.139173f, 0.121869f, 0.104528f, 0.0871557f, 0.0697565f, 0.052336f, 0.0348995f, 0.0174524f, 1.7949e-09f,
-0.0174524f, -0.0348995f, -0.052336f, -0.0697565f, -0.0871557f, -0.104528f, -0.121869f, -0.139173f, -0.156434f, -0.173648f,
-0.190809f, -0.207912f, -0.224951f, -0.241922f, -0.258819f, -0.275637f, -0.292372f, -0.309017f, -0.325568f, -0.34202f,
-0.358368f, -0.374607f, -0.390731f, -0.406737f, -0.422618f, -0.438371f, -0.45399f, -0.469472f, -0.48481f, -0.5f,
-0.515038f, -0.529919f, -0.544639f, -0.559193f, -0.573576f, -0.587785f, -0.601815f, -0.615662f, -0.62932f, -0.642788f,
-0.656059f, -0.669131f, -0.681998f, -0.694658f, -0.707107f, -0.71934f, -0.731354f, -0.743145f, -0.75471f, -0.766044f,
-0.777146f, -0.788011f, -0.798635f, -0.809017f, -0.819152f, -0.829038f, -0.838671f, -0.848048f, -0.857167f, -0.866025f,
-0.87462f, -0.882948f, -0.891007f, -0.898794f, -0.906308f, -0.913545f, -0.920505f, -0.927184f, -0.93358f, -0.939693f,
-0.945519f, -0.951057f, -0.956305f, -0.961262f, -0.965926f, -0.970296f, -0.97437f, -0.978148f, -0.981627f, -0.984808f,
-0.987688f, -0.990268f, -0.992546f, -0.994522f, -0.996195f, -0.997564f, -0.99863f, -0.999391f, -0.999848f, -1.0f,
-0.999848f, -0.999391f, -0.99863f, -0.997564f, -0.996195f, -0.994522f, -0.992546f, -0.990268f, -0.987688f, -0.984808f,
-0.981627f, -0.978148f, -0.97437f, -0.970296f, -0.965926f, -0.961262f, -0.956305f, -0.951057f, -0.945519f, -0.939693f,
-0.93358f, -0.927184f, -0.920505f, -0.913545f, -0.906308f, -0.898794f, -0.891007f, -0.882948f, -0.87462f, -0.866025f,
-0.857167f, -0.848048f, -0.838671f, -0.829038f, -0.819152f, -0.809017f, -0.798636f, -0.788011f, -0.777146f, -0.766044f,
-0.75471f, -0.743145f, -0.731354f, -0.71934f, -0.707107f, -0.694658f, -0.681998f, -0.669131f, -0.656059f, -0.642788f,
-0.62932f, -0.615662f, -0.601815f, -0.587785f, -0.573576f, -0.559193f, -0.544639f, -0.529919f, -0.515038f, -0.5f,
-0.48481f, -0.469472f, -0.45399f, -0.438371f, -0.422618f, -0.406737f, -0.390731f, -0.374607f, -0.358368f, -0.34202f,
-0.325568f, -0.309017f, -0.292372f, -0.275637f, -0.258819f, -0.241922f, -0.224951f, -0.207912f, -0.190809f, -0.173648f,
-0.156434f, -0.139173f, -0.121869f, -0.104528f, -0.0871557f, -0.0697565f, -0.052336f, -0.0348995f, -0.0174524f, -5.38469e-09f,
0.0174524f, 0.0348995f, 0.052336f, 0.0697565f, 0.0871557f, 0.104528f, 0.121869f, 0.139173f, 0.156434f, 0.173648f,
0.190809f, 0.207912f, 0.224951f, 0.241922f, 0.258819f, 0.275637f, 0.292372f, 0.309017f, 0.325568f, 0.34202f,
0.358368f, 0.374607f, 0.390731f, 0.406737f, 0.422618f, 0.438371f, 0.45399f, 0.469472f, 0.48481f, 0.5f,
0.515038f, 0.529919f, 0.544639f, 0.559193f, 0.573576f, 0.587785f, 0.601815f, 0.615661f, 0.62932f, 0.642788f,
0.656059f, 0.669131f, 0.681998f, 0.694658f, 0.707107f, 0.71934f, 0.731354f, 0.743145f, 0.75471f, 0.766044f,
0.777146f, 0.788011f, 0.798635f, 0.809017f, 0.819152f, 0.829038f, 0.838671f, 0.848048f, 0.857167f, 0.866025f,
0.87462f, 0.882948f, 0.891007f, 0.898794f, 0.906308f, 0.913545f, 0.920505f, 0.927184f, 0.93358f, 0.939693f,
0.945519f, 0.951057f, 0.956305f, 0.961262f, 0.965926f, 0.970296f, 0.97437f, 0.978148f, 0.981627f, 0.984808f,
0.987688f, 0.990268f, 0.992546f, 0.994522f, 0.996195f, 0.997564f, 0.99863f, 0.999391f, 0.999848f
};
const myfloat tanLookup[360] = {
0.0f,
0.0174551f, 0.0349208f, 0.0524078f, 0.0699268f, 0.0874887f, 0.105104f, 0.122785f, 0.140541f, 0.158384f, 0.176327f,
0.19438f, 0.212557f, 0.230868f, 0.249328f, 0.267949f, 0.286745f, 0.305731f, 0.32492f, 0.344328f, 0.36397f,
0.383864f, 0.404026f, 0.424475f, 0.445229f, 0.466308f, 0.487733f, 0.509525f, 0.531709f, 0.554309f, 0.57735f,
0.600861f, 0.624869f, 0.649408f, 0.674509f, 0.700208f, 0.726543f, 0.753554f, 0.781286f, 0.809784f, 0.8391f,
0.869287f, 0.900404f, 0.932515f, 0.965689f, 1.0f, 1.03553f, 1.07237f, 1.11061f, 1.15037f, 1.19175f,
1.2349f, 1.27994f, 1.32704f, 1.37638f, 1.42815f, 1.48256f, 1.53987f, 1.60033f, 1.66428f, 1.73205f,
1.80405f, 1.88073f, 1.96261f, 2.0503f, 2.14451f, 2.24604f, 2.35585f, 2.47509f, 2.60509f, 2.74748f,
2.90421f, 3.07768f, 3.27085f, 3.48741f, 3.73205f, 4.01078f, 4.33148f, 4.70463f, 5.14455f, 5.67128f,
6.31375f, 7.11537f, 8.14435f, 9.51436f, 11.4301f, 14.3007f, 19.0811f, 28.6363f, 57.29f, 5.57135e+08f,
-57.29f, -28.6363f, -19.0811f, -14.3007f, -11.4301f, -9.51436f, -8.14435f, -7.11537f, -6.31375f, -5.67128f,
-5.14455f, -4.70463f, -4.33148f, -4.01078f, -3.73205f, -3.48741f, -3.27085f, -3.07768f, -2.90421f, -2.74748f,
-2.60509f, -2.47509f, -2.35585f, -2.24604f, -2.14451f, -2.0503f, -1.96261f, -1.88073f, -1.80405f, -1.73205f,
-1.66428f, -1.60033f, -1.53987f, -1.48256f, -1.42815f, -1.37638f, -1.32704f, -1.27994f, -1.2349f, -1.19175f,
-1.15037f, -1.11061f, -1.07237f, -1.03553f, -1.0f, -0.965689f, -0.932515f, -0.900404f, -0.869287f, -0.8391f,
-0.809784f, -0.781286f, -0.753554f, -0.726543f, -0.700208f, -0.674509f, -0.649408f, -0.624869f, -0.600861f, -0.57735f,
-0.554309f, -0.531709f, -0.509525f, -0.487733f, -0.466308f, -0.445229f, -0.424475f, -0.404026f, -0.383864f, -0.36397f,
-0.344328f, -0.32492f, -0.305731f, -0.286745f, -0.267949f, -0.249328f, -0.230868f, -0.212557f, -0.19438f, -0.176327f,
-0.158384f, -0.140541f, -0.122785f, -0.105104f, -0.0874887f, -0.0699268f, -0.0524078f, -0.0349208f, -0.0174551f, -3.58979e-09f,
0.0174551f, 0.0349208f, 0.0524078f, 0.0699268f, 0.0874887f, 0.105104f, 0.122785f, 0.140541f, 0.158384f, 0.176327f,
0.19438f, 0.212557f, 0.230868f, 0.249328f, 0.267949f, 0.286745f, 0.305731f, 0.32492f, 0.344328f, 0.36397f,
0.383864f, 0.404026f, 0.424475f, 0.445229f, 0.466308f, 0.487733f, 0.509525f, 0.531709f, 0.554309f, 0.57735f,
0.600861f, 0.624869f, 0.649408f, 0.674509f, 0.700208f, 0.726543f, 0.753554f, 0.781286f, 0.809784f, 0.8391f,
0.869287f, 0.900404f, 0.932515f, 0.965689f, 1.0f, 1.03553f, 1.07237f, 1.11061f, 1.15037f, 1.19175f,
1.2349f, 1.27994f, 1.32704f, 1.37638f, 1.42815f, 1.48256f, 1.53986f, 1.60033f, 1.66428f, 1.73205f,
1.80405f, 1.88073f, 1.96261f, 2.0503f, 2.14451f, 2.24604f, 2.35585f, 2.47509f, 2.60509f, 2.74748f,
2.90421f, 3.07768f, 3.27085f, 3.48741f, 3.73205f, 4.01078f, 4.33148f, 4.70463f, 5.14455f, 5.67128f,
6.31375f, 7.11537f, 8.14435f, 9.51436f, 11.4301f, 14.3007f, 19.0811f, 28.6362f, 57.2899f, 1.85712e+08f,
-57.29f, -28.6363f, -19.0811f, -14.3007f, -11.4301f, -9.51437f, -8.14435f, -7.11537f, -6.31375f, -5.67128f,
-5.14455f, -4.70463f, -4.33148f, -4.01078f, -3.73205f, -3.48741f, -3.27085f, -3.07768f, -2.90421f, -2.74748f,
-2.60509f, -2.47509f, -2.35585f, -2.24604f, -2.14451f, -2.0503f, -1.96261f, -1.88073f, -1.80405f, -1.73205f,
-1.66428f, -1.60033f, -1.53987f, -1.48256f, -1.42815f, -1.37638f, -1.32704f, -1.27994f, -1.2349f, -1.19175f,
-1.15037f, -1.11061f, -1.07237f, -1.03553f, -1.0f, -0.965689f, -0.932515f, -0.900404f, -0.869287f, -0.8391f,
-0.809784f, -0.781286f, -0.753554f, -0.726543f, -0.700208f, -0.674509f, -0.649408f, -0.624869f, -0.600861f, -0.57735f,
-0.554309f, -0.531709f, -0.509525f, -0.487733f, -0.466308f, -0.445229f, -0.424475f, -0.404026f, -0.383864f, -0.36397f,
-0.344328f, -0.32492f, -0.305731f, -0.286745f, -0.267949f, -0.249328f, -0.230868f, -0.212557f, -0.19438f, -0.176327f,
-0.158384f, -0.140541f, -0.122785f, -0.105104f, -0.0874887f, -0.0699268f, -0.0524078f, -0.0349208f, -0.0174551f,
};
#define mysin(x) sinLookup[x]
#define mycos(x) cosLookup[x]
#define mytan(x) tanLookup[x]
#else
#define mycos(x) cos(x * 3.14 / 180)
#define mysin(x) sin(x * 3.14 / 180)
#define mytan(x) tan(x * 3.14 / 180)
#endif // LOW_ACCURACY

namespace peg
{

namespace detail
{

template <typename T>
T clamp(const T min_value, const T max_value, const T value)
{
	return value < min_value ? min_value
							 : (value > max_value ? max_value : value);
}

} // namespace detail

PixelEngine::PixelEngine(EngineState init)
{
#ifdef CUDA_CODE_COMPILE
	if(init==ENG_CUDA_READY){
		device_ = -1;
		cudaGetDevice(&device_);
		cudaGetDeviceProperties(&prop_, device_);

		if (!(prop_.major > 3 || (prop_.major == 3 && prop_.minor >= 5)))
		{
			std::cout << "GPU"<< device_ << "-" << prop_.name << "does not support CUDA Dynamic Parallelism." << std::endl;
			device_ = -1;
			custate_ = ENG_CUDA_NOT_SUPPORT;
		}
		else if(init==ENG_CUDA_READY){
			std::cout << "Use GPU device:" << device_ << ": " << prop_.name << std::endl;
			std::cout << "SM Count:" << prop_.multiProcessorCount << std::endl;
			std::cout << "Shared Memery Per Block:" << (prop_.sharedMemPerBlock / 1024) << " KB "<< std::endl;
			std::cout << "Max Threads Per Block:" << prop_.maxThreadsPerBlock << std::endl;
			std::cout << "Max Threads Per Multi Processor:" << prop_.maxThreadsPerMultiProcessor << std::endl;
			std::cout << "Max Threads Bunch:" << prop_.maxThreadsPerMultiProcessor / 32 << std::endl;
			custate_ = ENG_CUDA_READY;
		}
	}

#endif // CUDA_CODE_COMPILE

	f_state = ENG_READY;
}

PixelEngine::~PixelEngine()
{
	v_Pixels.clear();
}

/*

Will return A if B:

	ENG_READY		success
	ENG_RUNNING		aouther operation is running

Mask is:
				 [m  m  m]
		faceor * [m  m  m]
				 [m  m  m]

Return peg::ENG_ERR if:
	pixels data size < height * width * sizePerPixel
	pixels sizePerPixel or height or width is 0
	Mask matrix buffer < n * n
	Mask order <=1
	Mask order > height or width
*/
EngineState PixelEngine::smooth(Pixels & src, const Matrix & mask, float factor)
{
	myassert(src.sizePerPixel && src.sizePerPixel<=3 && src.height && src.width, "Src pixels member sizePerPixel or height or width is 0. peg::PixelEngine::smooth");
	myassert(src.data.size() >= (src.height * src.width * src.sizePerPixel), "Bad pixels size. peg::PixelEngine::smooth");
	myassert(mask.x >= 1 && mask.y >= 1, "Bad matrix size. peg::PixelEngine::smooth");
	myassert((mask.y <= src.height && mask.x <= src.width), "Bad matrix size. peg::PixelEngine::smooth");
	myassert(mask.data.size() >= mask.x * mask.y, "Bad matrix buffer size. peg::PixelEngine::smooth");

	if (f_state != ENG_READY)
		return ENG_BUSSY;

	f_state = ENG_RUNNING;


#ifdef CUDA_CODE_COMPILE
	if (device_ != -1 && custate_ == ENG_CUDA_READY) {
		std::uint8_t *buff,*out;
		mydouble* m;

		cusmooth(src.data.data(), src.width, src.height, src.sizePerPixel, mask.x , mask.y, mask.data.data(), factor,device_,&prop_);

		f_state = ENG_READY;
		return ENG_READY;
	}
	
#endif // CUDA_CODE_COMPILE

	std::vector<std::uint8_t> buff = src.data;
	std::uint16_t pixelBuff[3] = {};

	// Only handle the central area
	for (std::size_t row = (mask.y / 2 + mask.y % 2); row < (src.height - mask.y / 2); row += src.sizePerPixel)
	{
		for (std::size_t col = (mask.x / 2 + mask.x % 2); col < (src.width - mask.x / 2); col += src.sizePerPixel)
		{
			pixelBuff[0] = pixelBuff[1] = pixelBuff[2] = 0;
			switch (mask.x * mask.y)
			{
			// 3 * 3 matrix optimization
			case 9:
				pixelBuff[0] += buff[col - 1 + (row - 1) * src.width + 0] * mask.data[0] +
								buff[col + (row - 1) * src.width + 0] * mask.data[1] +
								buff[col + 1 + (row - 1) * src.width + 0] * mask.data[2] +
								buff[col - 1 + (row)*src.width + 0] * mask.data[3] +
								buff[col + (row)*src.width + 0] * mask.data[4] +
								buff[col + 1 + (row)*src.width + 0] * mask.data[5] +
								buff[col - 1 + (row + 1) * src.width + 0] * mask.data[6] +
								buff[col + (row + 1) * src.width + 0] * mask.data[7] +
								buff[col + 1 + (row + 1) * src.width + 0] * mask.data[8];	
				if (src.sizePerPixel > 1)
					pixelBuff[1] += buff[col - 1 + (row - 1) * src.width + 1] * mask.data[0] +
									buff[col + (row - 1) * src.width + 1] * mask.data[1] +
									buff[col + 1 + (row - 1) * src.width + 1] * mask.data[2] +
									buff[col - 1 + (row)*src.width + 1] * mask.data[3] +
									buff[col + (row)*src.width + 1] * mask.data[4] +
									buff[col + 1 + (row)*src.width + 1] * mask.data[5] +
									buff[col - 1 + (row + 1) * src.width + 1] * mask.data[6] +
									buff[col + (row + 1) * src.width + 1] * mask.data[7] +
									buff[col + 1 + (row + 1) * src.width + 1] * mask.data[8];
				if (src.sizePerPixel > 2)
					pixelBuff[2] += buff[col - 1 + (row - 1) * src.width + 2] * mask.data[0] +
									buff[col + (row - 1) * src.width + 2] * mask.data[1] +
									buff[col + 1 + (row - 1) * src.width + 2] * mask.data[2] +
									buff[col - 1 + (row)*src.width + 2] * mask.data[3] +
									buff[col + (row)*src.width + 2] * mask.data[4] +
									buff[col + 1 + (row)*src.width + 2] * mask.data[5] +
									buff[col - 1 + (row + 1) * src.width + 2] * mask.data[6] +
									buff[col + (row + 1) * src.width + 2] * mask.data[7] +
									buff[col + 1 + (row + 1) * src.width + 2] * mask.data[8];
				break;
			default:
				for (auto k = std::size_t{ 0 }; k < mask.x * mask.y; ++k)
				{
					pixelBuff[0] += buff[(col - mask.x / 2 + k % mask.x) + (row - mask.y / 2 + k / mask.y) * src.width + 0] * mask.data[k];
					if (src.sizePerPixel > 1)
						pixelBuff[1] += buff[(col - mask.x / 2 + k % mask.x) + (row - mask.y / 2 + k / mask.y) * src.width + 1] * mask.data[k];
					if (src.sizePerPixel > 2)
						pixelBuff[2] += buff[(col - mask.x / 2 + k % mask.x) + (row - mask.y / 2 + k / mask.y) * src.width + 2] * mask.data[k];
				}
				break;
			}
			
			src.data[col + row * src.width + 0] = pixelBuff[0] * factor;
			if (src.sizePerPixel > 1)
				src.data[col + row * src.width + 1] = pixelBuff[1] * factor;
			if (src.sizePerPixel > 2)
				src.data[col + row * src.width + 2] = pixelBuff[2] * factor;
		}
	}

	f_state = ENG_READY;
	return ENG_READY;
}
/*

Will return A if B:

	ENG_READY		success
	ENG_RUNNING		aouther operation is running

Mask is:
				  [m  m  m]
		faceorX * [m  m  m]
				  [m  m  m]

An std::runtime_error is thrown if:
	pixels data size < height * width * sizePerPixel
	pixels sizePerPixel or height or width is 0
	Mask matrix* buffer < n * n
	Mask order <=1
	Mask order > height or width
*/
EngineState PixelEngine::smooth2D(Pixels & src, const Matrix & mask1, const Matrix & mask2, float factor1, float factor2)
{
	myassert(src.sizePerPixel && src.sizePerPixel <= 3 && src.height && src.width, "Src pixels member sizePerPixel or height or width is 0. peg::PixelEngine::smooth2D");
	myassert(src.data.size() >= (src.height * src.width * src.sizePerPixel), "Bad pixels size. peg::PixelEngine::smooth2D");
	myassert(mask1.x >= 1 && mask1.y >= 1, "Bad matrix size. peg::PixelEngine::smooth2D");
	myassert((mask1.y <= src.height && mask1.x <= src.width), "Bad matrix size. peg::PixelEngine::smooth2D");
	myassert(mask1.data.size() >= mask1.x * mask1.y, "Bad matrix buffer size. peg::PixelEngine::smooth2D");

	if (f_state != ENG_READY)
		return ENG_BUSSY;

	f_state = ENG_RUNNING;

	std::vector<std::uint8_t> buff = src.data;
	std::uint16_t pixelBuff1[3] = {};
	std::uint16_t pixelBuff2[3] = {};

	for (std::size_t row = (mask1.y / 2 + mask1.y % 2); row < (src.height - mask1.y / 2); row += src.sizePerPixel)
	{
		for (std::size_t col = (mask1.x / 2 + mask1.x % 2); col < (src.width - mask1.x / 2); col += src.sizePerPixel)
		{
			pixelBuff1[0] = pixelBuff1[1] = pixelBuff1[2] = 0;
			pixelBuff2[0] = pixelBuff2[1] = pixelBuff2[2] = 0;
			switch (mask1.x * mask1.y)
			{
			case 9:
				pixelBuff1[0] += buff[col - 1 + (row - 1) * src.width + 0] * mask1.data[0] +
								 buff[col + (row - 1) * src.width + 0] * mask1.data[1] +
								 buff[col + 1 + (row - 1) * src.width + 0] * mask1.data[2] +
								 buff[col - 1 + (row)*src.width + 0] * mask1.data[3] +
								 buff[col + (row)*src.width + 0] * mask1.data[4] +
								 buff[col + 1 + (row)*src.width + 0] * mask1.data[5] +
								 buff[col - 1 + (row + 1) * src.width + 0] * mask1.data[6] +
								 buff[col + (row + 1) * src.width + 0] * mask1.data[7] +
								 buff[col + 1 + (row + 1) * src.width + 0] * mask1.data[8];

				pixelBuff2[0] += buff[col - 1 + (row - 1) * src.width + 0] * mask2.data[0] +
								 buff[col + (row - 1) * src.width + 0] * mask2.data[1] +
								 buff[col + 1 + (row - 1) * src.width + 0] * mask2.data[2] +
								 buff[col - 1 + (row)*src.width + 0] * mask2.data[3] +
								 buff[col + (row)*src.width + 0] * mask2.data[4] +
								 buff[col + 1 + (row)*src.width + 0] * mask2.data[5] +
								 buff[col - 1 + (row + 1) * src.width + 0] * mask2.data[6] +
								 buff[col + (row + 1) * src.width + 0] * mask2.data[7] +
								 buff[col + 1 + (row + 1) * src.width + 0] * mask2.data[8];
				if (src.sizePerPixel > 1)
				{
					pixelBuff1[1] += buff[col - 1 + (row - 1) * src.width + 1] * mask1.data[0] +
									buff[col + (row - 1) * src.width + 1] * mask1.data[1] +
									buff[col + 1 + (row - 1) * src.width + 1] * mask1.data[2] +
									buff[col - 1 + (row)*src.width + 1] * mask1.data[3] +
									buff[col + (row)*src.width + 1] * mask1.data[4] +
									buff[col + 1 + (row)*src.width + 1] * mask1.data[5] +
									buff[col - 1 + (row + 1) * src.width + 1] * mask1.data[6] +
									buff[col + (row + 1) * src.width + 1] * mask1.data[7] +
									buff[col + 1 + (row + 1) * src.width + 1] * mask1.data[8];

					pixelBuff2[1] += buff[col - 1 + (row - 1) * src.width + 1] * mask2.data[0] +
									buff[col + (row - 1) * src.width + 1] * mask2.data[1] +
									buff[col + 1 + (row - 1) * src.width + 1] * mask2.data[2] +
									buff[col - 1 + (row)*src.width + 1] * mask2.data[3] +
									buff[col + (row)*src.width + 1] * mask2.data[4] +
									buff[col + 1 + (row)*src.width + 1] * mask2.data[5] +
									buff[col - 1 + (row + 1) * src.width + 1] * mask2.data[6] +
									buff[col + (row + 1) * src.width + 1] * mask2.data[7] +
									buff[col + 1 + (row + 1) * src.width + 1] * mask2.data[8];
				}
				if (src.sizePerPixel > 2)
				{
					pixelBuff1[2] += buff[col - 1 + (row - 1) * src.width + 2] * mask1.data[0] +
									buff[col + (row - 1) * src.width + 2] * mask1.data[1] +
									buff[col + 1 + (row - 1) * src.width + 2] * mask1.data[2] +
									buff[col - 1 + (row)*src.width + 2] * mask1.data[3] +
									buff[col + (row)*src.width + 2] * mask1.data[4] +
									buff[col + 1 + (row)*src.width + 2] * mask1.data[5] +
									buff[col - 1 + (row + 1) * src.width + 2] * mask1.data[6] +
									buff[col + (row + 1) * src.width + 2] * mask1.data[7] +
									buff[col + 1 + (row + 1) * src.width + 2] * mask1.data[8];
					pixelBuff2[2] += buff[col - 1 + (row - 1) * src.width + 2] * mask2.data[0] +
									buff[col + (row - 1) * src.width + 2] * mask2.data[1] +
									buff[col + 1 + (row - 1) * src.width + 2] * mask2.data[2] +
									buff[col - 1 + (row)*src.width + 2] * mask2.data[3] +
									buff[col + (row)*src.width + 2] * mask2.data[4] +
									buff[col + 1 + (row)*src.width + 2] * mask2.data[5] +
									buff[col - 1 + (row + 1) * src.width + 2] * mask2.data[6] +
									buff[col + (row + 1) * src.width + 2] * mask2.data[7] +
									buff[col + 1 + (row + 1) * src.width + 2] * mask2.data[8];
				}
				break;
			default:
				for (auto k = std::size_t{ 0 }; k < mask1.x * mask1.y; ++k)
				{
					pixelBuff1[0] += buff[(col - mask1.x / 2 + k % mask1.x) + (row - mask1.y / 2 + k / mask1.y) * src.width + 0] * mask1.data[k];
					pixelBuff2[0] += buff[(col - mask2.x / 2 + k % mask2.x) + (row - mask2.y / 2 + k / mask2.y) * src.width + 0] * mask2.data[k];
					if (src.sizePerPixel > 1) {
						pixelBuff1[1] += buff[(col - mask1.x / 2 + k % mask1.x) + (row - mask1.y / 2 + k / mask1.y) * src.width + 1] * mask1.data[k];
						pixelBuff2[1] += buff[(col - mask2.x / 2 + k % mask2.x) + (row - mask2.y / 2 + k / mask2.y) * src.width + 1] * mask2.data[k];
					}
					if (src.sizePerPixel > 2) {
						pixelBuff1[2] += buff[(col - mask1.x / 2 + k % mask1.x) + (row - mask1.y / 2 + k / mask1.y) * src.width + 2] * mask1.data[k];
						pixelBuff2[2] += buff[(col - mask2.x / 2 + k % mask2.x) + (row - mask2.y / 2 + k / mask2.y) * src.width + 2] * mask2.data[k];
					}
				}
				break;
			}
			
			
			pixelBuff1[0] *= factor1; pixelBuff1[1] *= factor1; pixelBuff1[2] *= factor1;
			pixelBuff2[0] *= factor2; pixelBuff2[1] *= factor2; pixelBuff2[2] *= factor2;
			src.data[col + row * src.width + 0] = sqrtf(pixelBuff1[0] * pixelBuff1[0] + pixelBuff2[0] * pixelBuff2[0]);
			if (src.sizePerPixel > 1)
				src.data[col + row * src.width + 1] = sqrtf(pixelBuff1[1] * pixelBuff1[1] + pixelBuff2[1] * pixelBuff2[1]);
			if (src.sizePerPixel > 2)
				src.data[col + row * src.width + 2] = sqrtf(pixelBuff1[2] * pixelBuff1[2] + pixelBuff2[2] * pixelBuff2[2]);
		}
	}

	f_state = ENG_READY;
	return ENG_READY;
}

/*
Will resize pixels

Attention:it will change the size of Pixels->data

Mode: 0: linear

An std::runtime_error is thrown if:
	pixels data size < height * width * sizePerPixel
	pixels sizePerPixel or height or width is 0
*/
EngineState PixelEngine::resize(Pixels & src, std::uint16_t newWidth, std::uint16_t newHeight, std::uint8_t mode)
{
	myassert(src.sizePerPixel && src.sizePerPixel <= 3 && src.height && src.width, "Src pixels member sizePerPixel or height or width is 0. peg::PixelEngine::resize");
	myassert(newWidth && newHeight, "New Width and Height should't be 0. peg::PixelEngine::resize");
	myassert(src.data.size() >= (src.height * src.width * src.sizePerPixel), "Bad pixels size. peg::PixelEngine::resize");

	if (newWidth == src.width && newHeight == src.height)
		return ENG_READY;
	if (f_state != ENG_READY)
		return ENG_BUSSY;

	f_state = ENG_RUNNING;

	bool b_x2Big = newWidth > src.width ? true : false;
	bool b_y2Big = newHeight > src.height ? true : false;

	std::uint16_t x_big = newWidth > src.width ? newWidth : src.width;
	std::uint16_t x_small = newWidth < src.width ? newWidth : src.width;

	std::uint16_t y_big = newHeight > src.height ? newHeight : src.height;
	std::uint16_t y_small = newHeight < src.height ? newHeight : src.height;

	std::uint16_t x_step = x_big / x_small + (x_big % x_small == 0 ? 0 : 1);
	std::uint16_t y_step = y_big / y_small + (y_big % y_small == 0 ? 0 : 1);

	std::vector<std::uint8_t> buff(newHeight * newWidth * src.sizePerPixel);

	for (std::size_t row = 0; row < newHeight; row += b_y2Big ? y_step + src.sizePerPixel : 1 + src.sizePerPixel)
	{
		for (std::size_t col = 0; col < newWidth - 1; col += src.sizePerPixel)
		{
			if (b_x2Big)
			{
				// amplification
				for (std::size_t x = 0; x < x_step; ++x)
				{
					buff[row * newWidth + col + 0] = src.data[col * x_step + (b_y2Big ? row / y_step : row * y_step) * src.width + 0] +
													(src.data[col * x_step + (b_y2Big ? row / y_step : row * y_step) * src.width + 0] -
													src.data[(col + 1) * x_step + (b_y2Big ? row / y_step : row * y_step) * src.width + 0]) *
													x / x_step;
					if(src.sizePerPixel > 1)
						buff[row * newWidth + col + 0] = src.data[col * x_step + (b_y2Big ? row / y_step : row * y_step) * src.width + 1] +
														(src.data[col * x_step + (b_y2Big ? row / y_step : row * y_step) * src.width + 1] -
														src.data[(col + 1) * x_step + (b_y2Big ? row / y_step : row * y_step) * src.width + 1]) *
														x / x_step;
					if (src.sizePerPixel > 2)
						buff[row * newWidth + col + 0] = src.data[col * x_step + (b_y2Big ? row / y_step : row * y_step) * src.width + 2] +
														(src.data[col * x_step + (b_y2Big ? row / y_step : row * y_step) * src.width + 2] -
														src.data[(col + 1) * x_step + (b_y2Big ? row / y_step : row * y_step) * src.width + 2]) *
														x / x_step;
				}
			}
			else
			{
				//narrow
				buff[row * newWidth + col + 0] = src.data[col * x_step + (b_y2Big ? row / y_step : row * y_step) * src.width + 0];
				if (src.sizePerPixel > 1)buff[row * newWidth + col + 1] = src.data[col * x_step + (b_y2Big ? row / y_step : row * y_step) * src.width + 1];
				if (src.sizePerPixel > 2)buff[row * newWidth + col + 2] = src.data[col * x_step + (b_y2Big ? row / y_step : row * y_step) * src.width + 2];
			}

			if (b_y2Big && row > 0)
			{
				// amplification
				for (std::size_t y = 1; y < y_step; ++y)
				{
					buff[(row + y - y_step) * newWidth + col + 0] =
											buff[(row - y_step) * newWidth + col + 0] +
											(buff[(row - y_step) * newWidth + col + 0] -
											buff[(row)* newWidth + col + 0]) *
											y / y_step;
					if (src.sizePerPixel > 1)buff[(row + y - y_step) * newWidth + col + 1] =
											buff[(row - y_step) * newWidth + col + 1] +
											(buff[(row - y_step) * newWidth + col + 1] -
											buff[(row)* newWidth + col + 1]) *
											y / y_step;
					if (src.sizePerPixel > 2)buff[(row + y - y_step) * newWidth + col + 2] =
											buff[(row - y_step) * newWidth + col + 2] +
											(buff[(row - y_step) * newWidth + col + 2] -
											buff[(row)* newWidth + col + 2]) *
											y / y_step;
				}
			}
		}
	}

	src.data.clear();
	src.data = std::move(buff);

	src.height = newHeight;
	src.width = newWidth;

	f_state = ENG_READY;
	return ENG_READY;
}

/*
Assume the origin is in the upper left corner of the coordinate system

origin
	+--------------------. x+
	| \ [angle]
	|  \
	|   \
	|    \
	|     \
	|
	|
	y+

So, New X = height*cos(a+90)+width*cos(a)
	New Y = height*sin(a+90)+width*sin(a)

	Origin image's col row . new image's dx dy is:

	dX=(height - row)*cos(a+90)+col*cos(a)
	dY=row*sin(a+90)+col*sin(a)
	
An std::runtime_error is thrown if:
	pixels data size < height * width * sizePerPixel
	pixels sizePerPixel or height or width is 0

*/
EngineState PixelEngine::rotate(Pixels & src, myfloat angle, std::uint8_t mode)
{
	myassert(src.sizePerPixel && src.sizePerPixel <= 3 && src.height && src.width, "Src pixels member sizePerPixel or height or width is 0. peg::PixelEngine::rotate");
	myassert(src.data.size() >= (src.height * src.width * src.sizePerPixel), "Bad pixels size. peg::PixelEngine::rotate");

	if (f_state != ENG_READY)
		return ENG_BUSSY;

	f_state = ENG_RUNNING;
	//double angleR = ((abs(angle) - int(abs(angle) / 90) * 90) * 3.14 / 180);
	std::int16_t angleR = abs(angle) - int(abs(angle) / 90) * 90;
	bool b_change;
	switch ((int(angle) / 90) % 4)
	{
	case 0:
	case 2:
	default:
		b_change = false;
		break;
	case 1:
	case 3:
		b_change = true;
		break;
	}

	//std::uint16_t x = src.height * abs(sin(angleR)) + src.width * abs(cos(angleR));
	std::uint16_t x = src.height * abs(mysin(angleR)) + src.width * abs(mycos(angleR));

	//std::uint16_t y = src.height * abs(cos(angleR)) + src.width * abs(sin(angleR));
	std::uint16_t y = src.height * abs(mycos(angleR)) + src.width * abs(mysin(angleR));

	std::uint16_t dx = 0, dy = 0;
	std::vector<std::uint8_t> buff(x * y * src.sizePerPixel);

	for (std::size_t row = 0; row < src.height; row+=src.sizePerPixel)
	{
		for (std::size_t col = 0; col < src.width; col += src.sizePerPixel)
		{
			dy = row * mycos(angleR) + col * mysin(angleR);
			dx = (src.height - row) * mysin(angleR) + col * mycos(angleR);
			/*
			dx = b_change ? row * cos(angleR) + col * sin(angleR) :
				((src.height - row)*sin(angleR) + col * cos(angleR));
			dy = b_change ? ((src.height - row)*sin(angleR) + col * cos(angleR)):
				(row * cos(angleR) + col * sin(angleR));*/
			buff[b_change ? (dx * y + y - dy + 0) : (dx + dy * x + 0)] = src.data[row * src.width + col + 0];
			if (src.sizePerPixel > 1)buff[b_change ? (dx * y + y - dy + 1) : (dx + dy * x + 1)] = src.data[row * src.width + col + 1];
			if (src.sizePerPixel > 2)buff[b_change ? (dx * y + y - dy + 2) : (dx + dy * x + 2)] = src.data[row * src.width + col + 2];
		}
	}

	src.data.clear();
	src.data = std::move(buff);
	src.height = b_change ? x : y;
	src.width = b_change ? y : x;

	return f_state = ENG_READY;
}

/*
Will flip pixels

Mode: 0: flip vertically
	  1: flip horizontally

selectLine: 0: Flip the entire image (Default)
			N: Flip the first N row/col


An std::runtime_error is thrown if:
	pixels data size < height * width * sizePerPixel
	pixels sizePerPixel or height or width is 0
*/

EngineState PixelEngine::flip(Pixels & src, std::uint8_t mode, std::uint16_t selectLine = 0)
{
	myassert(src.sizePerPixel && src.sizePerPixel <= 3 && src.height && src.width, "Src pixels member sizePerPixel or height or width is 0. peg::PixelEngine::flip");
	myassert(src.data.size() >= (src.height * src.width * src.sizePerPixel), "Bad pixels size. peg::PixelEngine::flip");

	if (f_state != ENG_READY)
		return ENG_BUSSY;

	f_state = ENG_RUNNING;
	std::uint8_t buff[3] = {};
	switch (mode)
	{
	case 0:
		// vertically
		for (std::size_t row = 0; row < src.height / 2 && row < selectLine; row+=src.sizePerPixel)
			for (std::size_t col = 0; col < src.width; col += src.sizePerPixel)
			{
				buff[0] = src.data[row * src.width + col + 0];
				src.data[row * src.width + col + 0] = src.data[(src.height - row - 1) * src.width + col + 0];
				src.data[(src.height - row - 1) * src.width + col + 0] = buff[0];
				if (src.sizePerPixel > 1) {
					buff[1] = src.data[row * src.width + col + 1];
					src.data[row * src.width + col + 1] = src.data[(src.height - row - 1) * src.width + col + 1];
					src.data[(src.height - row - 1) * src.width + col + 1] = buff[1];
				}
				if (src.sizePerPixel > 2) {
					buff[2] = src.data[row * src.width + col + 2];
					src.data[row * src.width + col + 2] = src.data[(src.height - row - 1) * src.width + col + 2];
					src.data[(src.height - row - 1) * src.width + col + 2] = buff[2];
				}
			}
				
		break;
	default:
		// horizontally
		for (std::size_t row = 0; row < src.height; row += src.sizePerPixel)
			for (std::size_t col = 0; col < src.width / 2 && col < selectLine; row += src.sizePerPixel)
			{
				buff[0] = src.data[row * src.width + col + 0];
				src.data[row * src.width + col + 0] = src.data[row * src.width + (src.width - col) + 0];
				src.data[row * src.width + (src.width - col) + 0] = buff[0];
				if (src.sizePerPixel > 1) {
					buff[1] = src.data[row * src.width + col + 1];
					src.data[row * src.width + col + 1] = src.data[row * src.width + (src.width - col) + 1];
					src.data[row * src.width + (src.width - col) + 1] = buff[1];
				}
				if (src.sizePerPixel > 2) {
					buff[2] = src.data[row * src.width + col + 2];
					src.data[row * src.width + col + 2] = src.data[row * src.width + (src.width - col) + 2];
					src.data[row * src.width + (src.width - col) + 2] = buff[2];
				}
			}
		break;
	}

	return f_state = ENG_READY;
}

EngineState PixelEngine::HOG(
	Pixels const & src, Pixels & hog, Matrix & mX, Matrix & mY, 
	std::uint16_t startX,
	std::uint16_t startY, 
	std::uint16_t endX,
	std::uint16_t endY,
	std::size_t particle, 
	bool isWeighted)
{
	myassert(src.sizePerPixel && src.sizePerPixel <= 3 && src.height && src.width, "Src pixels member sizePerPixel or height or width is 0. peg::PixelEngine::HOG");
	myassert(src.data.size() >= (src.height * src.width * src.sizePerPixel), "Bad pixels size. peg::PixelEngine::HOG");
	myassert(mX.x >= 1 && mX.y >= 1 && mX.y <= src.height && mX.x <= src.width, "Bad matrix size. peg::PixelEngine::HOG");
	myassert(mX.data.size() >= mX.x * mX.y, "Bad matrix buffer size. peg::PixelEngine::HOG");
	myassert(mY.x >= 1 && mY.y >= 1 && mY.y <= src.height && mY.x <= src.width, "Bad matrix size. peg::PixelEngine::HOG");
	myassert(mY.data.size() >= mY.x * mY.y, "Bad matrix buffer size. peg::PixelEngine::HOG");
	myassert(mY.x == mX.y && mY.y == mX.x, "Matrix asymmetry. peg::PixelEngine::HOG");
	if (f_state != ENG_READY)
		return ENG_BUSSY;
	f_state = ENG_RUNNING;

	particle = particle < 2 ? 2 : particle;
	hog.width = particle;
	hog.height = 1;
	hog.sizePerPixel = 1;
	if (PixelsInit(hog) != ENG_SUCCESS)
	{
		f_state = ENG_READY;
		return ENG_MEMERY_INSUFFICIENT;
	}
	// Choose MAX edge in two matrices
	std::size_t edgeX = mX.x > mY.x ? mX.x : mY.x;
	std::size_t edgeY = mX.y > mY.y ? mX.y : mY.y;
	std::size_t hit, subhit;
	mydouble cx = 0, cy = 0, ori = 0, g = 0;
	myfloat dorg = 360 / particle;
	myfloat d2org = dorg / 2;
	// Check if the selection can accommodate the window
	startX = startX < (src.width - edgeX / 2) ? startX : 0;
	startY = startY < (src.height - edgeY / 2) ? startX : 0;
	// Check if the selection can accommodate the window
	endX = endX < edgeX / 2 ? edgeX : src.width;
	endY = endY < edgeY / 2 ? edgeY : src.height;

	for (std::size_t y = startY + edgeY / 2; y < endY - edgeY / 2; y+=src.sizePerPixel)
	{
		for (std::size_t x = startX + edgeX / 2; x < endX - edgeX / 2; x += src.sizePerPixel)
		{
			// Calculate single point X gradient value
			for (std::size_t y1 = 0; y1 < mX.y; y1++)
			{
				for (std::size_t x1 = 0; x1 < mX.x; x1++)
				{
					cx += src.data[(x - mX.x / 2 + x1) + (y - mX.y / 2 + y1) * src.width + 0] * mX.data[y1 * mX.x + x1];
					if (src.sizePerPixel > 1)cx += src.data[(x - mX.x / 2 + x1) + (y - mX.y / 2 + y1) * src.width + 1] * mX.data[y1 * mX.x + x1];
					if (src.sizePerPixel > 2)cx += src.data[(x - mX.x / 2 + x1) + (y - mX.y / 2 + y1) * src.width + 2] * mX.data[y1 * mX.x + x1];
				}
				cx /= src.sizePerPixel * mX.x;
			}
			cx /= mX.y;
			// Calculate single point Y gradient value
			for (std::size_t y1 = 0; y1 < mY.y; y1++)
			{
				for (std::size_t x1 = 0; x1 < mY.x; x1++)
				{
					cy += src.data[(x - mY.x / 2 + x1) + (y - mY.y / 2 + y1) * src.width + 0] * mY.data[y1 * mY.x + x1];
					if (src.sizePerPixel > 1)cy += src.data[(x - mY.x / 2 + x1) + (y - mY.y / 2 + y1) * src.width + 1] * mY.data[y1 * mY.x + x1];
					if (src.sizePerPixel > 1)cy += src.data[(x - mY.x / 2 + x1) + (y - mY.y / 2 + y1) * src.width + 2] * mY.data[y1 * mY.x + x1];
				}
				cy /= src.sizePerPixel * mY.x;
			}
			cy /= mY.y;
			// Calculate gradient value
			g = sqrt(cx * cx + cy * cy);
			ori = atan2(cy, cx) * 180 / 3.14159f + 180;
			hit = ((std::uint16_t)ori) / ((std::uint16_t)dorg);
			subhit = ((std::uint16_t)ori) / ((std::uint16_t)d2org);
			hit = hit == particle ? hit - 1 : hit;
			subhit = subhit == 2 * particle ? subhit - 1 : subhit;
			if (isWeighted)
			{
				if (subhit % 1)
				{
					//uper
					hog.data[hit] += g * (ori - d2org * subhit) / d2org;
					hog.data[hit+1 == particle ? 0: hit+1] += g * (1- (ori - d2org * subhit) / d2org);
				}
				else
				{
					//lower
					hog.data[hit] += g * (ori - d2org * subhit) / d2org;
					hog.data[hit == 0 ? particle-1 : hit-1] += g * (1- (ori - d2org * subhit) / d2org);
				}
				g *(ori - (360 / particle) * hit);
			}
			else
			{
				hog.data[hit] += 1;
			}
		}
	}

	return f_state = ENG_READY;
}

} // namespace peg

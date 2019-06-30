# CMakeList.txt: MakeProject 的 CMake 项目，在此处包括源代码并定义
# 项目特定的逻辑。
#
cmake_minimum_required (VERSION 3.8)
project(CharProject)
set(CMAKE_CXX_FLAGS "-std=c++14")
set(CMAKE_CXX_COMPILER "g++")
set(CMAKE_C_COMPILER "g++")
target_link_libraries(CharProject boost_system)
# 将源代码添加到此项目的可执行文件。
add_executable (CharProject "src/charProject.cpp" "src/include/PixelEngine.h" "src/PixelEngine.cpp" "src/include/PNM_IO.h" "src/PNM_IO.cpp")

# TODO: 如有需要，请添加测试并安装目标。

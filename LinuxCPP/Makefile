CC= g++ -Wswitch -Wno-deprecated-declarations -Wempty-body -Wconversion -Wreturn-type -Wparentheses -Wno-pointer-sign -Wno-format -Wunreachable-code -Wunused-function -Wunused-value -Wunused-variable 
VERSION = 0_91
TARGET = CharP_$(VERSION).out
CUDA_PATH = /usr/local/cuda/

# -Wall:waring -fopenmp 
SRCC		:= $(wildcard ./src/*.c)
SRCCPP		:= $(wildcard ./src/*.cpp)
SRCCU		:= $(wildcard ./src/include/CUDA/*.cu)
SRC		:= $(SRCC) $(SRCCPP) $(SRCCU)
OBJC  		:= $(SRCC:	%.c	=$(BUILD)%.o) 
OBJCPP  	:= $(SRCCPP:	%.cpp	=$(BUILD)%.o) 
OBJCU		:= $(SRCCU:	%.cu	=$(BUILD)%.o)
OBJ 		:= $(OBJC) $(OBJCPP) $(OBJCU)
DEP			:= $(OBJ:%.o=%.d)
BUILD = ./build/

NVCC = $(CUDA_PATH)/bin/nvcc
CUDA_LIBRARY_DIR = $(CUDA_PATH)/lib64/
CUDA_INCLUDE_DIR = $(CUDA_PATH)/include/
LDFLAGS 	 = -L$(CUDA_LIBRARY_DIR) -lcudart 
CFLAGS		 = -O0 -std=c++14 -I$(CUDA_INCLUDE_DIR) -pthread

$(TARGET):	$(OBJ)
	$(CC) -o $(TARGET) $(OBJ) $(CFLAGS)

-include $(INC_DIR)$(DEP)

%.o:	./src/include/CUDA/%.cu Makefile
	$(NVCC) --cuda -MMD -MP $<

%.o:	./src/%.c Makefile
	$(CC) -c -MMD -MP $< $(CFLAGS)

%.o:	./src/%.cpp Makefile
	$(CC) -c -MMD -MP $< $(CFLAGS)

clean:
	rm -f *.o *.d *.a $(TARGET)

# vim: set noexpandtab :


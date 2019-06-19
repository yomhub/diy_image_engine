CC			 = g++
VERSION = 0_91
TARGET = CharP_$(VERSION).o

# -Wall:waring
CFLAGS		 = -O0 -fopenmp -std=c++14
SRCC		:= $(wildcard ./src/*.c)
SRCCPP		:= $(wildcard ./src/*.cpp)
SRC		:= $(SRCC) $(SRCCPP)
OBJC  		:= $(SRCC:%.c=$(BUILD)%.o) 
OBJCPP  	:= $(SRCCPP:%.cpp=$(BUILD)%.o) 
OBJ 		:= $(OBJC) $(OBJCPP)
DEP			:= $(OBJ:%.o=%.d)
BUILD = ./build/

$(TARGET):	$(OBJ)
	$(CC) -o $(TARGET) $(OBJ) $(CFLAGS)

-include $(INC_DIR)$(DEP)

%.o:	./src/%.c Makefile
	$(CC) -c -MMD -MP $< $(CFLAGS)

%.o:	./src/%.cpp Makefile
	$(CC) -c -MMD -MP $< $(CFLAGS)

clean:
	rm -f *.o *.d *.a $(TARGET)

# vim: set noexpandtab :


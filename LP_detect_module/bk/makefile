################################################################################
# Automatically-generated file. Do not edit!
################################################################################

-include ../makefile.init

RM := rm -rf

SWIG = swig3.0
SWIG_INCS = -Iopencv-swig/lib -I/usr/include -I/usr/local/include/opencv4 
CXX = g++ -std=c++11 -fPIC -fpermissive  -I/usr/local/cuda/include -I./cpp -I./cpp/tensorRT -I./cpp/tensorRT/common -I/usr/include/x86_64-linux-gnu -I/usr/include/jsoncpp/ -I/usr/local/include/opencv4
INC_PATHS = -I/usr/local/cuda/include -I./cpp -I./cpp/tensorRT -I./cpp/tensorRT/common -I/usr/include/x86_64-linux-gnu -I/usr/include/jsoncpp/ -I/usr/local/include/opencv4
LIB_PATHS = -L/usr/lib/x86_64-linux-gnu -L/usr/local/lib -L./build_lib -L/usr/local/cuda/lib64 
FLAGS = -fPIC -fpermissive 
SHARED = -shared 

-include objects.mk
-include subdir.mk

ifneq ($(MAKECMDGOALS),clean)
ifneq ($(strip $(CC_DEPS)),)
-include $(CC_DEPS)
endif
ifneq ($(strip $(C++_DEPS)),)
-include $(C++_DEPS)
endif
ifneq ($(strip $(C_UPPER_DEPS)),)
-include $(C_UPPER_DEPS)
endif
ifneq ($(strip $(CXX_DEPS)),)
-include $(CXX_DEPS)
endif
ifneq ($(strip $(C_DEPS)),)
-include $(C_DEPS)
endif
ifneq ($(strip $(CPP_DEPS)),)
-include $(CPP_DEPS)
endif
endif

-include ../makefile.defs

# Add inputs and outputs from these tool invocations to the build variables 
PYTHONI = -I/usr/include/python3.8/
PYTHONL = -Xlinker -export-dynamic

SRCS_I = ./FaceDetect.i

TARGET_WRAP = ./FaceDetect_wrap.cxx
TARGET_LIB = _FaceDetect.so
TARGET_PYTHON = FaceDetect.py FaceDetect.pyc

OBJS_WRAP = ./FaceDetect_wrap.o

# All Target
all:face_detect

# Tool invocations
face_detect: $(OBJS) $(USER_OBJS)
	@echo 'Building target: $@'
	@echo 'Invoking: Cross G++ Linker'
	@echo 'Building python module: $@'
	$(SWIG) -c++ -python $(SWIG_INCS) -o $(TARGET_WRAP) $(SRCS_I)
	@echo '1 $@'
	$(CXX) $(FLAGS) $(PYTHONI) -c $(TARGET_WRAP) -o $(OBJS_WRAP) $(LIB_PATHS) $(LIBS)
	@echo '2 $@'
	$(CXX) $(PYTHONL) $(LIBFLAGS) $(SHARED) $(OBJS_WRAP) -o $(TARGET_LIB) $(INC_PATHS) $(LIB_PATHS) $(LIBS)
	@echo 'Done $@'
	@echo ' '
clean:
	$(RM) $(CC_DEPS)$(C++_DEPS)$(EXECUTABLES)$(OBJS)$(C_UPPER_DEPS)$(CXX_DEPS)$(C_DEPS)$(CPP_DEPS) $(OBJS_WRAP)
	$(RM) $(TARGET_WRAP) $(TARGET_LIB) $(TARGET_PYTHON)

.PHONY: all clean dependents

-include ../makefile.targets

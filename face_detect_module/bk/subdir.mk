################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
./cpp/face_detect_retinaTRT.cpp

OBJS += \
./cpp/face_detect_retinaTRT.o

CPP_DEPS += \
./cpp/face_detect_retinaTRT.d

# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: Cross G++ Compiler'
	g++ -std=c++0x -fPIC -fpermissive -I./cpp -I./cpp/tensorRT/common -I./cpp/tensorRT -I/usr/local/include/opencv4 -I/usr/include/jsoncpp/ -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '



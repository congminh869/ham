cmake_minimum_required(VERSION 3.5)
project(FaceDetect2)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
find_package(CUDA REQUIRED)

#set(CUDA_GEN_CODE "-gencode=arch=compute_72,code=sm_72")

#Build with PC
add_definitions(-DBUILD_ON_PC_HW)
#Build with Jetson
#add_definitions(-DBUILD_ON_JETSON_HW)

set(CMAKE_CXX_FLAGS  "-lpthread")
set(CMAKE_CXX_FLAGS  "-fpermissive")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS  "-ldl")
set(CMAKE_CXX_FLAGS  "-O3")
set(CMAKE_BUILD_TYPE Release)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_DIR}/include)
include_directories(/usr/local/cuda/include)
set(PROTOBUF_DIR "/home/evnadmin/tensorRT_cpp/protobuf/src/")


include_directories(
    ${PROJECT_SOURCE_DIR}/cpp
    ${OpenCV_INCLUDE_DIRS}
    ${PROTOBUF_DIR}
    /usr/include/x86_64-linux-gnu/
    cpp/tensorRT
    cpp/tensorRT/common
    /usr/include/jsoncpp/
    cpp
)

link_directories(
    ${PROTOBUF_DIR}/.libs
    /usr/local/lib/
    /usr/local/cuda/lib64
    /usr/lib/x86_64-linux-gnu/
)

set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -O0 -Wfatal-errors -pthread -fPIC -w -g")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 -D_GLIBCXX_USE_CXX11_ABI=1 -O0 -Xcompiler -g -w ${CUDA_GEN_CODE}")

AUX_SOURCE_DIRECTORY(./cpp/ DIR_SRCS)
AUX_SOURCE_DIRECTORY(./cpp/sort_tracking DIR_SRCS_TRACKING)
AUX_SOURCE_DIRECTORY(./cpp/retinaTRT_face DIR_SRCS_RETINA_TRT)
AUX_SOURCE_DIRECTORY(./cpp/tensorRT DIR_SRCS_TENSORRT)
AUX_SOURCE_DIRECTORY(./cpp/tensorRT/common DIR_SRCS_TENSORRT_COMMON)
AUX_SOURCE_DIRECTORY(./cpp/tensorRT/infer DIR_SRCS_TENSORRT_INFER)

#Build library
#add_library(${PROJECT_NAME} SHARED ${DIR_SRCS})
#target_link_libraries(${PROJECT_NAME} pthread)
#target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})


cuda_add_library(cudaplugins SHARED ${DIR_SRCS_TENSORRT} ${DIR_SRCS_RETINA_TRT} ${DIR_SRCS_TRACKING} ${DIR_SRCS_TENSORRT_COMMON} ${DIR_SRCS_TENSORRT_INFER})
#add_library(face_detect SHARED ${DIR_SRCS})

set(CMAKE_CXX_FLAGS  "-fpermissive")
target_link_libraries(cudaplugins ${OpenCV_LIBS})
target_link_libraries(cudaplugins pthread)
target_link_libraries(cudaplugins openblas)
target_link_libraries(cudaplugins cublas)
target_link_libraries(cudaplugins cuda)
target_link_libraries(cudaplugins cudart)
target_link_libraries(cudaplugins cudnn)
target_link_libraries(cudaplugins protobuf)
target_link_libraries(cudaplugins nvinfer)
target_link_libraries(cudaplugins nvinfer_plugin)
target_link_libraries(cudaplugins nvparsers)

CXX = g++ -std=c++11
INC_PATHS =  -I/usr/local/cuda/include -I../cpp -I../cpp/tensorRT -I../cpp/tensorRT/common -I/usr/include/x86_64-linux-gnu -I/usr/include/jsoncpp/ -I/usr/local/include/opencv4
LIB_PATHS = -L/usr/lib/x86_64-linux-gnu -L/usr/local/lib -L../build_lib -L/usr/local/cuda/lib64 
LIBS = -lpthread -lopencv_imgcodecs -lopencv_core -lopencv_videoio -lopencv_imgproc -lopencv_highgui -ljsoncpp -lcrypto -lssl -lcudaplugins
FLAGS = -fPIC -fpermissive 
SHARED = -shared 
RM = rm -rf
TARGET = run_test

SRCS_CPP = ../cpp/face_detect_retinaTRT.cpp
SRCS_CPP += ./main.cpp
OBJS_CPP = ./cpp/face_detect_retinaTRT.o
OBJS_CPP += ./main.o

all: 
	$(CXX) $(FLAGS) $(LIBFLAGS) $(SRCS_CPP) -o $(TARGET) $(INC_PATHS) $(LIB_PATHS) $(LIBS)

clean:
	$(RM) $(OBJS_CPP)
	$(RM) $(TARGET)

	


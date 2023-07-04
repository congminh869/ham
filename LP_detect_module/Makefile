CXX = g++ -std=c++11
INC_PATHS =  -I./cpp/sort_tracking -I/usr/local/cuda/include -I./cpp -I./cpp/tensorRT -I./cpp/tensorRT/common -I/usr/include/x86_64-linux-gnu -I/usr/include/jsoncpp/ -I/usr/local/include/opencv4
LIB_PATHS = -L/usr/lib/x86_64-linux-gnu -L/usr/local/lib -L./build_lib -L/usr/local/cuda/lib64 
LIBS = -lpthread -lopencv_imgcodecs -lopencv_core -lopencv_videoio -lopencv_imgproc -lopencv_highgui -ljsoncpp -lcrypto -lssl -lcudaplugins
FLAGS = -fPIC -fpermissive 
SHARED = -shared 
RM = rm -rf

SWIG = swig3.0
SWIG_INCS = -Iopencv-swig/lib -I/usr/include -I/usr/local/include/opencv4 

# the python interface through swig
PYTHONI = -I/usr/include/python3.8/
PYTHONL = -Xlinker -export-dynamic

SRCS_CPP = ./cpp/face_detect_retinaTRT.cpp

SRCS_I = ./FaceDetect.i

TARGET_WRAP = ./FaceDetect_wrap.cxx
TARGET_LIB = _FaceDetect.so
TARGET_PYTHON = FaceDetect.py FaceDetect.pyc


OBJS_CPP = ./cpp/face_detect_retinaTRT.o
OBJS_WRAP = ./FaceDetect_wrap.o


all: 
	$(CXX) $(FLAGS) -c $(SRCS_CPP) -o $(OBJS_CPP) $(INC_PATHS) $(LIB_PATHS) $(LIBS)
	$(SWIG) -c++ -python $(SWIG_INCS) -o $(TARGET_WRAP) $(SRCS_I)
	$(CXX) $(FLAGS) $(PYTHONI) -c $(TARGET_WRAP) -o $(OBJS_WRAP) $(INC_PATHS) $(LIB_PATHS) $(LIBS)
	$(CXX) $(PYTHONL) $(LIBFLAGS) $(SHARED) $(OBJS_CPP) $(OBJS_WRAP) -o $(TARGET_LIB) $(INC_PATHS) $(LIB_PATHS) $(LIBS)

clean:
	$(RM) $(OBJS_CPP) $(OBJS_WRAP)
	$(RM) $(TARGET_WRAP) $(TARGET_LIB) $(TARGET_PYTHON)

	


%module FaceDetect

%include <opencv.i>

%cv_instantiate_all_defaults

%{
    #include "./cpp/common.h"
	#include "./cpp/face_detect.h"
%}

%include <std_string.i>
%include <std_vector.i> 

%template(DetectBoxVector) std::vector<DetectBox>;
%template(PointsVector) std::vector<cv::Point>;

%include "./cpp/common.h"
%include "./cpp/face_detect.h"





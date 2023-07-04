#ifndef FACE_DETECT_H
#define FACE_DETECT_H

#include <retinaTRT_face/retinaface.hpp>
#include <tensorRT/infer/trt_infer.hpp>
#include <tensorRT/common/ilogger.hpp>
#include <time.h>
#include <cstring>
#include "common.h"
#include "sort_tracking.h"

std::vector<DetectBox> NMS(const std::vector<DetectBox> &box, float nms);

class FaceDetectRetinaTRT
{
public:
	FaceDetectRetinaTRT(std::string file_config);
	~FaceDetectRetinaTRT();

	std::vector<DetectBox> Detect(int id_cam, cv::Mat img);
	std::vector<DetectBox> DetectBatch(int batch_size, int id_cam, cv::Mat img);
private:
	int f_load_config(std::string file_config);
	int load_config(std::string str_config);

	bool flag_init = false;

	//Retina TRT
	std::shared_ptr<RetinaFace::Infer> detecter;
	int rows_img_detect =  800;
	int cols_img_detect = 800;
	std::string model_path ="";
	int device_id = 0;
	
	std::vector<cv::Mat> images_b;
	std::list<SortTracking> * trackers;
};

#endif

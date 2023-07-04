#include "face_detect.h"
#include <json/json.h>
#include <iostream>
#include <string.h>

using namespace FaceDetector;
using namespace RetinaFace;

std::vector<DetectBox> NMS(const vector<struct DetectBox> &box, float nms){
	size_t count = box.size();
	std::vector<pair<size_t, float>> order(count);
	for (size_t i = 0; i < count; ++i) {
		order[i].first = i;
		order[i].second = box[i].class_confidence;
	}

	sort(order.begin(), order.end(),
			[](const pair<int, float> &ls, const pair<int, float> &rs) {
				return ls.second > rs.second;
			});

	std::vector<int> keep;
	std::vector<bool> exist_box(count, true);
	for (size_t _i = 0; _i < count; ++_i) {
		size_t i = order[_i].first;
		float x1, y1, x2, y2, w, h, iarea, jarea, inter, ovr;
		if (!exist_box[i])
			continue;
		keep.push_back(i);
		for (size_t _j = _i + 1; _j < count; ++_j) {
			size_t j = order[_j].first;
			if (!exist_box[j])
				continue;
			// DEBUG_RectFailed(box[i].bbox);
			x1 = int(max(box[i].bbox.x, box[j].bbox.x));
			y1 = int(max(box[i].bbox.y, box[j].bbox.y));
			x2 = int(min(box[i].bbox.x + box[i].bbox.width, box[j].bbox.x + box[j].bbox.width));
			y2 = int(min(box[i].bbox.y + box[i].bbox.height, box[j].bbox.y + box[j].bbox.height));
			w = int(max(float(0.0), x2 - x1 + 1));
			h = int(max(float(0.0), y2 - y1 + 1));
			iarea = box[i].bbox.width * box[i].bbox.height;
			jarea = box[j].bbox.width * box[j].bbox.height;
			inter = w * h;
			ovr = inter / (iarea + jarea - inter);
			if (ovr >= nms)
				exist_box[j] = false;
		}
	}

	std::vector<struct DetectBox> result;
	result.reserve(keep.size());
	for (size_t i = 0; i < keep.size(); ++i) {
		result.push_back(box[keep[i]]);
	}

	return result;
}


FaceDetectRetinaTRT::FaceDetectRetinaTRT(std::string file_config)
{
	trackers = new std::list<SortTracking>();
    trackers->clear();
	TRT::set_device(0);
	if(f_load_config(file_config) < 0)
	{
		std::cout << "Init FaceDetectRetinaTRT failed. Exit program!" << std::endl;
		exit(-1);
	}
	flag_init = true;
	detecter = RetinaFace::create_infer(model_path, 0, 0.7);
	if(!detecter)
	{
		flag_init = false;
	}
}

FaceDetectRetinaTRT::~FaceDetectRetinaTRT()
{

}

std::vector<DetectBox> FaceDetectRetinaTRT::Detect(int id_cam, cv::Mat img)
{
	std::vector<DetectBox> result, result_no_tracking; 
	//cv::Mat det_img;
	//cv::resize(img, det_img, Size(cols_img_detect, rows_img_detect));
	//Detect face
	std::vector<cv::Mat> images;
	// DEBUG_MatEmpty(img);
	images.emplace_back(img);
	detecter->commits(images).back().get();
	auto images_faces = detecter->commits(images);
	images_faces.back().get();
	//Convert BoxArray to DectectBox
	for(int i = 0 ; i < images_faces.size(); i++)
	{
		auto faces  = images_faces[i].get();
		for(int j = 0; j < faces.size(); ++j){
            auto& face = faces[j];

            DetectBox result_db;

           	if(face.left < 0)
           		continue;
           	if(face.top < 0)
           		continue;           	
           	if(face.right > img.cols)
           		continue;
           	if(face.bottom > img.rows)
           		continue;

			result_db.bbox = cv::Rect(int(face.left), int(face.top), int(face.right - face.left), int(face.bottom - face.top));
			// DEBUG_RectFailed(result_db.bbox);
			result_db.class_id = 0;
			result_db.class_confidence = face.confidence;
			for (int k = 0; k < 5; ++k) {
				result_db.landmark_points.push_back(cv::Point(face.landmark[k*2+0] - face.left,face.landmark[k*2+1] - face.top));
			}
	    	result_db.area = result_db.bbox.width * result_db.bbox.height ;
	    	result_db.camera_id = id_cam;
        	
        	if(result_db.class_confidence > 0.6 && result_db.bbox.height >= 60)
        		result_no_tracking.push_back(result_db); 
        }
	}
	std::vector<DetectBox> result_no_tracking_nms = NMS(result_no_tracking, 0.35);

	//Crop face image
	for(int i = 0 ; i < result_no_tracking_nms.size(); i++)
	{
		// DEBUG_MatEmpty(img);
		// DEBUG_RectFailed(result_no_tracking_nms[i].bbox);
		cv::Mat crop_face = img(result_no_tracking_nms[i].bbox);
		result_no_tracking_nms[i].img_face = crop_face.clone();
		result_no_tracking_nms[i].x_org = result_no_tracking_nms[i].bbox.x;
		result_no_tracking_nms[i].y_org = result_no_tracking_nms[i].bbox.y;
		// result.push_back(result_no_tracking_nms[i]);
	}

	// Tracking
	bool f_existed_sort = false;
	for(int i = 0; i < trackers->size(); i++)
	{
	    std::list<SortTracking>::iterator it = trackers->begin();
		std::advance(it, i);
		if(id_cam == (*it).cam_id)
		{
			f_existed_sort = true;
			(*it).update(img,result_no_tracking_nms, result);
			break;
		}		
	}

	if(f_existed_sort == false)
	{
	    SortTracking tracker(id_cam);
	    trackers->push_back(tracker);

	    std::list<SortTracking>::iterator it = trackers->end();
	    (*it).update(img, result_no_tracking_nms, result);
	}
	return result;

	// return result_no_tracking_nms;
}

std::vector<DetectBox> FaceDetectRetinaTRT::DetectBatch(int batch_size, int id_cam, cv::Mat img)
{
	std::vector<DetectBox> result, result_no_tracking;
	int pos_camera = -1;

	//cv::Mat det_img;
	//cv::resize(img, det_img, Size(cols_img_detect, rows_img_detect));
	
	images_b.emplace_back(img);
	
	if(images_b.size() < batch_size)
	    return result;
	    
	//Detect face
	detecter->commits(images_b).back().get();
	auto images_faces = detecter->commits(images_b);
	images_faces.back().get();
	//images_b.clear();

	//Convert BoxArray to DectectBox
	for(int i = 0 ; i < images_faces.size(); i++)
	{
		auto faces  = images_faces[i].get();
		for(int j = 0; j < faces.size(); ++j){
            auto& face = faces[j];

            DetectBox result_db;
			result_db.bbox = cv::Rect_<float>(face.left, face.top, face.right - face.left, face.bottom - face.top);
			DEBUG_RectFailed(result_db.bbox);
			result_db.class_id = 0;
			result_db.class_confidence = face.confidence;
			for (int k = 0; k < 5; ++k) {
				result_db.landmark_points.push_back(cv::Point(face.landmark[k*2+0] - face.left,face.landmark[k*2+1] - face.top));
			}
	    	result_db.area = result_db.bbox.width * result_db.bbox.height ;
	    	result_db.camera_id = pos_camera;
        	if(result_db.class_confidence > 0.8 && result_db.bbox.height >= 120)
        		result_no_tracking.push_back(result_db);  
        }
	}
	std::vector<DetectBox> result_no_tracking_nms = NMS(result_no_tracking, 0.35);
	
	//Crop face image
	for(int i = 0 ; i < result_no_tracking_nms.size(); i++)
	{
		DEBUG_MatEmpty(img);
		DEBUG_RectFailed(result_no_tracking_nms[i].bbox);
		cv::Mat crop_face = img(result_no_tracking_nms[i].bbox);
		DEBUG_MatEmpty(crop_face);
		result_no_tracking_nms[i].img_face = crop_face.clone();
		result.push_back(result_no_tracking_nms[i]);
	}

	return result;
}



int FaceDetectRetinaTRT::f_load_config(std::string file_config)
{
	std::ifstream config_file(file_config);
	std::stringstream config;
	config << config_file.rdbuf();
	string config_str = config.str();

	return load_config(config_str);
}

int FaceDetectRetinaTRT::load_config(std::string str_config)
{
	Json::Value root;
    Json::Reader reader;
    
	bool parsingSuccessful = reader.parse(str_config, root);
    if ( !parsingSuccessful )
    {
        // report to the user the failure and their locations in the document.
        std::cout  << "Failed to parse configuration: "<< reader.getFormattedErrorMessages() << std::endl;
    }
    rows_img_detect = root["rows_img_detect"].asInt();
    cols_img_detect = root["cols_img_detect"].asInt();
    model_path = root["model_path"].asString();
    device_id = root["device"].asInt();

	return 1;
}


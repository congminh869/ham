#ifndef COMMON_H
#define COMMON_H

#include <string>
#include <vector>
#include <fstream>
#include <cassert>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/video/tracking.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <NvInfer.h>
#include <opencv2/calib3d.hpp>

//#define CAFFE_TO_TENSORRT


//#define ENABLE_TRACKING_THREAD
#define DEBUG_MAT_EMPTY
#define DEBUG_RECT_FAILED
//#define DEBUG_LANDMARK

#if NV_TENSORRT_MAJOR >= 8
#define TRT_NOEXCEPT noexcept
#define TRT_CONST_ENQUEUE const
#else
#define TRT_NOEXCEPT
#define TRT_CONST_ENQUEUE
#endif

/*Log color*/
#define RESET   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */
#define BLUE    "\033[34m"      /* Blue */
#define MAGENTA "\033[35m"      /* Magenta */
#define CYAN    "\033[36m"      /* Cyan */
#define WHITE   "\033[37m"      /* White */
#define BOLDBLACK   "\033[1m\033[30m"      /* Bold Black */
#define BOLDRED     "\033[1m\033[31m"      /* Bold Red */
#define BOLDGREEN   "\033[1m\033[32m"      /* Bold Green */
#define BOLDYELLOW  "\033[1m\033[33m"      /* Bold Yellow */
#define BOLDBLUE    "\033[1m\033[34m"      /* Bold Blue */
#define BOLDMAGENTA "\033[1m\033[35m"      /* Bold Magenta */
#define BOLDCYAN    "\033[1m\033[36m"      /* Bold Cyan */
#define BOLDWHITE   "\033[1m\033[37m"      /* Bold White */

/*Default values*/
#define DEFAULT_SHOW_SCREEN 1
#define DEFAULT_SIMILARY_THR 0.7
#define DEFAULT_SIMILARY_THR_ADD 0.8
#define DEFAULT_RECOG_DISPLAY_LIST 5
#define DEFAULT_UNKNOWN_NAME "uk_"

#define DEFAULT_SERVER_DB_PATH "/static/db/feature.db"
#define DEFAULT_SERVER_HISTORY_IMG_PATH "/static/img"
#define DEFAULT_SERVER_HISTORY_NEW_FACES_PATH "/faces/*.jpg"
#define DEFAULT_SERVER_HISTORY_TRAINED_FACES_PATH "/trained"
#define DEFAULT_SERVER_FACE_DATABASE_PATH "/home/mq/crm/feature.db"

#define DEFAULT_LAUNCH_URL "/api/launch"
#define DEFAULT_IMAGE_URL "/api/image"
#define DEFAULT_HISTORY_URL "/api/history"
#define DEFAULT_PLAYNAME_URL "/api/playName"
#define DEFAULT_LOCALHOST "http://127.0.0.1:1234"
#define DEFAULT_CREATE_UNKNOW_USER_URL "/api/create_unknown"

#define DEFAULT_FACE_CAPTURE_RATE 1
#define DEFAULT_FACE_CAPTURE_MAX  20
#define DEFAULT_FACE_CAPTURE_SIZE 112
//#define DEFAULT_FACE_CAPTURE_SIZE 90
#define DEFAULT_FEATURE_SIZE 128
//#define DEFAULT_FACE_MIN_SIZE 120
#define DEFAULT_FACE_MIN_SIZE 112

#define DEFAULT_FRAME_WIDTH 720
#define DEFAULT_FRAME_HEIGH 1280

#define CHECK(status)                                   \
{                                                       \
    if (status != 0)                                    \
    {                                                   \
        std::cout << "Cuda failure: " << status;        \
        abort();                                        \
    }                                                   \
}


typedef struct DetectBox
{
    DetectBox() : camera_id(-1), class_id(-1), isRecognized(0),
                isUnknow(0), live_time(0), id_tracking(-1),
                area(0.0), exist(false),
                mask_confidence(-1.0) {};

    cv::Rect_<float> bbox;
    int camera_id;
    int class_id;
    int id_tracking;
    float class_confidence;
    std::vector<cv::Point> landmark_points;
    float mask_confidence;
    std::string name;
    int isRecognized;
    int isUnknow;
    std::chrono::steady_clock::time_point time_in;
    double live_time;
    float area;
    bool exist;
    uint64_t count_frame;
    cv::Mat img;
    cv::Mat img_face;
    int x_org;
    int y_org;
}DetectBox;

class Logger : public nvinfer1::ILogger
{
public:
    void log(nvinfer1::ILogger::Severity severity, const char* msg) TRT_NOEXCEPT override
    {
        // suppress info-level messages
        if (severity == Severity::kINFO) return;

        switch (severity)
        {
            case Severity::kINTERNAL_ERROR: std::cerr << "INTERNAL_ERROR: "; break;
            case Severity::kERROR: std::cerr << "ERROR: "; break;
            case Severity::kWARNING: std::cerr << "WARNING: "; break;
            case Severity::kINFO: std::cerr << "INFO: "; break;
            default: std::cerr << "UNKNOWN: "; break;
        }
        std::cerr << msg << std::endl;
    }
};


#ifdef DEBUG_MAT_EMPTY
#define DEBUG_MatEmpty(frame)                                                        \
{                                                                                    \
    if (frame.empty())                                                               \
    {                                                                                \
        std::cout << "[" << __FILE__ << "][" << __LINE__ << "]: Frame empty\n";        \
    }                                                                                \
}
#else
#define DEBUG_MatEmpty(frame){}                                                       
#endif

#ifdef DEBUG_RECT_FAILED
#define DEBUG_RectFailed(rect)                                                        \
{                                                                                    \
    if (rect.x < 0 || rect.y < 0 || rect.width < 0 || rect.height < 0)                                                               \
    {                                                                                \
        std::cout << "[" << __FILE__ << "][" << __LINE__ << "]: Rect (" << rect.x << "; " << rect.y << "; " << rect.width << "; " <<rect.height << ") failed\n"  ;        \
    }\
    else{                                                                               \
        std::cout << "[" << __FILE__ << "][" << __LINE__ << "]: Rect (" << rect.x << "; " << rect.y << "; " << rect.width << "; " <<rect.height << ")\n"  ;        \
    }                                                                                \
}
#else
#define DEBUG_RectFailed(rect){}                                                       
#endif

#ifdef DEBUG_LANDMARK
#define DEBUG_Landmark(lm)                                                        \
{                                                                                   \
    printf("[%s][%d]: LANDMARK = [(%d, %d); (%d, %d); (%d, %d); (%d, %d); (%d, %d)]\n", \
        __FILE__, __LINE__, lm[0].x, lm[0].y, lm[1].x, lm[1].y,lm[2].x, lm[2].y,lm[3].x, lm[3].y,lm[4].x, lm[4].y);  \
}
#define DEBUG_Landmark2(count, lm)                                                        \
{                                                                                   \
    printf("[%s][%d][Frame %d]: LANDMARK = [(%d, %d); (%d, %d); (%d, %d); (%d, %d); (%d, %d)]\n", \
        __FILE__, __LINE__,count, lm[0].x, lm[0].y, lm[1].x, lm[1].y,lm[2].x, lm[2].y,lm[3].x, lm[3].y,lm[4].x, lm[4].y);  \
} 
#else
#define DEBUG_Landmark(lm){}     
#define DEBUG_Landmark2(count, lm){}                                
#endif


#endif



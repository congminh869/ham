#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <common/ilogger.hpp>
#include <chrono>
#include "face_detect.h"


#define BATCH_SIZE 20

int main()
{
    cv::VideoCapture cap("rtsp://admin:MQ123456@192.168.6.132:554");
    if(!cap.isOpened())
    {
        printf("Cannnot open RTSP link\n");
        return -1;
    }
    
    FaceDetectRetinaTRT detecter("../config.txt");
    
    int count_batch = 0;
    double total_time = 0;
    while(1)
    {
        cv::Mat frame;
        cap >> frame;
        if(frame.empty())
        {
            continue;
        }
        cv::resize(frame, frame, cv::Size(800,450));
        auto t1 = iLogger::timestamp_now_float();
        detecter.DetectBatch(BATCH_SIZE,1, frame);
        count_batch++;
        auto t2 =iLogger::timestamp_now_float();
        auto time_span =t2 - t1; 
        total_time += time_span;
        if(count_batch == BATCH_SIZE){
            auto time_fee_per_image = total_time / 1 / BATCH_SIZE;
            INFO("***********************************************************************************************");
            INFO("Retinaface limit case performance: %.3f ms / image, fps = %.2f", time_fee_per_image, 1000 / time_fee_per_image);
            INFO("***********************************************************************************************");
          
            count_batch = 0;
        	//double fps_out = BATCH_SIZE/total_time;
        	//printf("FPS %2.5f\n", fps_out); 
        	total_time = 0; 
    	}         
    }
}


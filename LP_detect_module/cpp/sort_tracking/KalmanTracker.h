///////////////////////////////////////////////////////////////////////////////
// KalmanTracker.h: KalmanTracker Class Declaration

#ifndef KALMAN_H
#define KALMAN_H 2

#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

#define StateType Rect_<float>


// This class represents the internel state of individual tracked objects observed as bounding box.
class KalmanTracker
{
public:
	KalmanTracker()
	{
		init_kf(StateType());
		m_time_since_update = 0;
		m_hits = 0;
		m_hit_streak = 0;
		m_age = 0;
		m_id = kf_count;
		//kf_count++;
	}
	KalmanTracker(StateType initRect)
	{
		init_kf(initRect);
		m_time_since_update = 0;
		m_hits = 0;
		m_hit_streak = 0;
		m_age = 0;
		m_id = kf_count;
		kf_count++;
	}

	~KalmanTracker()
	{
		m_history.clear();
	}

	StateType predict();
	void update(StateType stateMat);
	
	StateType get_state();
	StateType get_rect_xysr(float cx, float cy, float s, float r);

	static int kf_count;

	int m_time_since_update;
	int m_hits;
	int m_hit_streak;
	int m_age;
	int m_id;
    bool m_isCounted = false;

    /*MQ-NDT variables*/
    /*int m_should_delete = 0;
    int m_isUpdating = 0;
	int m_recog_time = 0;
	int m_img_index = 0;
	int m_time_since_captured = 0;
	int m_is_recognized = 0;
	float mask_confidence  = 0;
	std::vector<Point2d> m_face5Pts;
	char name[128];
	StateType m_pos;

	double m_yaw = 0;
	double m_pitch =0;
	double m_roll = 0;*/

    cv::Mat img;
    cv::Mat img_face;
    int class_id;
    float class_confidence;
    std::vector<cv::Point> landmark_points;
    int x_org;
    int y_org;
    float area;
    int camera_id;
    int count_frame;

	/*End MQ-NDT variables*/

private:
	void init_kf(StateType stateMat);

	cv::KalmanFilter kf;
	cv::Mat measurement;

	std::vector<StateType> m_history;
};




#endif

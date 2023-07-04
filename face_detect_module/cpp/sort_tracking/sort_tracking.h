#ifndef SORT_TRACKING_H
#define SORT_TRACKING_H

#include <iostream>
#include <cstring>
#include <sys/types.h>
#include <stdlib.h>
#include <unistd.h>
#include <dirent.h>
#include <vector>
#include <set>
#include <assert.h>
#include <ctime>
#include <opencv2/core.hpp>
#include "common.h"
#include "Hungarian.h"
#include "KalmanTracker.h"

class SortTracking
{
public:
	SortTracking(int _monitor_id);
	~SortTracking();
	
	int update(cv::Mat m1, std::vector<DetectBox> input, std::vector<DetectBox> &result);

	void reset()
	{
		trackers.clear();
	}
	void setROITracking(int x_min = 0, int y_min = 0, int x_max = 0, int y_max = 0)
	{
		roi_tracking[0] = x_min;
		roi_tracking[1] = y_min;
		roi_tracking[2] = x_max;
		roi_tracking[3] = y_max;
	}

	int cam_id; 

protected:
	unsigned int frame_count = 0;
	int max_age = 10;
	int min_hits = 3;
	double iouThreshold = 0.05;
	std::vector<KalmanTracker> trackers;
	
	std::vector<cv::Rect_<float>> predictedBoxes;
	std::vector<vector<double>> iouMatrix;
	std::vector<int> assignment;
	std::set<int> unmatchedDetections;
	std::set<int> unmatchedTrajectories;
	std::set<int> allItems;
	std::set<int> matchedItems;
	std::vector<cv::Point> matchedPairs;

	unsigned int trkNum = 0;
	unsigned int detNum = 0;

	int total_id = 0;
	int roi_tracking[4] = {0, 0, 0, 0};
};

#endif

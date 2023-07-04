#include "sort_tracking.h"

SortTracking::SortTracking(int _cam_id) 
{
	cam_id =  _cam_id;
	setROITracking(15,15,15,15);
	trackers.clear();
	predictedBoxes.clear();
}

SortTracking::~SortTracking()
{

}

double GetIOU(Rect_<float> bb_test, Rect_<float> bb_gt)
{
	float in = (bb_test & bb_gt).area();
	float un = bb_test.area() + bb_gt.area() - in;

	if (un < DBL_EPSILON)
		return 0;

	return (double)(in / un);
}

int SortTracking::update(cv::Mat m1, std::vector<DetectBox> input, std::vector<DetectBox> &result)
{
	std::vector<DetectBox> detData;
	KalmanTracker::kf_count = 0;

	if(input.size() <= 0)
	{
		return -1;
	}
	/*Roi tracking filter*/
	for (unsigned int i = 0; i < input.size(); i++)
	{
		/*Convert value to */ 
		int xmin_d = roi_tracking[0];
		int ymin_d = roi_tracking[1];
		int xmax_d = roi_tracking[2];
		int ymax_d = roi_tracking[3];
		if (input[i].bbox.x >= xmin_d && input[i].bbox.y >= ymin_d 
			&& input[i].bbox.x + input[i].bbox.width  < (float) m1.cols - xmax_d
			&& input[i].bbox.y + input[i].bbox.height	< (float) m1.rows- ymax_d) 
		{
			detData.push_back(input[i]);
		}
	}
	// the first frame met
	if (trackers.size() == 0)
	{
		// initialize kalman trackers using first detections.
		for (unsigned int i = 0; i < detData.size(); i++)
		{
			std::vector<DetectBox>::iterator it = detData.begin();
			std::advance(it, i);

			KalmanTracker trk = KalmanTracker((*it).bbox);			
		    trk.class_id = (*it).class_id;
		    trk.class_confidence = (*it).class_confidence;
		    trk.img_face = (*it).img_face.clone();
		    trk.camera_id = (*it).camera_id;
			trk.x_org = (*it).x_org;
			trk.y_org = (*it).y_org;
			(*it).time_in = std::chrono::steady_clock::now();
			trackers.push_back(trk);
		}
		return 0;
	}
	// 3.1. get predicted locations from existing trackers.
	predictedBoxes.clear();
	for (auto it = trackers.begin(); it != trackers.end();)
	{
		Rect_<float> pBox = (*it).predict();
		if (pBox.x >= 0 && pBox.y >= 0)
		{
			predictedBoxes.push_back(pBox);
			it++;
		}
		else
		{
			it = trackers.erase(it);
		}
	}

	// 3.2. associate detections to tracked object (both represented as bounding boxes)
	// dets : detFrameData[fi]
	trkNum = predictedBoxes.size();
	detNum = detData.size();
	iouMatrix.clear();
	iouMatrix.resize(trkNum, vector<double>(detNum, 0));
	for (unsigned int i = 0; i < trkNum; i++) // compute iou matrix as a distance matrix
	{
		for (unsigned int j = 0; j < detNum; j++)
		{
			std::vector<DetectBox>::iterator it = detData.begin();
			std::advance(it, j);
			// use 1-iou because the hungarian algorithm computes a minimum-cost assignment.
			std::vector<cv::Rect_<float>>::iterator it_pred = predictedBoxes.begin(); 
			std::advance(it_pred, i);
			iouMatrix[i][j] = 1 - GetIOU((*it_pred), (*it).bbox);
		}
	}
	// solve the assignment problem using hungarian algorithm.
	// the resulting assignment is [track(prediction) : detection], with len=preNum
	HungarianAlgorithm HungAlgo;
	assignment.clear();
	HungAlgo.Solve(iouMatrix, assignment);
	// find matches, unmatched_detections and unmatched_predictions
	unmatchedTrajectories.clear();
	unmatchedDetections.clear();
	allItems.clear();
	matchedItems.clear();
	if (detNum > trkNum) //	there are unmatched detections
	{
		for (unsigned int n = 0; n < detNum; n++)
			allItems.insert(n);

		for (unsigned int i = 0; i < trkNum; ++i)
			matchedItems.insert(assignment[i]);

		set_difference(allItems.begin(), allItems.end(),
			matchedItems.begin(), matchedItems.end(),
			insert_iterator<set<int>>(unmatchedDetections, unmatchedDetections.begin()));
	}
	else if (detNum < trkNum) // there are unmatched trajectory/predictions
	{
		for (unsigned int i = 0; i < trkNum; ++i)
			if (assignment[i] == -1) // unassigned label will be set as -1 in the assignment algorithm
				unmatchedTrajectories.insert(i);
	}
	else
		;
	// filter out matched with low IOU
	matchedPairs.clear();
	for (unsigned int i = 0; i < trkNum; ++i)
	{
		if (assignment[i] == -1) // pass over invalid values
			continue;
		if (1 - iouMatrix[i][assignment[i]] < iouThreshold)
		{
			unmatchedTrajectories.insert(i);
			unmatchedDetections.insert(assignment[i]);
		}
		else
			matchedPairs.push_back(cv::Point(i, assignment[i]));
	}
	///////////////////////////////////////
	// 3.3. updating trackers
	// update matched trackers with assigned detections.
	// each prediction is corresponding to a tracker
	int detIdx, trkIdx;
	for (unsigned int i = 0; i < matchedPairs.size(); i++)
	{
		trkIdx = matchedPairs[i].x;
		detIdx = matchedPairs[i].y;
		//trackers[trkIdx].m_type = detData[detIdx].type;
		std::vector<DetectBox>::iterator it = detData.begin();
		std::advance(it, detIdx);
		std::vector<KalmanTracker>::iterator it_trackers = trackers.begin();
		std::advance(it_trackers, trkIdx);

		(*it_trackers).update((*it).bbox);
		(*it_trackers).class_id = (*it).class_id;
		(*it_trackers).class_confidence = (*it).class_confidence;
		(*it_trackers).img_face = (*it).img_face.clone();
		(*it_trackers).x_org = (*it).x_org;
		(*it_trackers).y_org = (*it).y_org;
		(*it_trackers).camera_id = (*it).camera_id;
	}
	// create and initialise new trackers for unmatched detections
	for (auto umd : unmatchedDetections)
	{
		std::vector<DetectBox>::iterator it = detData.begin();
		std::advance(it, umd);
		KalmanTracker tracker = KalmanTracker((*it).bbox);

		tracker.class_id = (*it).class_id;
		tracker.class_confidence = (*it).class_confidence;
		tracker.img_face = (*it).img_face.clone();
		tracker.x_org = (*it).x_org;
		tracker.y_org = (*it).y_org;
		tracker.camera_id = (*it).camera_id;
		trackers.push_back(tracker);
	}
	// get trackers' output
	result.clear();
	for (auto it = trackers.begin(); it != trackers.end();)
	{
		if (((*it).m_time_since_update < 1) &&((*it).m_hit_streak >= min_hits || frame_count <= min_hits))
		{
			if ((*it).m_isCounted == false)
		    {
			    (*it).m_id = total_id++;
			    (*it).m_isCounted = true;
		    }
			DetectBox res;

			res.class_id = (*it).class_id;
			res.class_confidence = (*it).class_confidence;
			res.camera_id = (*it).camera_id;
			res.img_face = (*it).img_face.clone();
			res.x_org = (*it).x_org;
			res.y_org = (*it).y_org;
			res.bbox = (*it).get_state();

			// DEBUG_RectFailed(res.bbox);
			res.id_tracking = (*it).m_id + 1;
			auto t2 = std::chrono::steady_clock::now();
			res.live_time = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - res.time_in).count();
			result.push_back(res);
			it++;
		}
		else
			it++;

		// remove dead tracklet
		if (it != trackers.end() && (*it).m_time_since_update > max_age)
			it = trackers.erase(it);
	}
	return 1;
}

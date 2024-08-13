#pragma once
#include<opencv2/core.hpp>
#include<opencv2/features2d.hpp>
#include<vector>

class Image {
	public :	
	Image() = default;
	Image(cv::Mat rgb);

	void compute_keypoints_opencv();
	
	
	std::vector<cv::KeyPoint> get_keypoints(){return keypoints;};
	bool empty(){return rgb.empty();};
	cv::Mat get_descriptors(){return descriptors;};
	cv::Mat get_rgb(){return rgb;};
	cv::Mat get_red_channel(){return red_channel;};
	cv::Mat get_gray(){return keypoint_channel;};
	cv::Mat get_mask(){return mask_;};

	private : 
	cv::Mat rgb;
	cv::Mat mask_;
	cv::Mat red_channel;

	void mask();
	void to_gray(cv::Mat& frame);
        void remove_parts(cv::Mat& frame);	
	void remove_sea(cv::Mat& frame);

	cv::Mat keypoint_channel;
	std::vector<cv::KeyPoint> keypoints;
	cv::Mat descriptors;
	
	void extract_red_channel(){cv::extractChannel(rgb, red_channel, 2);};
		

//	#ifdef OPENCV_SIFT
	cv::Ptr<cv::SiftFeatureDetector> sift_opencv = cv::SiftFeatureDetector::create(0, 3, 0.03, 10, 1.6);
//	#endif

};




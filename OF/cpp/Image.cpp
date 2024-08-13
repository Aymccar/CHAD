#include"Image.h"
#include<vector>
#include<tuple>
#include<opencv2/core.hpp>
#include<opencv2/features2d.hpp>
#include<opencv2/imgproc.hpp>

#include<iostream>
using namespace cv;
using namespace std;

Image::Image(const Mat rgb){
	this->rgb = rgb;
	mask();
	extract_red_channel();
	compute_keypoints_opencv();	
}


void Image::to_gray(Mat& frame){
	cvtColor(frame, frame, COLOR_BGR2GRAY);	
}

void Image::remove_parts(Mat& frame){
	vector<vector<Point>> parts = 
	{
		{Point(110, 0), Point(330, 75)},
	};

	for (vector<Point> shape : parts) {
		rectangle(frame, shape[0], shape[1], 0, -1);
	}	
}

void Image::remove_sea(Mat& frame){
	
}

void Image::mask() {
	mask_ = Mat::ones(rgb.size(), CV_8UC1);	
	remove_parts(mask_); 

	rgb.copyTo(keypoint_channel);
	to_gray(keypoint_channel);
}

void Image::compute_keypoints_opencv(){
	sift_opencv->detectAndCompute(keypoint_channel, mask_, keypoints, descriptors);
}

#pragma once
#include<opencv2/core.hpp>
#include<thread>
#include<vector>
#include<opencv2/features2d.hpp>
#include"Image.h"
#include"VideoStreamWidget.h"
#include"Interface.h"
#include"cpp-httplib/httplib.h"
#include<map>


class OpticalFlowProcessor {
	public :
	OpticalFlowProcessor() = default;
	OpticalFlowProcessor(std::string src, int apiPref, int framerate, int height, int width, std::string fourcc_);
	OpticalFlowProcessor(std::string src);
	
	void start();

	void update_reference();
	void request_reference_update();
	void request_config_update();
	
	private : 
	VideoStreamWidget V;

	std::thread* t_process;

	cv::Mat frame_rgb;
	Image frame;
	Image reference;
	
	std::mutex reference_mut;

	std::vector<float> Vel{3};
	int qual;
	
	void run();
	void process();
	void matching();

	void set_ref();
	
	cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE);
	std::vector<cv::DMatch> matches;
	
	Interface interface{1106, '@'};	
	

	/////////////////// WEB /////////////////////////
	void start_web();
	httplib::Server svr;
	
	void openFile(std::string path, std::string& buffer);
	void paramsToJson(std::string& buffer);
	

	void new_frame_req();
	void config();
	void dashboard();

	std::map<std::string, int> params {
		{"radius", 3},
		{"Vel_scale", 1}, 
		{"norm_val", 1},
		{"dead_zone_xy", 10},
		{"dead_zone_z", 10},
	};
};

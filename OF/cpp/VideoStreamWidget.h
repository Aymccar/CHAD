#pragma once
#include<string>
#include<thread>
#include<opencv2/videoio.hpp>
#include<opencv2/core.hpp>


class VideoStreamWidget {
public :
	VideoStreamWidget()=default;
	VideoStreamWidget(std::string src, int apiPref, int framerate, int height, int width, std::string fourcc);
	VideoStreamWidget(std::string src);

	void start();
	void run();

	int get_frame(cv::Mat &frame);

private :
	cv::Mat frame;
	cv::VideoCapture cap;
	
	int status = 0;
  	bool is_open = false;
	bool is_video = false;	

	std::mutex mut;
	std::thread t;

};

#include"OpticalFlowProcessor.h"
#include"Image.h"
#include"VideoStreamWidget.h"
#include<opencv2/core.hpp>
#include<opencv2/imgproc.hpp>
#include<thread>
#include<chrono>
#include<string>
#include<iostream>
#include<opencv2/highgui.hpp>
#include"Interface.h"
#include"cpp-httplib/httplib.h"
using namespace cv;
using namespace std;


OpticalFlowProcessor::OpticalFlowProcessor(string src,
		int apiPref,
		int framerate, 
		int height,
		int width,
		string fourcc_):
	V{src, apiPref, framerate, height, width, fourcc_}
{

	t_process = new thread(&OpticalFlowProcessor::start, this);
	t_process->detach();

	start_web();
}       

OpticalFlowProcessor::OpticalFlowProcessor(string src):
	V{src}
{
	t_process = new thread(&OpticalFlowProcessor::start, this);
	t_process->detach();

	start_web();
}       

void OpticalFlowProcessor::start_web(){
	auto ret = svr.set_mount_point("/config", "./www/config");	
	new_frame_req();
	config();
	dashboard();
	svr.listen("localhost", 8080);
}

void OpticalFlowProcessor::start(){ //Add in another thread ?
	run();
}

void OpticalFlowProcessor::run(){
	while(1){
		int status = V.get_frame(frame_rgb);
		if (status == 0){
			frame = Image(frame_rgb);

			//add to ref if there isn't one
			if (reference.empty()){
				set_ref();
			}

			process();


		}
		else {
			this_thread::sleep_for(chrono::milliseconds(100));
			continue;}
	}
}

void OpticalFlowProcessor::set_ref() {

	lock_guard<mutex> guard(reference_mut);
	reference = frame;
	cout<<"New ref"<<endl;
}

void OpticalFlowProcessor::matching() {
	matches.clear();
	vector<vector<DMatch>> temp_matches;

	if ((reference.get_descriptors().empty()) or (frame.get_descriptors().empty())) {
		return;
	}
	matcher->knnMatch(reference.get_descriptors(), frame.get_descriptors(), temp_matches, 2);

	for (int i=0; i<temp_matches.size(); i++) {
		if (temp_matches[i][0].distance <  0.75 * temp_matches[i][1].distance){
			matches.push_back(temp_matches[i][0]);
		}	
	}

}

void OpticalFlowProcessor::process() {

	matching();	

	if (matches.size() == 0) {
		imshow("live", frame.get_gray());
		return;
	}

	Mat frame_plot = frame.get_gray();

	addWeighted(frame_plot, 1, 255*(1-frame.get_mask()), -0.3, 0, frame_plot);	

	cvtColor(frame_plot, frame_plot,COLOR_GRAY2BGR);



	vector<KeyPoint> keypoint_ref = reference.get_keypoints();
	vector<KeyPoint> keypoint_frame = frame.get_keypoints();

	vector<KeyPoint> matched_keypoint_ref(matches.size());
	vector<KeyPoint> matched_keypoint_frame(matches.size());

	for (int i=0; i<matches.size(); i++) {	
		matched_keypoint_ref[i] = keypoint_ref[matches[i].queryIdx];
		matched_keypoint_frame[i] = keypoint_frame[matches[i].trainIdx];
	}

	//calculation on Z relative displacement with red component
	Mat red_channel_frame;
	frame.get_red_channel().convertTo(red_channel_frame, CV_32FC1);

	Mat red_channel_ref;
	reference.get_red_channel().convertTo(red_channel_ref, CV_32FC1);

	vector<float> x_frame(matched_keypoint_frame.size());
	vector<float> y_frame(matched_keypoint_frame.size());

	vector<float> x_ref(matched_keypoint_ref.size());
	vector<float> y_ref(matched_keypoint_ref.size());


	vector<float> mean_loc;
	for (int i=0; i<matched_keypoint_frame.size(); i++){

		Point2f point_f = matched_keypoint_frame[i].pt;
		Point2f point_r = matched_keypoint_ref[i].pt;

		x_ref[i] = point_r.x;
		y_ref[i] = point_r.y;

		x_frame[i] = point_f.x;
		y_frame[i] = point_f.y;


		if(!((abs(red_channel_frame.size[1] - (point_f.x+5))<params["radius"]) or
					(abs(red_channel_frame.size[0] - (point_f.y+5))<params["radius"]) or
					(abs(red_channel_ref.size[1] - (point_r.x+5))<params["radius"]) or
					(abs(red_channel_ref.size[0] - (point_r.y+5))<params["radius"])
					or
					(abs(0 - (point_f.x-5))<params["radius"]) or
					(abs(0 - (point_f.y-5))<params["radius"]) or
					(abs(0 - (point_r.x-5))<params["radius"]) or
					(abs(0 - (point_r.y-5))<params["radius"]))){

			int size_loc = 0;
			float sum_loc = 0;

			for (int i = -params["radius"]; i<params["radius"]; i++){
				for (int j = -params["radius"]; j<params["radius"]; j++){
					sum_loc += red_channel_frame.at<float>(point_f.y+i, point_f.x+j) - red_channel_ref.at<float>(point_r.y+i, point_r.x+j); 
					size_loc += 1;

					//Plot		
					Vec3b& color = frame_plot.at<Vec3b>(point_f.y+i, point_f.x+j);
					color = Vec3b{255, 255, 204}; 
				}
			}
			mean_loc.push_back(sum_loc/size_loc);
		}
	}

	sort(mean_loc.begin(), mean_loc.end());
	if (mean_loc.size()%2==0){Vel[2]=(mean_loc[mean_loc.size()/2]+mean_loc[mean_loc.size()/2+1])/2;}
	else {Vel[2]=mean_loc[mean_loc.size()/2];}

	//Calculation along X and Y axis
	vector<float> x(x_frame.size());
	vector<float> y(y_frame.size());

	for (int i=0; i < x_frame.size(); i++) {

		x[i] = x_frame[i] - x_ref[i];
		y[i] = y_frame[i] - y_ref[i];
	}

	sort(x.begin(), x.end());
	if (x.size()%2==0){Vel[0] = (x[(x.size()/2)]+x[(x.size()/2)+1])/2;}
	else {Vel[0] = x[x.size()/2];}

	sort(y.begin(), y.end());
	if (y.size()%2==0){Vel[1] = (y[(y.size()/2)]+y[(y.size()/2)+1])/2;}
	else {Vel[1] = y[y.size()/2];}

	int n = matches.size();
	//Quality is just the number on sift point scaled between 0 and 255

	qual = std::min(255, std::max(0, n/params["norm_val"]));


	for (int i=0; i<3; i++){
		Vel[i] = Vel[i]/params["Vel_scale"];
	}
	
	
	if (abs(Vel[0])<params["dead_zone_xy"]/10) {Vel[0] = 0;}
	if (abs(Vel[1])<params["dead_zone_xy"]/10) {Vel[1] = 0;}
	if (abs(Vel[2])<params["dead_zone_z"]/10) {Vel[2] = 0;}

	cout<<"Velocity : "<<Vel[0]<<", "<<Vel[1]<<", "<<Vel[2]<<" Quality : "<<qual<<endl;

	interface.send(Vel, qual);

	//////////////////////////////////////////////////////////
	//			PLOT				//
	//////////////////////////////////////////////////////////


	drawKeypoints(frame_plot, frame.get_keypoints(), frame_plot, Scalar(0, 127, 0));

	Point org(frame_plot.size[1]/2, frame_plot.size[0]/2);
	Point arrow(frame_plot.size[1]/2 + Vel[0], frame_plot.size[0]/2 + Vel[1]);

	arrowedLine(frame_plot, org, arrow, Scalar(255, 0, 0), 5);	

	Point org_text(frame_plot.size[1]/2 - 10, frame_plot.size[0]/2);

	putText(frame_plot, to_string(Vel[2]), org_text, FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255),  2);

	imshow("live", frame_plot);
	waitKey(5);
}

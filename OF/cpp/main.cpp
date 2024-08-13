#include"OpticalFlowProcessor.h"
#include<opencv2/core.hpp>

int main() {
    //OpticalFlowProcessor ofp("/dev/video0", cv::CAP_V4L2, 30, 640, 480, "YUYV");
    OpticalFlowProcessor("video_test.mp4");
    
	return 0;
}

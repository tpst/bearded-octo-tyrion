#include <opencv\cv.h>
#include <opencv\highgui.h>
#include "opencv2/gpu/gpu.hpp"

#include "objectFinder.h"

using namespace cv;
using namespace std;
using namespace cv::gpu;

int main(int argc, char** argv)
{
	cv::VideoCapture cap;
	cap.open(0);
	if (!cap.isOpened())
	{
		cout << "Cannot open webcam feed" << endl;
		return -1;
	}
	featureMatch matcher;
	
	//Mat frame = imread("captured.jpg");
	cv::gpu::GpuMat frame,object1;
	cv::Mat frame_temp;

	object1.upload(imread("MotionTracker.jpg", CV_LOAD_IMAGE_GRAYSCALE));

	while(1) 
	{
		cap.read(frame_temp);
		

		matcher.match(frame_temp);

	}
	
	return 0;
}
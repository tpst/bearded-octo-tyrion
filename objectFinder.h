#ifndef FEATUREMATCH_H
#define FEATUREMATCH_H

#include "opencv2/nonfree/gpu.hpp"

class featureMatch {

	public: 
		int minHessian;
		float ratio;
		double confidence;
		double distance1;

		cv::gpu::SURF_GPU detector;//works
		cv::Ptr<cv::gpu::BFMatcher_GPU> matcher;

		cv::gpu::GpuMat object;
		cv::gpu::GpuMat descriptors_object_gpu, descriptors_scene_gpu;
		cv::gpu::GpuMat keypoints_object_gpu, keypoints_scene_gpu;
		cv::gpu::GpuMat trainId, distance;

		std::vector< float > descriptors_object, descriptors_scene;
		//std::vector< cv::DMatch > matches;
		std::vector< std::vector< cv::DMatch >> matches1;
		std::vector< std::vector< cv::DMatch >> matches2;

	    std::vector< cv::KeyPoint > keypoints_object, keypoints_scene;
		std::vector<cv::Point2f> obj_corners;

		//match object to scene
		int match(cv::Mat& frame);

		 // Clear matches for which NN ratio is > than threshold
		 // return the number of removed points
		 // (corresponding entries being cleared,
		 // i.e. size will be 0)
		int ratioTest(std::vector<std::vector<cv::DMatch>> &matches);
		
		void symmetryTest(const std::vector<std::vector<cv::DMatch> > &matches1,
						  const std::vector<std::vector<cv::DMatch> > &matches2,
						  std::vector<cv::DMatch>& symMatches);

		cv::Mat ransacTest(const std::vector<cv::DMatch>& matches,
					   const std::vector<cv::KeyPoint>& keypoints1,
					   const std::vector<cv::KeyPoint>& keypoints2,
					   std::vector<cv::DMatch>& outMatches);

		featureMatch();


};

#endif
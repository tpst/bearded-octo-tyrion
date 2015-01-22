#include <opencv\cv.h>
#include <opencv\highgui.h>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/gpu.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/gpu/gpu.hpp"

#include "objectFinder.h"

using namespace cv;
using namespace std;
using namespace cv::gpu;

featureMatch::featureMatch(void)
: obj_corners(4), ratio(0.65f), confidence(0.5), distance1(1.5) //declare size of vector
{
	cout << "Matcher created" << endl;
	minHessian = 800;
	
	object.upload(imread("MotionTracker.jpg", CV_LOAD_IMAGE_GRAYSCALE));

	matcher = new BFMatcher_GPU(NORM_L2);
	detector = SURF_GPU(minHessian); //works

	detector(object, GpuMat(), keypoints_object_gpu, descriptors_object_gpu);
	//download results
	
	detector.downloadKeypoints(keypoints_object_gpu, keypoints_object);
	detector.downloadDescriptors(descriptors_object_gpu, descriptors_object);
	
	cout << "FOUND " << keypoints_object_gpu.cols << " keypoints on object (gpu)" << endl;
	cout << "FOUND " << descriptors_object_gpu.cols << " descriptors on object (gpu)" << endl;

    //Get the corners from the object
    obj_corners[0] = cvPoint(0,0);
    obj_corners[1] = cvPoint( object.cols, 0 );
    obj_corners[2] = cvPoint( object.cols, object.rows );
    obj_corners[3] = cvPoint( 0, object.rows );
	
	//this all works
}

int featureMatch::ratioTest(std::vector<std::vector<cv::DMatch> > &matches)
{
    int removed=0;
      // for all matches
    for (std::vector<std::vector<cv::DMatch> >::iterator
             matchIterator= matches.begin();
         matchIterator!= matches.end(); ++matchIterator) {
           // if 2 NN has been identified
           if (matchIterator->size() > 1) {
               // check distance ratio
               if ((*matchIterator)[0].distance/
                   (*matchIterator)[1].distance > ratio) {
                  matchIterator->clear(); // remove match
                  removed++;
               }
           } else { // does not have 2 neighbours
               matchIterator->clear(); // remove match
               removed++;
           }
    }
    return removed;
  }

// Insert symmetrical matches in symMatches vector
  void featureMatch::symmetryTest(
      const std::vector<std::vector<cv::DMatch> >& matches1,
      const std::vector<std::vector<cv::DMatch> >& matches2,
      std::vector<cv::DMatch>& symMatches) {
    // for all matches image 1 -> image 2
    for (std::vector<std::vector<cv::DMatch> >::
             const_iterator matchIterator1= matches1.begin();
         matchIterator1!= matches1.end(); ++matchIterator1) {
       // ignore deleted matches
       if (matchIterator1->size() < 2)
           continue;
       // for all matches image 2 -> image 1
       for (std::vector<std::vector<cv::DMatch> >::
          const_iterator matchIterator2= matches2.begin();
           matchIterator2!= matches2.end();
           ++matchIterator2) {
           // ignore deleted matches
           if (matchIterator2->size() < 2)
              continue;
           // Match symmetry test
           if ((*matchIterator1)[0].queryIdx ==
               (*matchIterator2)[0].trainIdx &&
               (*matchIterator2)[0].queryIdx ==
               (*matchIterator1)[0].trainIdx) {
               // add symmetrical match
                 symMatches.push_back(
                   cv::DMatch((*matchIterator1)[0].queryIdx,
                             (*matchIterator1)[0].trainIdx,
                             (*matchIterator1)[0].distance));
                 break; // next match in image 1 -> image 2
           }
       }
    }
  }

// Identify good matches using RANSAC
  // Return fundemental matrix
cv::Mat featureMatch::ransacTest(
      const std::vector<cv::DMatch>& matches,
      const std::vector<cv::KeyPoint>& keypoints1,
      const std::vector<cv::KeyPoint>& keypoints2,
      std::vector<cv::DMatch>& outMatches) 
  {
		// Convert keypoints into Point2f
		 std::vector<cv::Point2f> points1, points2;
		 cv::Mat fundemental;
		for (std::vector<cv::DMatch>::
         const_iterator it= matches.begin();
			it!= matches.end(); ++it) 
		{
		   // Get the position of left keypoints
		   float x= keypoints1[it->queryIdx].pt.x;
		   float y= keypoints1[it->queryIdx].pt.y;
		   points1.push_back(cv::Point2f(x,y));
		   // Get the position of right keypoints
		   x= keypoints2[it->trainIdx].pt.x;
		   y= keypoints2[it->trainIdx].pt.y;
		   points2.push_back(cv::Point2f(x,y));
		}
   // Compute F matrix using RANSAC
   std::vector<uchar> inliers(points1.size(),0);
   if (points1.size()>0&&points2.size()>0)
   {
      cv::Mat fundemental= cv::findFundamentalMat( cv::Mat(points1), cv::Mat(points2), inliers, CV_FM_RANSAC, distance1, confidence); // confidence probability
      // extract the surviving (inliers) matches
      std::vector<uchar>::const_iterator
                         itIn= inliers.begin();
      std::vector<cv::DMatch>::const_iterator
                         itM= matches.begin();
      // for all matches
      for ( ;itIn!= inliers.end(); ++itIn, ++itM) {
         if (*itIn) { // it is a valid match
             outMatches.push_back(*itM);
          }
       }
    }
   return fundemental;
  }

int featureMatch::match(cv::Mat& frame_temp) 
{
	cv::gpu::GpuMat frame,output_gpu;
	frame.upload(frame_temp);
	cv::gpu::cvtColor(frame,output_gpu,CV_BGR2GRAY);

	detector(output_gpu, GpuMat(), keypoints_scene_gpu, descriptors_scene_gpu);
	matcher->knnMatch( descriptors_object_gpu, descriptors_scene_gpu, matches1, 2, GpuMat());
	matcher->knnMatch( descriptors_scene_gpu, descriptors_object_gpu, matches2, 2, GpuMat());
	
	//matcher->match( descriptors_object_gpu, descriptors_scene_gpu, matches, GpuMat());

	detector.downloadKeypoints(keypoints_scene_gpu, keypoints_scene);
	detector.downloadDescriptors(descriptors_scene_gpu, descriptors_scene);

	std::vector< cv::DMatch > symMatches;
	std::vector< cv::DMatch > matches;


	// Remove matches for which NN ratio is > than threshold
	int removed = ratioTest(matches1);
	removed = ratioTest(matches2);

	//Remove non-symmetrical matches
	symmetryTest(matches1,matches2,symMatches);

	cv::Mat fundamental = ransacTest(symMatches, keypoints_object, keypoints_scene, matches);


	//-- Show detected matches

	Mat img_matches;

	drawMatches(Mat(object), keypoints_object, Mat(output_gpu), keypoints_scene, matches, img_matches, Scalar::all(-1), Scalar::all(-1),
               vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	imshow( "Good Matches & Object detection", img_matches );

	if(matches.size() > 5 )
	{
		//-- Localize the object
	  std::vector<Point2f> obj;
	  std::vector<Point2f> scene;

	  for( int i = 0; i < matches.size(); i++ )
	  {
		//-- Get the keypoints from the good matches
		obj.push_back( keypoints_object[ matches[i].queryIdx ].pt );
		scene.push_back( keypoints_scene[ matches[i].trainIdx ].pt );
	  }

	  Mat H = findHomography( obj, scene, CV_RANSAC );

	  std::vector<Point2f> scene_corners(4);

	  perspectiveTransform( obj_corners, scene_corners, H);

	  //-- Draw lines between the corners (the mapped object in the scene - image_2 )
	  line( img_matches, scene_corners[0] + Point2f( object.cols, 0), scene_corners[1] + Point2f( object.cols, 0), Scalar(0, 255, 0), 4 );
	  line( img_matches, scene_corners[1] + Point2f( object.cols, 0), scene_corners[2] + Point2f( object.cols, 0), Scalar( 0, 255, 0), 4 );
	  line( img_matches, scene_corners[2] + Point2f( object.cols, 0), scene_corners[3] + Point2f( object.cols, 0), Scalar( 0, 255, 0), 4 );
	  line( img_matches, scene_corners[3] + Point2f( object.cols, 0), scene_corners[0] + Point2f( object.cols, 0), Scalar( 0, 255, 0), 4 );

	  //-- Show detected matches
	  imshow( "Good Matches & Object detection", img_matches );

	}

	waitKey(1);
	return 1; 
}


	
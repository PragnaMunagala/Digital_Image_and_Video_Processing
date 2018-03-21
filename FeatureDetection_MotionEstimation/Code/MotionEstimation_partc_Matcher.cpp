#include <stdio.h>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

/* Main Function */
int main()
{
	VideoCapture cap("C:/Users/munag/Desktop/Dabba.mp4"); // open the video input

	if (!cap.isOpened())  // if input open success
		return -1;

	double dWidth = cap.get(CV_CAP_PROP_FRAME_WIDTH); //width of frames of input
	double dHeight = cap.get(CV_CAP_PROP_FRAME_HEIGHT); //height of frames of input
	Size frameSize(static_cast<int>(dWidth), static_cast<int>(dHeight));  //frame size

	VideoWriter output("C:/Users/munag/Desktop/MatcherOp.avi", CV_FOURCC('P', 'I', 'M', '1'), 20, frameSize, true);   //output video
	if (!output.isOpened()) //if initialization not successful
	{
		cout << "ERROR: Failed to write the video" << endl;
		return -1;
	}
	int i = 0;
	Mat frame;
	Mat canvas(dHeight, dWidth, CV_8UC3, Scalar(0, 0, 0));   //to draw the output
	vector<KeyPoint> currKeypoints, prevKeypoints;
	Mat currDescriptors, prevDescriptors, prevFrame;
	vector<Point2f> pt1, pt2;
		
	//Ptr<SURF> detector = SURF::create(500);   /******* SURF detector ******/
	Ptr<SIFT> detector = SIFT::create(500);     /******* SIFT detector ******/	

	while (1) {
		clock_t begin = clock();    //start time

		bool success = cap.read(frame);   //check for frame empty or not
		if (!success)
			break;

		/* Keypoints and descriptors for each frame and drawing keypoints */
		cap >> frame;
		detector->detect(frame, currKeypoints);
		Mat img_keypoints_1;
		drawKeypoints(frame, currKeypoints, img_keypoints_1, Scalar::all(-1),
			DrawMatchesFlags::DEFAULT);
		if (currKeypoints.empty())
			continue;
		detector->compute(frame, currKeypoints, currDescriptors);

		/* For first frame */
		if (i == 0) {
			prevDescriptors = currDescriptors;
			prevKeypoints = currKeypoints;
			prevFrame = frame;
		}

		/* Converting Keypoints to Points format */
		KeyPoint::convert(prevKeypoints, pt1); 
		KeyPoint::convert(currKeypoints, pt2);

		/* Flann matcher to match the descriptors within consecutive frames */
		FlannBasedMatcher matcher;
		vector< DMatch > matches;
		matcher.match(prevDescriptors, currDescriptors, matches);

		/* To find the good matches */
		double max_dist = 0; double min_dist = 100;
		//max and min distances between keypoints
		for (int j = 0; j < prevDescriptors.rows; j++)
		{
			double dist = matches[j].distance;
			if (dist < min_dist) min_dist = dist;
			if (dist > max_dist) max_dist = dist;
		}
		//"good" matches
		vector< DMatch > good_matches;
		for (int j = 0; j < prevDescriptors.rows; j++)
		{
			if (matches[j].distance < 3 * min_dist)
			{
				good_matches.push_back(matches[j]);
			}
		}

		/* Used four colors to draw the good matches to distingush */
		for (int j = 0; j < (int)good_matches.size(); j++)
		{
			if (j % 4 == 0)
				line(canvas, pt1[good_matches[j].queryIdx], pt2[good_matches[j].trainIdx], Scalar(0, 255, 0), 2, -1);
			else if (j % 4 == 1)
				line(canvas, pt1[good_matches[j].queryIdx], pt2[good_matches[j].trainIdx], Scalar(255, 0, 0), 2, -1);
			else if (j % 4 == 2)
				line(canvas, pt1[good_matches[j].queryIdx], pt2[good_matches[j].trainIdx], Scalar(0, 0, 255), 2, -1);
			else
				line(canvas, pt1[good_matches[j].queryIdx], pt2[good_matches[j].trainIdx], Scalar(255, 255, 255), 2, -1);
		}
		
		addWeighted(frame, 0.8, canvas, 0.3, 0, frame);		//adding onto the canvas
		output.write(frame);	//writing to output file

		/* Swaping data for next iteration */
		swap(prevDescriptors, currDescriptors);
		swap(prevFrame, frame);
		swap(prevKeypoints, currKeypoints);

		imshow("Output", frame);
		clock_t end = clock(); //End time
		double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;   //elapsed time for each iteration
		cout << "Execution time for one frame = " << elapsed_secs << " second(s).\n" << endl;
		waitKey(10);
		i++;
	}
	waitKey(0);
	return 0;
}
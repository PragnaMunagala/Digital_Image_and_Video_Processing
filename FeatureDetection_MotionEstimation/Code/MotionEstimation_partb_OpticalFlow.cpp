#include <stdio.h>
#include <iostream>
#include "opencv2/core.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
int main()
{
	VideoCapture cap("C:/Users/munag/Desktop/Dabba.mp4"); //Location of the input video
	if (!cap.isOpened())  // check if we succeeded
		return -1;

	double dWidth = cap.get(CV_CAP_PROP_FRAME_WIDTH); //get the width of frames of the video
	double dHeight = cap.get(CV_CAP_PROP_FRAME_HEIGHT); //get the height of frames of the video
	Size frameSize(static_cast<int>(dWidth), static_cast<int>(dHeight));
	VideoWriter output("C:/Users/munag/Desktop/OpticalFlowOut.avi", CV_FOURCC('P', 'I', 'M', '1'), 20, frameSize, true); // Location of the output video
	if (!output.isOpened()) //if not initialize the VideoWriter successfully, exit the program
	{
		cout << "ERROR: Failed to write the video" << endl;
		return -1;
	}

	int i = 0;
	Mat frame;
	Mat canvas(dHeight, dWidth, CV_8UC3, Scalar(0, 0, 0)); // cavas black image for drawing the tracking line

	// For SURF detector uncomment the below line
	Ptr<SURF> detector = SURF::create(500);

	// For SIFT detector uncomment the below line
	//Ptr<SIFT> detector = SIFT::create(500);

	std::vector<KeyPoint> currKeypoints;
	Mat prevFrame;
	vector<Point2f> pt1, pt2;
	int win_size = 20;

	while (1)
	{
		clock_t begin = clock();    //start time

		bool success = cap.read(frame);
		if (!success)
			break;
		cap >> frame; // reading the video frame

		// For the first frame, there is no previous frame. Hence assigning current frame to previous frame
		// The keypoints are detected for the first frame
		if (i == 0) {
			detector->detect(frame, currKeypoints);
			KeyPoint::convert(currKeypoints, pt1);
			frame.copyTo(prevFrame);
			i++;
			continue;
		}

		vector<uchar> features_found;
		//OpticalFlow is calculated for the previous frame to current frame and are stored in pt2.
		calcOpticalFlowPyrLK(prevFrame, frame, pt1, pt2, features_found, noArray(), Size(win_size * 2 + 1, win_size * 2 + 1), 5,
			TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, .3));

		// loop for drawing the tracking points in different colors. The lines are drawn on the canvas image
		for (int j = 0; j < (int)pt1.size(); j++) {
			if (!features_found[j])
				continue;
			if (j % 4 == 0)
				line(canvas, pt1[j], pt2[j], Scalar(255, 0, 0), 2, -1);
			else if (j % 4 == 1)
				line(canvas, pt1[j], pt2[j], Scalar(0, 255, 0), 2, -1);
			else if (j % 4 == 2)
				line(canvas, pt1[j], pt2[j], Scalar(0, 0, 255), 2, -1);
			else
				line(canvas, pt1[j], pt2[j], Scalar(255, 255, 255), 2, -1);
		}
		// The current frame and the canvas image are blended into the current frame
		addWeighted(frame, 0.8, canvas, 0.3, 0, frame);
		output.write(frame); // writing the frame into the file
		pt1 = pt2; // assigning the current points as previous points for the next iteration
		pt2.clear();
		frame.copyTo(prevFrame); // coping current frame to previous frame for the next iteration
		imshow("OpticalFlow", frame); // showing the tracking frame
		clock_t end = clock(); //End time
		double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;   //elapsed time for each iteration
		cout << "Execution time for one frame = " << elapsed_secs << " second(s).\n" << endl;

		waitKey(10);
		i++;
	}
	waitKey(0);
	return 0;
}
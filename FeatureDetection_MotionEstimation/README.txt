Project 1
Group 7


############## Introduction ##############
In this project, SIFT and SURF feature detectors are used to detect the keypoints and descriptors of the input video
and Motion Estimation is done using two approaches:
1. Using SIFT/SURF feature detectors and KLT Optical Flow
2. Using SIFT/SURF feature descriptors and FLANN Matcher
We used two different inputs Dabba.mp4 and Smiley.mp4 present in Input folder. Outputs corresponding to each of the above approach for each input 
can be viewed in Output folder.


############## Steps to Execute ##############
1. Install opencv as mentioned in install opencv instructions document.
2. Setup an opencv project settings as mentioned in opencv setup and make sure the Debug and x64 options are selected.
3. Download and extract the code files from Code folder and Input vides from Input folder.

For Part(b) - Optical Flow
	4. Download the MotionEstimation_partb_OpticalFlow.cpp file from the submitted zipped folder.
	5. Open the attached MotionEstimation_partb_OpticalFlow.cpp file in the project created in Microsoft visual studio.

	6. Change the path of the input to be loaded in line 15 with the path where the input videos are downloaded 
		Ex: Folder location needs to be updated in
			VideoCapture cap("C:/Users/munag/Desktop/Dabba.mp4");

	7. Change the path of the output to be saved in line 22 to the required destination location on your PC.
		Ex: Folder location needs to be updated in
			VideoWriter output("C:/Users/munag/Desktop/OpticalFlowOut.avi", CV_FOURCC('P', 'I', 'M', '1'), 20, frameSize, true);

	8. By default, SURF descriptor is run, for testing with SIFT descriptor, comment line 34 and uncomment line 37.

	9. Run the code using the Windows Debugger visible in visual studio or press F5.
	10. The output with SURF/SIFT and optical flow can be seen and the output will be saved in the mentioned destination.

For part(c) - Feature descriptors and FLANN Matcher
	4. Download the MotionEstimation_partc_Matcher.cpp file from the submitted zipped folder.
	5. Open the attached MotionEstimation_partc_Matcher.cpp file in the project created in Microsoft visual studio.

	6. Change the path of the input to be loaded in line 16 with the path where the input videos are downloaded 
		Ex: Folder location needs to be updated in
			VideoCapture cap("C:/Users/munag/Desktop/Dabba.mp4");

	7. Change the path of the output to be saved in line 25 to the required destination location on your PC.
		Ex: Folder location needs to be updated in
			VideoWriter output("C:/Users/munag/Desktop/MatcherOp.avi", CV_FOURCC('P', 'I', 'M', '1'), 20, frameSize, true);

	8. By default, SIFT descriptor is run, for testing with SURF descriptor, comment line 39 and uncomment line 38.

	9. Run the code using the Windows Debugger visible in visual studio or press F5.
	10. The keypoints in each frame can be seen.
	11. The motion flow with SURF/SIFT and FLANN Matcher can be seen and the output will be saved in the mentioned destination.


############## Output ##############
Each frame data can be seen and the output video showing the motion estimation will be saved in the mentioned destination.


############## System Used ##############
Microsoft Windows 10
Visual Studio 2015


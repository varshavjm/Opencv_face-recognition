#include<stdio.h>
#include<math.h>
#include<opencv\cv.h>
#include<opencv\highgui.h>
#include<opencv2\objdetect\objdetect.hpp>
#include<opencv2\highgui\highgui.hpp>
#include<opencv2\imgproc\imgproc.hpp>
#include<opencv2/contrib/contrib.hpp>
#include<vector>
#include<fstream>
#include<iostream>
#include<conio.h>
using namespace cv;
using namespace std;

int main(int argc,char*argv[])
{
	
	CascadeClassifier face_cascade,eye_cascade;
	std::vector<Mat> images;
	std::vector<int> labels;
	if(!face_cascade.load("C:\\harr\\haarcascade_frontalface_alt2.xml")){
		printf("error loading cascade file for face");
		return 1;
	}

	if(!eye_cascade.load("C:\\harr\\haarcascade_eye.xml")){
		printf("error loading cascade file for eye");
		return 1;
	}
Ptr<FaceRecognizer> model = createEigenFaceRecognizer(25,10000);
/*	if (model.empty()) {
        cerr << "ERROR: The FaceRecognizer algorithm  is not available in your version of OpenCV. Please update to OpenCV v2.4.1 or newer." << endl;
        exit(1);
    }*/
    char ch;
    model->load("fishertrain.yml");
    if(1){
	Mat check_img,grey_img;
	vector<Rect>faces;
	
	/*
	Provide correct image path in variable filename
	Remember its label value stored while training the data with video
	*/
	//char filename[]="3.jpg";
	char filename[40];
	strcpy(filename,argv[1]);
	check_img = imread(filename, CV_LOAD_IMAGE_COLOR);// Read the file
	int ok;
	
	//waitKey(20);
	//	imshow("test",check_img);
		cvtColor(check_img , grey_img,CV_BGR2GRAY);
		cv::equalizeHist(grey_img,grey_img);
		face_cascade.detectMultiScale(grey_img,faces,1.1,6,CV_HAAR_DO_CANNY_PRUNING|CV_HAAR_SCALE_IMAGE,Size(0,0),Size(300,300));
		Mat faceROI;
		ok=0;
		for(int i=0;i<faces.size();i++){
			Point pt1(faces[i].x + faces[i].width,faces[i].y + faces[i].height);
			Point pt2(faces[i].x,faces[i].y);
			faceROI = grey_img(faces[i]);
			//faceROI = check_img(faces[i]);
			ok = 1;
		}
		if(ok==1){
			check_img = faceROI.clone();
			cv::resize(check_img,check_img,Size(320,200),0,0,1);
		//	imshow("test",check_img);
		//	waitKey(3);
		//	imshow("Result",check_img);
	 
	 int predictlabel;
	 double confidence;
	//	cv::resize(check_img,check_img,Size(320,200),0,0,1);
	//	cvtColor(check_img , grey_img,CV_BGR2GRAY);
		

    model->predict(check_img,predictlabel,confidence);
	int threshhold=model->getDouble("threshold");
    
	//This is the stored label of correspoding video 
    string result_message = format("Predicted class = %d, confidence val: %lf actual class=%s, threshold=%d", predictlabel,confidence,filename,threshhold);
    cout<< result_message << endl;
		}
	//waitKey(100);
	}
	waitKey(0);  // Wait for a keystroke in the window
	getch();

	return 0;
	}
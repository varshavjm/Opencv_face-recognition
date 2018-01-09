#include<stdio.h>
#include<math.h>
#include<opencv\cv.h>
#include<conio.h>
#include<opencv\highgui.h>
#include<opencv2\objdetect\objdetect.hpp>
#include<opencv2\highgui\highgui.hpp>
#include<opencv2\imgproc\imgproc.hpp>
#include<opencv2/contrib/contrib.hpp>
#include<vector>
#include<fstream>
#include<iostream>
#include <vld.h>
using namespace cv;
using namespace std;


static Mat norm_0_255(InputArray _src) {
	Mat src = _src.getMat();
	// Create and return normalized image:
	Mat dst;
	switch(src.channels()) {
	case 1:
		cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
		break;
	case 3:
		cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
		break;
	default:
		src.copyTo(dst);
		break;
	}
	return dst;
}
int main(int argc, char *argv[])
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
	Mat cap_img , grey_img , old_img;
	std::vector<Rect> faces,eyes;
	int count = 1;
	int j=0;
	int go=0;
	int label_cnt=1;

	count=atoi(argv[1]);

int i=0;
		char videoname[30],str[30];
		Mat cpy_img,temp_img,faceROI;
		Mat mirroredFace;
	
		while(count){
		cout<<"\nI am working on video "<<label_cnt<<endl;
		strcpy(videoname,"");
		itoa(label_cnt,videoname,10);
		strcat(videoname,".avi");
		VideoCapture cap(videoname);
		cout<<cap.isOpened();
		if (!cap.isOpened())
		{
			std::cout << "!!! Failed to open file: "<< std::endl;
			getch();
			return -1;
		}
		int no_frames=0;
		while(1){
			no_frames++;
			if (!cap.read(cap_img))             
				break;
			
			waitKey(10);
		
			if(no_frames%20!=0)
				continue;
			 cpy_img = cap_img.clone(); 
			try{
				cvtColor(cap_img, grey_img,CV_BGR2GRAY);
			}
			catch(...){
			
				cout<<"hello ";
			}
			if(go==0)
			cv::equalizeHist(grey_img,grey_img);
			face_cascade.detectMultiScale(grey_img,faces,1.1,6,CV_HAAR_DO_CANNY_PRUNING|CV_HAAR_SCALE_IMAGE,Size(0,0),Size(300,300));
			for(int i=0;i<faces.size();i++){
				 Point pt1(faces[i].x + faces[i].width,faces[i].y + faces[i].height);
				 Point pt2(faces[i].x,faces[i].y);
				faceROI = grey_img(faces[i]);
				cv::resize(faceROI,faceROI,Size(320,200),0,0,1);
			//	cv::equalizeHist(faceROI,faceROI);
				flip(faceROI, mirroredFace, 1);
				images.push_back(mirroredFace);
				images.push_back(faceROI);
				imshow("result",faceROI);	
				labels.push_back(label_cnt);
				labels.push_back(label_cnt);
				old_img = cpy_img;
				cout<<"stored image no "<<no_frames/20<<endl;
					
			}
			
			}
		label_cnt++;
		cap.release();
		count--;
	}
	cout<<"\nbefore creating recognizer";
	Ptr<FaceRecognizer> model = createEigenFaceRecognizer(21);
		if (model.empty()) {
	cerr << "ERROR: The FaceRecognizer algorithm  is not available in your version of OpenCV. Please update to OpenCV v2.4.1 or newer." << endl;
	exit(1);
	}
	cout<<"\nAfter creating recognizer";
	model->train(images, labels);
	model->save("eigentrain.yml");
	cout<<"\nTrained!! Press a key";
	return 0;
}

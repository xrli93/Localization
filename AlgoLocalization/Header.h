#pragma once
//void SIFT_KMeans()
//{
//	Mat labels;
//	Mat centers;
//
//	Mat img_1;
//	img_1 = imread("C:\\Users\\Hoth\\Downloads\\opencv-logo.png", IMREAD_COLOR); // Read the file
//	const cv::Mat input = cv::imread("input.jpg", 0); //Load as grayscale
//
//	cv::Ptr<Feature2D> f2d = xfeatures2d::SIFT::create();
//	//cv::Ptr<Feature2D> f2d = xfeatures2d::SURF::create();
//	//cv::Ptr<Feature2D> f2d = ORB::create();
//	// you get the picture, i hope..
//
//	//-- Step 1: Detect the keypoints:
//	std::vector<KeyPoint> keypoints_1, keypoints_2;
//	f2d->detect(img_1, keypoints_1);
//
//	//-- Step 2: Calculate descriptors (feature vectors)    
//	Mat descriptors_1, descriptors_2;
//	f2d->compute(img_1, keypoints_1, descriptors_1);
//	cout << descriptors_1;
//	descriptors_1.convertTo(descriptors_1, CV_32FC2);
//	//cv::kmeans(keypoints_1, 3, labels, cv::TermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 10, 0.1), 3, cv::KMEANS_PP_CENTERS, centers);
//	cv::kmeans(descriptors_1, 3, labels, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0), 3, cv::KMEANS_PP_CENTERS, centers);
//	cout << "end";
//	cin.get();
//
//}

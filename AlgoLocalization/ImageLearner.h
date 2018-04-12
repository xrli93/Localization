#pragma once
#include"TreeDict.h"
#define THRESHOLD_FIRST_VOTE 0
//using namespace cv;
template <class T>
class ImageLearner
{
public:
	TreeDict<T> mDict{};
	int mFeatureCount = 0;
public:
	ImageLearner() {};
	~ImageLearner() {};
public:
	virtual cv::Mat CalculateFeatures(cv::Mat img) = 0;
	//virtual int IdentifyImage(cv::Mat img) = 0;
	void LearnImage(cv::Mat img, int label)
	{
		cv::Mat features = CalculateFeatures(img);
		// Set origin node to zero
		cv::Mat origin = cv::Mat(1, features.cols, CV_32FC1, cv::Scalar(0.));
		mDict.SetRootNodeCenter(origin);
		for (size_t i = 0; i < features.rows; i++)
		{
			mDict.AddFeature(features.row(i), label);
			//std::cout << "feature No." << i << endl;
		}
		mFeatureCount += (int)features.rows;

	}

	int IdentifyImage(cv::Mat img, double* quality)
	{
		cv::Mat features = CalculateFeatures(img);
		vector<double> votes(NUM_ROOMS, 0);
		for (size_t i = 0; i < features.rows; i++)
		{
			vector<Word<cv::Mat> *> wordList = mDict.Search(features.row(i));
			typename vector<Word<cv::Mat> *>::iterator iter;
			for (iter = wordList.begin(); iter != wordList.end(); iter++)
			{
				vector<double> lVote = (*iter)->Vote();
				transform(votes.begin(), votes.end(), lVote.begin(), votes.begin(),
					plus<int>());
			}
		}
		vector<double>::iterator maxIter = max_element(votes.begin(), votes.end());
		double maxVote = *maxIter;
		double sumVote = 0;
		int result = distance(votes.begin(), maxIter);
		for (size_t i = 0; i < votes.size(); i++)
		{
			sumVote += votes[i];
		}
		*maxIter = 0; // set max vote to zero
		double secondVote = *max_element(votes.begin(), votes.end());
		*maxIter = maxVote;
		*quality = (maxVote - secondVote) / sumVote;
		cout << "Quality of vote is " << *quality << endl;
		if (*quality > THRESHOLD_FIRST_VOTE)
		{
			return result;
		}
		else
		{
			return -1;
		}
	}

};

template <class T>
class SIFTImageLearner : public ImageLearner<T>
{
public:
	cv::Mat CalculateFeatures(cv::Mat img)
	{
		cv::Ptr<cv::Feature2D> f2d = cv::xfeatures2d::SIFT::create();

		std::vector<cv::KeyPoint> keypoints;
		f2d->detect(img, keypoints);

		cv::Mat descriptors;
		f2d->compute(img, keypoints, descriptors);
		return descriptors;
	}

	// returns the room number, -1 if unidentified
};


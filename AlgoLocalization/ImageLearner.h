#pragma once
#include"TreeDict.h"
#include "opencv2/xfeatures2d.hpp"
#include "Constants.h"
using namespace cv;
using namespace Localization;
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
    virtual Mat CalculateFeatures(const Mat& img) = 0;
    //virtual int IdentifyImage(Mat img) = 0;
    void LearnImage(Mat img, int label)
    {
        Mat features = CalculateFeatures(img);
        // Set origin node to zero
        Mat origin = Mat(1, features.cols, CV_32FC1, Scalar(0.));
        mDict.SetRootNodeCenter(origin);
        for (size_t i = 0; i < features.rows; i++)
        {
            mDict.AddFeature(features.row(i), label);
            //std::cout << "feature No." << i << endl;
        }
        mFeatureCount += (int)features.rows;

    }

    int IdentifyImage(Mat img, double* quality)
    {
        Mat features = CalculateFeatures(img);
        vector<double> votes(NUM_ROOMS, 0);
        for (size_t i = 0; i < features.rows; i++)
        {
            vector<Word<Mat> *> wordList = mDict.Search(features.row(i));
            typename vector<Word<Mat> *>::iterator iter;
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
        *maxIter = maxVote; // restore
        *quality = (maxVote - secondVote) / sumVote;
        cout << "Quality of vote is " << *quality << endl;
        return (*quality > THRESHOLD_FIRST_VOTE) ? result : -1;
    }

};

//template <class T>
class SIFTImageLearner : public ImageLearner<Mat>
{
public:
    Mat CalculateFeatures(const Mat& img)
    {
        Ptr<Feature2D> f2d = xfeatures2d::SIFT::create();

        std::vector<KeyPoint> keypoints;
        f2d->detect(img, keypoints);

        Mat descriptors;
        f2d->compute(img, keypoints, descriptors);
        return descriptors;
    }

};

class ColorHistogramLearner : public ImageLearner<Mat>
{
public:
    // https://docs.opencv.org/2.4/modules/imgproc/doc/histograms.html?highlight=hsv
    Mat GetHue(const Mat& img)
    {
        Mat hsv;
        cvtColor(img, hsv, CV_BGR2HSV);
        vector<Mat> channels;
        split(hsv, channels);
        Mat hue = channels[0];
        return hue;
    }
    Mat CalculateFeatures(const Mat& img)
    {
        int nBins = DIM_COLOR_HIST;
        float hRanges[] = { 0, 180 };
        const float* histRanges = { hRanges };
        Mat features(0, nBins, CV_32FC1);
        vector<Mat> imgWindows = GetWindows(img);
        for (size_t i = 0; i < imgWindows.size(); i++)
        {
            Mat img = imgWindows[i];
            Mat hue = GetHue(img);
            MatND hist;
            calcHist(&hue, 1, 0, Mat(), hist, 1, &nBins, &histRanges, true, false);
            features.push_back(hist.t());
        }
        cout << "feaures count" << features.size() << endl;
        return features;
    }

    vector<Mat> GetWindows(const Mat& img)
    {
        vector<Mat> imgWindows;
        int width = img.cols;
        int height = img.rows;
        // Type 1: 40 x 40 every 20 pixels
        // Type 2: 20 x 20 every 10 pixels
        int size = 40;
        int stride = 20;
        // TODO: use stride (but will cause too)
        int nX = (int)((width - size) / stride);
        int nY = (int)((height - size) / stride);


        for (size_t i = 0; i < nX; i++)
        {
            for (size_t j = 0; j < nY; j++)
            {
                Rect window(i * stride, j * stride, size, size);
                Mat imgWindow(img(window));
                imgWindows.push_back(imgWindow);
            }

        }
        return imgWindows;
    }


};

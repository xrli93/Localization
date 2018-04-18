#pragma once
#include"TreeDict.h"
#include "opencv2/xfeatures2d.hpp"
#include "Constants.h"
using namespace cv;
using namespace Localization;


namespace Localization
{
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

        // First level voting
        int IdentifyImage(Mat img, double* quality = NULL, int ref = -1) // DEBUG: ref permit to check the output
        {
            Mat features = CalculateFeatures(img);
            vector<double> votes(NUM_ROOMS, 0);
            for (size_t i = 0; i < features.rows; i++)
            {
                vector<Word<Mat> *> wordList = mDict.Search(features.row(i), MAX_CHILD_NUM, FULL_SEARCH);
                typename vector<Word<Mat> *>::iterator iter;
                for (iter = wordList.begin(); iter != wordList.end(); iter++)
                {
                    // voting by words
                    vector<double> lVote = (*iter)->Vote();
                    transform(votes.begin(), votes.end(), lVote.begin(), votes.begin(),
                        plus<int>());
                }
            }
            if (quality == NULL)
            {
                quality = new double;
            }
            int result = CountVotes(votes, quality, THRESHOLD_FIRST_VOTE);
            if (ref > 0 && result > -1 && result != ref) // wrong result
            {
                //ReducImage(img, features, result, ref); // Check features
                ReducImage(features, ref);
            }
            return result;
        }

        void ReducImage(Mat img, Mat& features, int result, int ref)
        {
            static int count = 0;
            for (size_t i = 0; i < features.rows; i++)
            {
                vector<Word<Mat> *> wordList = mDict.Search(features.row(i), MAX_CHILD_NUM, FULL_SEARCH);
                vector<Word<Mat> *>::iterator iter;
                for (iter = wordList.begin(); iter != wordList.end(); iter++)
                {
                    Word<Mat> *word = (*iter);
                    if (word->GetLabels()[ref] != true)
                    {
                        cout << to_string(++count) << ", ";
                        word->UpdateLabel(ref); // Learn image
                    }
                }
            }
        }

        void ReducImage(Mat& features, int ref)
        {
            for (size_t i = 0; i < features.rows; i++)
            {
                mDict.AddFeature(features.row(i), ref);
            }
        }


        int CountFeatures()
        {
            return mFeatureCount;
        }

        void RemoveCommonWords()
        {
            mDict.RemoveCommonWords();
        }

        int CountWords()
        {
            return mDict.CountWords();
        }

        int CountNodes()
        {
            return mDict.CountNodes();
        }

        vector<int> AnalyseDict()
        {
            return mDict.AnalyseWords();
        }

    };

    int CountVotes(vector<double>& votes, double* quality = NULL, double threshold = THRESHOLD_FIRST_VOTE)
    {
        int result;
        vector<double>::iterator maxIter = max_element(votes.begin(), votes.end());
        double maxVote = *maxIter;
        double sumVote = 0;
        result = distance(votes.begin(), maxIter);
        for (size_t i = 0; i < votes.size(); i++)
        {
            sumVote += votes[i];
        }
        *maxIter = 0; // set max vote to zero
        double secondVote = *max_element(votes.begin(), votes.end());
        *maxIter = maxVote; // restore
        double lQuality = (maxVote - secondVote) / sumVote;
        *quality = lQuality;
        return (lQuality > threshold) ? result : -1;
    }



    class SIFTImageLearner : public ImageLearner<Mat>
    {
    public:
        SIFTImageLearner()
        {
            mDict.SetFeatureMethod(FEATURE_SIFT);
            mDict.SetRadius();
            mDict.SetFrontier();
        }

        SIFTImageLearner(double radius)
        {
            mDict.SetFeatureMethod(FEATURE_SIFT);
            mDict.SetRadius(radius);
        }


        Mat CalculateFeatures(const Mat& img)
        {

            Mat lImg(img);
            cvtColor(img, lImg, COLOR_BGR2GRAY);
            if (ENABLE_EQUALIZER)
            {
                equalizeHist(lImg, lImg);
            }
            //imshow("dfd", lImg);
            //waitKey(100);
            Ptr<Feature2D> f2d = xfeatures2d::SIFT::create(1000, 4, 0.03, 10, 1.6);

            std::vector<KeyPoint> keypoints;
            f2d->detect(lImg, keypoints);

            Mat descriptors;
            f2d->compute(lImg, keypoints, descriptors);
            return descriptors;
        }

    };

    class ColorHistogramLearner : public ImageLearner<Mat>
    {
    public:
        // https://docs.opencv.org/2.4/modules/imgproc/doc/histograms.html?highlight=hsv
        ColorHistogramLearner()
        {
            mDict.SetFeatureMethod(FEATURE_COLOR);
            mDict.SetRadius();
            mDict.SetFrontier();
        }
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
                if (ENABLE_HISTOGRAM_NORMALIZATION)
                {
                    normalize(hist, hist, 1, 0, NORM_L1, -1, Mat());
                }
                features.push_back(hist.t());
            }
            return features;
        }

        vector<Mat> GetWindows(const Mat& img)
        {
            vector<Mat> imgWindows;
            int width = img.cols;
            int height = img.rows;
            // Type 1: 40 x 40 every 20 pixels
            // Type 2: 20 x 20 every 10 pixels
            CalculateWindows(img, width, height, 80, 40, &imgWindows); // Type 1
            //CalculateWindows(img, width, height, 120, 40, &imgWindows); // Type 1
            return imgWindows;
        }

        void CalculateWindows(const Mat& img, int width, int height, int size, int stride, vector<Mat>* windows)
        {
            int nX = (int)((width - size) / stride);
            int nY = (int)((height - size) / stride);
            for (size_t i = 0; i < nX; i++)
            {
                for (size_t j = 0; j < nY; j++)
                {
                    Rect window(i * stride, j * stride, size, size);
                    Mat imgWindow(img(window));
                    windows->push_back(imgWindow);
                }
            }
        }
    };
}


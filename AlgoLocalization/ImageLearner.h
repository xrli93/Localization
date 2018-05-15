#pragma once
#include"TreeDict.h"
#include "opencv2/xfeatures2d.hpp"
#include "Constants.h"
//#include <boost/serialization/base_object.hpp>
using namespace cv;
using namespace Localization;

namespace Localization
{

    int CountVotes(vector<float>& votes, shared_ptr<float> quality = NULL, float threshold = THRESHOLD_FIRST_VOTE, double sumVotes = 0)
    {
        int result;
        vector<float>::iterator maxIter = max_element(votes.begin(), votes.end());
        float maxVote = *maxIter;
        float lSumVote = 0;
        result = distance(votes.begin(), maxIter);
        if (sumVotes == 0)
        {
            for (size_t i = 0; i < votes.size(); ++i)
            {
                lSumVote += votes[i];
            }
        }
        else
        {
            lSumVote = sumVotes;
        }
        *maxIter = 0; // set max vote to zero
        float secondVote = *max_element(votes.begin(), votes.end());
        *maxIter = maxVote; // restore
        float lQuality = (maxVote - secondVote) / lSumVote;
        *quality = lQuality;
        if (threshold == THRESHOLD_SECOND_VOTE && DISP_INCREMENTAL)
        {
            cout << maxVote << " " << secondVote << " " << lSumVote << " " << *quality << " " << endl;
        }
        return (lQuality >= threshold) ? result : -1;
    }

    template <class T>
    class ImageLearner
    {
    protected:
        TreeDict<T> mDict{};
        int mFeatureCount = 0;
        vector<int> mRoomFeaturesCount;
        vector<shared_ptr<Mat>> mListFeaturesSeen{};

        friend class cereal::access;
        template<class Archive>
        void serialize(Archive & archive)
        {
            archive(mDict, mFeatureCount, mRoomFeaturesCount, mListFeaturesSeen);
        }
    public:
        ImageLearner()
        {
            mRoomFeaturesCount = vector<int>(mConfig.GetRoomCount(), 0);
        }

        ~ImageLearner() {};

        void InitFeatureList()
        {
            mListFeaturesSeen.clear();
        }
        void AddRoom()
        {
            mRoomFeaturesCount.push_back(0);
        }
        void RemoveRoom(const string& room)
        {
            mRoomFeaturesCount.erase(mRoomFeaturesCount.begin() + mConfig.GetRoomIndex(room));
            mDict.RemoveRoom(room);
        }


    public:

        virtual Mat CalculateFeatures(const Mat& img) = 0;

        // Learn one image and return the number of features learrnt
        // TODO: perhaps split and expose features so that it can be also used by TopoMap
        int LearnImage(const Mat& img, int label)
        {
            Mat features = CalculateFeatures(img);
            // Set origin node to zero
            Mat origin = Mat(1, features.cols, CV_8UC1, Scalar(0.));
            mDict.SetRootNodeCenter(origin);
            for (int i = 0; i < features.rows; ++i)
            {
                mDict.AddFeature(features.row(i), label);
                //std::cout << "feature No." << i << endl;
            }
            int nFeatures = (int)features.rows;
            mFeatureCount += nFeatures;
            mRoomFeaturesCount[label] += nFeatures;
            return nFeatures;
        }

        // First level voting
        int IdentifyImage(const Mat& img, shared_ptr<float> quality = NULL) // DEBUG: ref permit to check the output
        {
            shared_ptr<Mat> features = make_shared<Mat>(CalculateFeatures(img));
            vector<float> votes(mConfig.GetRoomCount(), 0);
            for (size_t i = 0; i < features->rows; ++i)
            {
                vector<shared_ptr<Word<Mat> > > wordList = mDict.Search(features->row(i), FULL_SEARCH);
                typename vector<shared_ptr<Word<Mat> > >::iterator iter;
                for (iter = wordList.begin(); iter != wordList.end(); iter++)
                {
                    // voting by words
                    vector<float> lVote = (*iter)->Vote();
                    transform(votes.begin(), votes.end(), lVote.begin(), votes.begin(),
                        plus<int>());
                }
            }
            if (quality == NULL)
            {
                shared_ptr<float> quality = make_shared<float>();
            }
            int result = CountVotes(votes, quality, THRESHOLD_FIRST_VOTE);
            mListFeaturesSeen.push_back(features);
            return result;
        }

        // Actually can directly use AddFeature function....
        void ReducImage(Mat& features, int ref)
        {
            for (size_t i = 0; i < features.rows; ++i)
            {
                mDict.AddFeature(features.row(i), ref);
            }
        }

        // Follows directly a call to IdentifyImage()
        void ReducImage(int label)
        {
            for (size_t i = 0; i < mListFeaturesSeen.size(); i++)
            {
                Mat lFeatures = *(mListFeaturesSeen[i]);
                mFeatureCount += lFeatures.rows;
                for (size_t j = 0; j < lFeatures.rows; j++)
                {
                    mDict.AddFeature(lFeatures.row(j), label);
                }
            }
            mListFeaturesSeen.clear();
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


        vector<int> CountRoomFeatures()
        {
            return mRoomFeaturesCount;
        }

    };


    class SIFTImageLearner : public ImageLearner<Mat>
    {
    public:
        SIFTImageLearner()
        {
            mDict.SetFeatureMethod(USE_SIFT);
            mDict.SetRadius();
        }

        SIFTImageLearner(float radius)
        {
            mDict.SetFeatureMethod(USE_SIFT);
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
            if (ENABLE_CLAHE)
            {
                Ptr<CLAHE> clahe = createCLAHE(10, Size(8, 8));
                clahe->apply(lImg, lImg);
            }
            //imshow(" ", lImg);
            //waitKey(100);
            Ptr<Feature2D> f2d = xfeatures2d::SIFT::create(NUM_MAX_SIFT, 4, 0.03, 10, 1.6);
            //Ptr<Feature2D> f2d = xfeatures2d::SURF::create(400);
            //Ptr<Feature2D> f2d = ORB::create(150);

            std::vector<KeyPoint> keypoints;
            f2d->detect(lImg, keypoints);

            Mat descriptors;
            f2d->compute(lImg, keypoints, descriptors);
            if (descriptors.dims < 2)
            {
                cout << "Error getting desccriptors." << endl;
            }
            //cout << descriptors.type();
            //cout << descriptors.size();
            return descriptors;
        }

    };

    class ColorHistogramLearner : public ImageLearner<Mat>
    {
    public:
        // https://docs.opencv.org/2.4/modules/imgproc/doc/histograms.html?highlight=hsv
        ColorHistogramLearner()
        {
            mDict.SetFeatureMethod(USE_COLOR);
            mDict.SetRadius();
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
            Mat features(0, nBins, CV_8UC1);
            vector<Mat> imgWindows = GetWindows(img);
            for (size_t i = 0; i < imgWindows.size(); ++i)
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
            CalculateWindows(img, width, height, 120, 40, &imgWindows);
            CalculateWindows(img, width, height, 40, 40, &imgWindows);
            //CalculateWindows(img, width, height, 120, 40, &imgWindows); // Type 1
            return imgWindows;
        }

        void CalculateWindows(const Mat& img, int width, int height, int size, int stride, vector<Mat>* windows)
        {
            int nX = (int)((width - size) / stride);
            int nY = (int)((height - size) / stride);
            for (size_t i = 0; i < nX; ++i)
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


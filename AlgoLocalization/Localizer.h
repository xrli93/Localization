#pragma once

#include "TreeDict.h"
#include "opencv2/xfeatures2d.hpp"
#include "Constants.h"
#include "ImageLearner.h"
#include "TopoMap.h"
using namespace cv;
using namespace Localization;

namespace Localization
{

    class Orientation
    {
    private:
        double pi = 3.1415926; // radian or ?
        Ptr<Feature2D> f2d;
        Ptr<Feature2D> extract;
        BFMatcher matcher;

        vector<Mat> mDescriptors;
        vector<float> mAngles;

        friend class cereal::access;
        template<class Archive>
        void serialize(Archive & archive)
        {
            archive(mDescriptors, mAngles);
        }

    public:
        Orientation()
        {
            f2d = AgastFeatureDetector::create();
            extract = xfeatures2d::FREAK::create();
            matcher = BFMatcher(NORM_HAMMING);
        };

        ~Orientation() {}

        // Grayscale image
        void LearnImg(const Mat& img, float iOrientation)
        {
            std::vector<KeyPoint> keypoints;
            Mat descriptors;
            f2d->detect(img, keypoints);
            extract->compute(img, keypoints, descriptors);
            mDescriptors.push_back(descriptors);
            mAngles.push_back(iOrientation);
        }

        float GetOrientation(const Mat& img)
        {
            std::vector<KeyPoint> keypoints;
            Mat lDescriptors;
            f2d->detect(img, keypoints);
            extract->compute(img, keypoints, lDescriptors);
            const float nnRatio = 0.75f;
            vector<int> nMatchList(mDescriptors.size());
            for (int i = 1; i <= mDescriptors.size(); ++i)
            {
                int nMatches = 0;
                vector<vector<DMatch>> nnMatches;
                matcher.knnMatch(lDescriptors, mDescriptors[i - 1], nnMatches, 2);
                for (auto& x : nnMatches)
                {
                    if (x[0].distance < nnRatio * x[1].distance)
                    {
                        ++nMatches;
                    }
                }
                nMatchList[i - 1] = nMatches;
            }

            // Find two best results
            vector<int>::iterator maxIter = max_element(nMatchList.begin(), nMatchList.end());
            int maxMatch = *maxIter;
            int bestIndex = distance(nMatchList.begin(), maxIter);
            *maxIter = 0; // set max vote to zero

            auto secondIter = max_element(nMatchList.begin(), nMatchList.end());
            int secondMatch = *secondIter;
            int secondIndex = distance(nMatchList.begin(), secondIter);
            *maxIter = maxMatch; // restore

            int sumMatches = maxMatch * maxMatch + secondMatch * secondMatch;
            float factor1 = (maxMatch * maxMatch * 1.0) / sumMatches;
            float factor2 = (secondMatch * secondMatch * 1.0) / sumMatches;

            float bestAngle = mAngles[bestIndex];
            float secondAngle = mAngles[secondIndex];
            vector<float> lAngles{ bestAngle, secondAngle };

            float midAngle = Angles::CircularMean(lAngles, vector<float> {factor1, factor2});
            float stdDev = Angles::CircularStdDev(vector<float> {bestAngle, secondAngle});
            if (stdDev > MAX_STD_DEV || maxMatch < MIN_MATCH)
            {
                midAngle = NO_ORIENTATION;
            }
            return midAngle;
        }
    };

    class SimpleLocalizer
    {
    public:
        Ptr<Feature2D> f2d;
        Ptr<Feature2D> extract;
        BFMatcher matcher;
        map<string, vector<Mat>> mDescriptors;

        friend class cereal::access;
        template<class Archive>
        void serialize(Archive & archive)
        {
            archive(mDescriptors);
        }

        SimpleLocalizer()
        {
            f2d = BRISK::create(15, 5);
            extract = xfeatures2d::FREAK::create();
            matcher = BFMatcher(NORM_HAMMING);
        }


        void LearnImage(const Mat& img, string room, int iLandmark = -1, float iOrientation = 0)
        {
            Mat lDescriptors;
            vector<KeyPoint> lKeypoints;
            f2d->detect(img, lKeypoints);
            if (lKeypoints.size() != 0)
            {
                extract->compute(img, lKeypoints, lDescriptors);
            }
            auto lIter = mDescriptors.find(room);
            if (lIter == mDescriptors.end())
            {
                mDescriptors.insert(make_pair(room, vector<Mat> {lDescriptors}));
            }
            else
            {
                lIter->second.push_back(lDescriptors);
            }

        }

        string IdentifyRoom(const Mat& img, bool* finished, int ref = -1)
        {
            static int trys = 0;
            std::vector<KeyPoint> keypoints;
            Mat lDescriptors;
            f2d->detect(img, keypoints);
            extract->compute(img, keypoints, lDescriptors);

            const float nnRatio = 0.75f;
            static vector<float> lVotes(mDescriptors.size(), 0);
            for (auto& x : mDescriptors) // Rooms
            {
                vector<Mat> lRoomDescriptors = x.second;
                string lRoom = x.first;
                for (size_t i = 0; i < lRoomDescriptors.size(); i++) // Features in room
                {
                    int nMatches = 0;
                    vector<vector<DMatch>> nnMatches;
                    Mat refDescriptor = lRoomDescriptors[i];
                    if (refDescriptor.cols != 0)
                    {
                        matcher.knnMatch(lDescriptors, refDescriptor, nnMatches, 2);
                        for (auto& y : nnMatches)
                        {
                            if (y[0].distance < nnRatio * y[1].distance)
                            {
                                ++nMatches;
                            }
                        }
                        lVotes[mConfig.GetRoomIndex(lRoom)] += nMatches;
                    }
                    else
                    {
                        lVotes[mConfig.GetRoomIndex(lRoom)] += 0;
                    }
                }
            }
            for (size_t i = 0; i < lVotes.size(); i++)
            {
                cout << lVotes[i] << endl;
                //cout << mConfig.GetRoomName(i) << ", " << lVotes[i] << endl;
            }
            cout << endl;
            shared_ptr<float> quality = make_shared<float>(0);
            int lResult = CountVotes(lVotes, quality, 0.1);
            if (*quality >= THRESHOLD_SECOND_VOTE || ++trys >= NUM_MAX_IMAGES)
            {
                fill(lVotes.begin(), lVotes.end(), 0);
                trys = 0;
                *finished = true;
                return mConfig.GetRoomName(lResult);
            }
            return "";
        }

    };

    class Localizer
    {
    private:
        SIFTImageLearner mSIFTLearner;
        ColorHistogramLearner mColorLearner;
        Orientation mOrientation;
        SimpleLocalizer mSimpleLocalizer;
        TopoMap mMap;
        string mLastRoomLearned;

        friend class cereal::access;
        template<class Archive>
        void serialize(Archive & archive)
        {
            archive(mSIFTLearner, mColorLearner, mOrientation);
        }
    public:
        Localizer()
        {
            mSIFTLearner = SIFTImageLearner();
            mColorLearner = ColorHistogramLearner();
            mOrientation = Orientation();
        };

        ~Localizer() { };
        vector<int> AnalyseDict(int featureMethod)
        {
            return (featureMethod == USE_COLOR) ? mColorLearner.AnalyseDict() : mSIFTLearner.AnalyseDict();
        }

        vector<int> CountWords()
        {
            return vector<int> {mSIFTLearner.CountWords(), mColorLearner.CountWords()};
        }

        vector<int> CountNodes()
        {
            return vector<int> {mSIFTLearner.CountNodes(), mColorLearner.CountNodes()};
        }

        vector<int> CountFeatures()
        {
            return vector<int> {mSIFTLearner.CountFeatures(), mColorLearner.CountFeatures()};
        }

        void GetAnalyseOrientations()
        {
            cout << "SIFT angle variation: " << Average(mSIFTLearner.AnalyseOrientations()) << endl;
            cout << "Color angle variation: " << Average(mColorLearner.AnalyseOrientations()) << endl;
        }

        void RemoveCommonWords()
        {
            mSIFTLearner.RemoveCommonWords();
            mColorLearner.RemoveCommonWords();
        }
        void AddRoom(const string& room)
        {
            mConfig.AddRoomName(room);
            mSIFTLearner.AddRoom();
            mColorLearner.AddRoom();

        }
        void RemoveRoom(const string& room)
        {
            mSIFTLearner.RemoveRoom(room);
            mColorLearner.RemoveRoom(room);
            mConfig.RemoveRoom(room);
        }

        // TODO: Optimize structure
        void LearnImage(const Mat& img, string room, int iLandmark = -1, float iOrientation = 0)
        {
            int label = mConfig.GetRoomIndex(room);
            //Mat lImg = Mat(240, 320 CV_8UC1);
            //resize(img, lImg, lImg.size(), 0, 0, INTER_LINEAR);
            if (img.cols == 320)
            {
                // TODO: use color or not?
                mSIFTLearner.LearnImage(img, label, iLandmark, iOrientation);
                mColorLearner.LearnImage(img, label, iLandmark, iOrientation);
            }
            else
            {
                cout << "Formatting image" << endl;
                Mat lImg = Mat(240, 320, CV_8UC1);
                resize(img, lImg, lImg.size(), 0, 0, INTER_LINEAR);
                mSIFTLearner.LearnImage(lImg, label, iLandmark, iOrientation);
                mColorLearner.LearnImage(lImg, label, iLandmark, iOrientation);
            }

            // Add room connection
            if (!mLastRoomLearned.empty())
            {
                if (mLastRoomLearned.compare(room) != 0) // different room
                {
                    mMap.AddRoomConnection(mLastRoomLearned, room);
                }
            }
            mLastRoomLearned = room;
        }

        bool IsConnected(const string& room1, const string& room2)
        {
            return mMap.IsConnected(room1, room2);
        }


        // To determine if necessary to continue learning
        bool LearntEnoughFeatures()
        {
            vector<int> featuresCountSIFT = mSIFTLearner.CountRoomFeatures();
            vector<int> featuresCountColor = mColorLearner.CountRoomFeatures();
            int minCountSIFT = *min_element(featuresCountSIFT.begin(), featuresCountSIFT.end());
            int minCountColor = *min_element(featuresCountColor.begin(), featuresCountColor.end());
            if (minCountColor < NUM_MIN_FEATURES || minCountSIFT < NUM_MIN_FEATURES)
            {
                return false;
            }
            return true;
        }

        // Incremental learning. 
        // After each image, returns Room number if successfully recognized
        // Returns empty string if need more image, or unidentified
        string IdentifyRoom(const Mat& img, bool* halt = NULL, int ref = -1)
        {
            static vector<float> secondVotes(mConfig.GetRoomCount(), 0);
            static int trys = 0;
            shared_ptr<float> quality = make_shared<float>(0.);
            int SIFTVote = mSIFTLearner.IdentifyImage(img, quality);
            int ColorVote = mColorLearner.IdentifyImage(img, quality);

            double sumFactor = (PRIORITIZE_SIFT) ? (1 + WEIGHT_COLOR) : 2;
            if (SIFTVote > -1)
            {
                if (VERBOSE)
                {
                    cout << "SIFT voting" << SIFTVote << endl;
                }
                secondVotes[SIFTVote] += 1;
            }
            if (ColorVote > -1)
            {
                if (VERBOSE)
                {
                    cout << "Color voting" << ColorVote << endl;
                }
                secondVotes[ColorVote] += (PRIORITIZE_SIFT) ? WEIGHT_COLOR : 1;
            }
            double sumSecondVotes = 0;
            for (size_t i = 0; i < secondVotes.size(); ++i)
            {
                sumSecondVotes += secondVotes[i];
            }
            //int result = Localization::CountVotes(secondVotes, &quality, THRESHOLD_SECOND_VOTE, sumFactor * (trys+1));
            int roomIndex = Localization::CountVotes(secondVotes, quality, THRESHOLD_SECOND_VOTE);
            if ((*quality >= THRESHOLD_SECOND_VOTE && sumSecondVotes > 1) || ++trys >= NUM_MAX_IMAGES)
            {
                fill(secondVotes.begin(), secondVotes.end(), 0);
                trys = 0;
                *halt = true;
                return mConfig.GetRoomName(roomIndex);
            }
            return "";
        }

        // Second level voting for images on two feature spaces
        // Function for testing on databases
        int IdentifyRoom(vector<Mat> images, shared_ptr<float> quality = NULL, bool verbose = false, int ref = -1)
        {
            vector<float> secondVotes(mConfig.GetRoomCount(), 0);
            for (size_t i = 0; i < images.size(); ++i)
            {
                Mat img = images[i];
                int SIFTVote = mSIFTLearner.IdentifyImage(img, quality);
                int ColorVote = mColorLearner.IdentifyImage(img, quality);

                if (SIFTVote > -1)
                {
                    if (verbose)
                    {
                        cout << "SIFT voting" << SIFTVote << endl;
                    }
                    secondVotes[SIFTVote] += 1;
                }
                else if (verbose)
                {
                    cout << "SIFT not voting" << endl;
                }

                if (ColorVote > -1)
                {
                    if (verbose)
                    {
                        cout << "Color voting" << ColorVote << endl;
                    }
                    // SIFT Important
                    secondVotes[ColorVote] += (PRIORITIZE_SIFT) ? WEIGHT_COLOR : 1;
                }
                else if (verbose)
                {
                    cout << "Color not voting" << endl;
                }

                // SIFT important
                if (PRIORITIZE_SIFT && images.size() == 1 && ColorVote > -1 && SIFTVote > -1)
                {
                    if (ColorVote != SIFTVote)
                    {
                        return SIFTVote;
                    }
                }
            }
            return Localization::CountVotes(secondVotes, quality, THRESHOLD_SECOND_VOTE);
        }

        void LearnOrientation(const Mat& img, float iOrientation)
        {
            mOrientation.LearnImg(img, iOrientation);
        }

        float GetOrientation(const Mat& img)
        {
            return mOrientation.GetOrientation(img);
        }

        float GetOrientationToLandmark(const Mat& img, int iLandmark)
        {
            // For the moment SIFT only, can easy add Color
            float lSIFTAngle = mSIFTLearner.GetOrientationToLandmark(img, iLandmark);
            return lSIFTAngle;
        }
    };

}




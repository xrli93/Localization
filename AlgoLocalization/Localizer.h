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
    public:
        FREEImageLearner mFREELearner;
        ColorHistogramLearner mColorLearner;
        Orientation mOrientation;
        SimpleLocalizer mSimpleLocalizer;
        TopoMap mMap;
        string mLastRoomLearned;

        friend class cereal::access;
        template<class Archive>
        void serialize(Archive & archive)
        {
            archive(mFREELearner, mColorLearner, mOrientation, mMap, mLastRoomLearned);
        }
    public:
        Localizer()
        {
            mFREELearner = FREEImageLearner();
            mColorLearner = ColorHistogramLearner();
            mOrientation = Orientation();
        };

        ~Localizer() { };
        vector<int> AnalyseDict(int featureMethod)
        {
            return (featureMethod == USE_COLOR) ? mColorLearner.AnalyseDict() : mFREELearner.AnalyseDict();
        }

        vector<int> CountWords()
        {
            return vector<int> {mFREELearner.CountWords(), mColorLearner.CountWords()};
        }

        vector<int> CountNodes()
        {
            return vector<int> {mFREELearner.CountNodes(), mColorLearner.CountNodes()};
        }

        vector<int> CountFeatures()
        {
            return vector<int> {mFREELearner.CountFeatures(), mColorLearner.CountFeatures()};
        }

        void GetAnalyseOrientations()
        {
            cout << "FREE angle variation: " << Average(mFREELearner.AnalyseOrientations()) << endl;
            cout << "Color angle variation: " << Average(mColorLearner.AnalyseOrientations()) << endl;
        }

        void RemoveCommonWords()
        {
            mFREELearner.RemoveCommonWords();
            mColorLearner.RemoveCommonWords();
        }
        void AddRoom(const string& room)
        {
            mConfig.AddRoomName(room);
            mFREELearner.AddRoom();
            mColorLearner.AddRoom();

        }
        void RemoveRoom(const string& room)
        {
            mFREELearner.RemoveRoom(room);
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
                mFREELearner.LearnImage(img, label, iLandmark, iOrientation);
                mColorLearner.LearnImage(img, label, iLandmark, iOrientation);
            }
            else
            {
                cout << "Formatting image" << endl;
                Mat lImg = Mat(240, 320, CV_8UC1);
                resize(img, lImg, lImg.size(), 0, 0, INTER_LINEAR);
                mFREELearner.LearnImage(lImg, label, iLandmark, iOrientation);
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
            vector<int> featuresCountFREE = mFREELearner.CountRoomFeatures();
            vector<int> featuresCountColor = mColorLearner.CountRoomFeatures();
            int minCountFREE = *min_element(featuresCountFREE.begin(), featuresCountFREE.end());
            int minCountColor = *min_element(featuresCountColor.begin(), featuresCountColor.end());
            if (minCountColor < NUM_MIN_FEATURES || minCountFREE < NUM_MIN_FEATURES)
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
            int FREEVote = mFREELearner.IdentifyImage(img, quality);
            int ColorVote = mColorLearner.IdentifyImage(img, quality);

            double sumFactor = (PRIORITIZE_FREE) ? (1 + WEIGHT_COLOR) : 2;
            if (FREEVote > -1)
            {
                if (VERBOSE)
                {
                    cout << "FREE voting" << FREEVote << endl;
                }
                secondVotes[FREEVote] += 1;
            }
            if (ColorVote > -1)
            {
                if (VERBOSE)
                {
                    cout << "Color voting" << ColorVote << endl;
                }
                secondVotes[ColorVote] += (PRIORITIZE_FREE) ? WEIGHT_COLOR : 1;
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

            // CLASSICAL
            vector<float> secondVotes(mConfig.GetRoomCount(), 0);
            for (size_t i = 0; i < images.size(); ++i)
            {
                Mat img = images[i];
                int FREEVote = mFREELearner.IdentifyImage(img, quality);
                int ColorVote = mColorLearner.IdentifyImage(img, quality);

                if (FREEVote > -1)
                {
                    if (verbose)
                    {
                        cout << "FREE voting" << FREEVote << endl;
                    }
                    secondVotes[FREEVote] += 1;
                }
                else if (verbose)
                {
                    cout << "FREE not voting" << endl;
                }

                if (ColorVote > -1)
                {
                    if (verbose)
                    {
                        cout << "Color voting" << ColorVote << endl;
                    }
                    // FREE Important
                    secondVotes[ColorVote] += (PRIORITIZE_FREE) ? WEIGHT_COLOR : 1;
                }
                else if (verbose)
                {
                    cout << "Color not voting" << endl;
                }

                // FREE important
                if (PRIORITIZE_FREE && images.size() == 1 && ColorVote > -1 && FREEVote > -1)
                {
                    if (ColorVote != FREEVote)
                    {
                        return FREEVote;
                    }
                }
            }
            return Localization::CountVotes(secondVotes, quality, THRESHOLD_SECOND_VOTE);
        }

        // Orientation + Navigation
        void LearnOrientation(const Mat& img, float iOrientation, string iRoom, int iLandmark = 0)
        {
            //mOrientation.LearnImg(img, iOrientation, iLandmark);
            mMap.LearnOrientation(img, iOrientation, iRoom, iLandmark);
        }

        float GetOrientation(const Mat& img, int iPath = -1)
        {
            //return mOrientation.GetOrientation(img);
            return mMap.GetOrientation(img, iPath);
        }

        // ---------------- Odometry / Paths -----------------

        // Find the closest landmark in the room
        int FindNearestLandmark(string iRoom, const Mat& img, int nImgs = 6)
        {
            //return mOrientation.FindNearestLandmark(img, nImgs);
            return mMap.FindNearestLandmark(iRoom, img, nImgs);
        }

        // Find the closest landmark on the path
        int FindNearestLandmark(int iPath, const Mat& img, int nImgs = 1)
        {
            return mMap.FindNearestLandmark(iPath, img, nImgs);
        }

        // Check if a step is finished
        bool FinishedStep(int iPath, int iStep, float iOdometry)
        {
            return mMap.ReachedNextLandmark(iPath, iStep, iOdometry);
        }

        // Update progress in the pass (which landmarks to search)
        void UpdatePathProgress(int iPath, int iStep)
        {
            mMap.UpdatePathProgress(iPath, iStep);
        }

        int GetLandmarkAtStep(int iPath, int iStep)
        {
            return mMap.GetLandmarkAtStep(iPath, iStep);
        }

        // Find the index of landmark in a path
        int FindLandmarkInPath(int iPath, int iLandmark)
        {
            return mMap.FindLandmarkInPath(iPath, iLandmark);
        }

        // Add a landmark (already with learned images to path
        void AddLandmarkToPath(int iPath, int iLandmark)
        {
            mMap.AddLandmarkToPath(iPath, iLandmark);
        }

        // Add distance and orientation information between landmarks
        void UpdateLandmarkDistance(int iStart, int iEnd, float iDist, float iOrientation)
        {
            mMap.UpdateLandmark(iStart, iEnd, iDist, iOrientation);
        }


#pragma region Paths

        // add landmark in iInRoom that leands to iDestRoom
        void AddKeyLandmark(string iInRoom, int iLandmark, string iDestRoom)
        {
            mMap.AddKeyLandmark(iInRoom, iLandmark, iDestRoom);
        }

        void ConnectRoom(string iRoom1, string iRoom2)
        {
            return mMap.AddRoomConnection(iRoom1, iRoom2);
        }

        int FindPath(int iStart, int iEnd)
        {
            return mMap.FindPathBetweenLandmarks(iStart, iEnd);
        }

        int FindPathToRoom(int iStartLandmark, string iRoom)
        {
            return mMap.FindPathToRoom(iStartLandmark, iRoom);
        }
        
    };

}




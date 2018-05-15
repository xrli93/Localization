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

    class Localizer
    {
    private:
        SIFTImageLearner mSIFTLearner;
        ColorHistogramLearner mColorLearner;
        TopoMap mMap;
        string mLastRoomLearned;

        friend class cereal::access;
        template<class Archive>
        void serialize(Archive & archive)
        {
            archive(mSIFTLearner, mColorLearner);
        }
    public:
        Localizer() { };

        ~Localizer() { };
        //vector<int> AnalyseDict(int featureMethod)
        //{
        //    return (featureMethod == USE_COLOR) ? mColorLearner.AnalyseDict() : mSIFTLearner.AnalyseDict();
        //}

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

        void LearnImageOnLandmark(const Mat& img, string room, int landmark, float angle)
        {

        }

        // Actually without internal storage of img and lable
        // TODO: Optimize structure
        void LearnImage(const Mat& img, string room)
        {
            int label = mConfig.GetRoomIndex(room);
            //Mat lImg = Mat(240, 320 CV_8UC1);
            //resize(img, lImg, lImg.size(), 0, 0, INTER_LINEAR);
            if (img.cols == 320)
            {
                mSIFTLearner.LearnImage(img, label);
                mColorLearner.LearnImage(img, label);
            }
            else
            {
                cout << "Formatting image" << endl;
                Mat lImg = Mat(240, 320, CV_8UC1);
                resize(img, lImg, lImg.size(), 0, 0, INTER_LINEAR);
                mSIFTLearner.LearnImage(lImg, label);
                mColorLearner.LearnImage(lImg, label);
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

    };
}




#pragma once

#include"TreeDict.h"
#include "opencv2/xfeatures2d.hpp"
#include "Constants.h"
#include"ImageLearner.h"
using namespace cv;
using namespace Localization;

namespace Localization
{

    class Localizer
    {
    private:
        SIFTImageLearner mSIFTLearner;
        ColorHistogramLearner mColorLearner{};
        vector<double> secondVotes;
        vector<Mat> imageCollection;
        vector<int> labelCollection;
    public:
        Localizer()
        {
            secondVotes = vector<double>(NUM_ROOMS, 0);
        };


        ~Localizer() { };

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


        void AddImage(Mat img, int label)
        {
            Mat lImg = Mat(320, 240, CV_8UC1); // TODO
            resize(img, lImg, lImg.size(), 0, 0, INTER_LINEAR);
            imageCollection.push_back(lImg);
            labelCollection.push_back(label);
        }

        void LearnCollection()
        {
            for (size_t i = 0; i < imageCollection.size(); i++)
            {
                Mat img = imageCollection[i];
                int label = labelCollection[i];
                mSIFTLearner.LearnImage(img, label);
                mColorLearner.LearnImage(img, label);
            }
        }

        // Second level voting for images on two feature spaces
        int IdentifyRoom(vector<Mat> images, double* quality = NULL)
        {
            fill(secondVotes.begin(), secondVotes.end(), 0);
            for (size_t i = 0; i < images.size(); i++)
            {
                Mat img = images[i];
                int SIFTVote = mSIFTLearner.IdentifyImage(img, quality);
                int ColorVote = mColorLearner.IdentifyImage(img, quality);
                if (SIFTVote > -1)
                {
                    cout << "SIFT voting" << SIFTVote << endl;
                    secondVotes[SIFTVote] += 1;
                }
                else
                {
                    cout << "SIFT not voting" << endl;
                }

                if (ColorVote > -1)
                {
                    cout << "Color voting" << ColorVote << endl;
                    secondVotes[ColorVote] += 1;
                }
                else
                {
                    cout << "Color not voting" << endl;
                }
            }
            return Localization::CountVotes(secondVotes, quality, THRESHOLD_SECOND_VOTE);
        }

    };
}




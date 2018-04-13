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
        SIFTImageLearner SIFTLearner{};
        ColorHistogramLearner ColorLearner{};
        vector<double> secondVotes;
        vector<Mat> imageCollection;
        vector<int> labelCollection;
    public:
        Localizer()
        {
            secondVotes = vector<double>(NUM_ROOMS, 0);
        };

        ~Localizer() { };

        void AddImage(Mat img, int label)
        {
            imageCollection.push_back(img);
            labelCollection.push_back(label);
        }

        void LearnCollection()
        {
            for (size_t i = 0; i < imageCollection.size(); i++)
            {
                Mat img = imageCollection[i];
                int label = labelCollection[i];
                SIFTLearner.LearnImage(img, label);
                ColorLearner.LearnImage(img, label);
            }
        }

        int IdentifyRoom(vector<Mat> images, vector<int> labels)
        {
            for (size_t i = 0; i < images.size(); i++)
            {
                Mat img = images[i];
                int label = labels[i];
                int SIFTVote = SIFTLearner.IdentifyImage(img);
                int ColorVote = ColorLearner.IdentifyImage(img);
                if (SIFTVote > 0)
                {
                    secondVotes[SIFTVote] += 1;
                }

                if (ColorVote > 0)
                {
                    secondVotes[ColorVote] += 1;
                }
            }
            double quality;
            return Localization::CountVotes(secondVotes, &quality);
        }

    };
}




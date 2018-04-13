#pragma once
#include<vector>
#include<assert.h>
#include<cmath>
#include "Constants.h"
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

using namespace std;


namespace Localization
{

    template <class T>
    class Word
    {
    private:
        T mCenter; // word center in feature space
        double mRadius = RADIUS; // radius of word 
        vector<bool> mPresenceRooms; // seen in which rooms

    private:
        void initMPresenceRooms()
        {
            for (int i = 0; i < NUM_ROOMS; i++)
            {
                mPresenceRooms.push_back(false);
            }
        }
    public:
        Word()
        {
            initMPresenceRooms();
        };

        Word(T feature) : mCenter(feature) { initMPresenceRooms(); }

        Word(T feature, int indexRoom) : mCenter(feature)
        {
            initMPresenceRooms();
            UpdateLabel(indexRoom);
        }

        Word(T feature, int indexRoom, double radius) : mCenter(feature), mRadius(radius)
        {
            initMPresenceRooms();
            UpdateLabel(indexRoom);
        }

        ~Word() {};

        T GetCenter()
        {
            return mCenter;
        }

        void SetCenter(T center)
        {
            mCenter = center;
        }

        void UpdateLabel(int indexRoom)
        {
            if (indexRoom < NUM_ROOMS)
            {
                mPresenceRooms[indexRoom] = true;
            }
            // else error
        }

        bool ContainFeature(T feature)
        {
            return (Localization::CalculateDistance(feature, mCenter) < mRadius) ? true : false;
        }

        void Display()
        {
            for (int i = 0; i < NUM_ROOMS; i++)
            {
                cout << mPresenceRooms[i] << endl;
            }
        }

        vector<double> Vote()
        {
            vector<double> scores(NUM_ROOMS, 0);
            int roomsSeen = 0;
            for (size_t i = 0; i < NUM_ROOMS; i++)
            {
                roomsSeen += mPresenceRooms[i] ? 1 : 0;
            }
            if (roomsSeen > 0)
            {
                for (size_t i = 0; i < scores.size(); i++)
                {
                    scores[i] = mPresenceRooms[i] ? log(NUM_ROOMS * 1.0 / roomsSeen) / log(NUM_ROOMS) : 0;
                }
            }
            return scores;
        }

    };

    template <class T>
    double CalculateDistance(const T& x, const T& y) {}

    template<> double CalculateDistance(const vector<double>& x, const vector<double>& y)
    {
        assert(x.size() == y.size());
        double norm = 0;
        for (size_t i = 0; i != x.size(); i++)
        {
            norm += (x[i] - y[i]) * (x[i] - y[i]);
        }
        return sqrt(norm);
    }


    template<> double CalculateDistance(const int& x, const int& y)
    {
        return abs(x - y);
    }

    template<> double CalculateDistance(const cv::Mat& x, const cv::Mat& y)
    {
        if (x.cols == DIM_SIFT) //  Sift features
        {
            return cv::norm(x - y, cv::NORM_L2);
        }
        else if (x.cols == DIM_COLOR_HIST)
        {
            //TODO: implement or find diffusion distance as mentioned in Filliat 08 [27]
            return compareHist(x, y, CV_COMP_CHISQR);  // lower the closer. CV_COMP_BHATTACHARYYA possible.
        }
    }
}

//class WordSIFT : public Word<Mat>
//{
//public:
//	double CalculateDistance(Mat x, Mat y)
//	{
//		return 0;
//	}
//};
//

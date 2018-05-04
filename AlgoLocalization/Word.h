#pragma once
#include<vector>
#include<assert.h>
#include<cmath>
#include<memory>
#include "Constants.h"
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "cereal\archives\portable_binary.hpp"
#include "cereal\types\vector.hpp"
#include "cereal\types\memory.hpp"
using namespace std;
using namespace cv;

namespace Localization
{

    template <class T>
    class Word
    {
    private:
        T mCenter; // word center in feature space
        float mRadius = RADIUS; // radius of word 
        vector<bool> mPresenceRooms; // seen in which rooms

        friend class cereal::access;
        template<class Archive>
        void serialize(Archive & archive)
        {
            archive(mCenter, mRadius, mPresenceRooms);
        }

        // Serialization
        //friend class boost::serialization::access;
        //template<typename Archive>
        //void serialize(Archive& ar, const unsigned version)
        //{
        //    ar & mCenter & mRadius & mPresenceRooms;
        //}

        // cereal

    private:
        void initMPresenceRooms()
        {
            for (int i = 0; i < GetNumRoom(); ++i)
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

        Word(T feature, int indexRoom, float radius) : mCenter(feature), mRadius(radius)
        {
            initMPresenceRooms();
            UpdateLabel(indexRoom);
        }

        ~Word() {};

        T GetCenter()
        {
            return mCenter;
        }

        float GetRadius()
        {
            return mRadius;
        }

        void SetCenter(T center)
        {
            mCenter = center;
        }

        vector<bool> GetLabels()
        {
            return mPresenceRooms;
        }

        void UpdateLabel(int indexRoom)
        {
            if (indexRoom < GetNumRoom())
            {
                mPresenceRooms[indexRoom] = true;
            }
            else
            {
                mPresenceRooms.push_back(true);
                assert(mPresenceRooms.size() - 1 == indexRoom);
            }
        }

        bool ContainFeature(T feature)
        {
            return (Localization::CalculateDistance(feature, mCenter) < mRadius) ? true : false;
        }

        bool PresentInAll()
        {
            return all_of(mPresenceRooms.begin(), mPresenceRooms.end(),
                [](bool x) {return x; }) ? true : false;
        }

        void Display()
        {
            for (int i = 0; i < GetNumRoom(); ++i)
            {
                cout << mPresenceRooms[i];
            }
            cout << endl;
        }

        // Vote using inverse document frequency 
        vector<float> Vote()
        {
            vector<float> scores(GetNumRoom(), 0);
            int roomsSeen = 0;
            for (size_t i = 0; i < GetNumRoom(); ++i)
            {
                roomsSeen += mPresenceRooms[i] ? 1 : 0;
            }
            // DEBUG: Only unique words vote
            //if (roomsSeen == 1)
            //{
            //    for (size_t i = 0; i < scores.size(); ++i)
            //    {
            //        scores[i] = mPresenceRooms[i] ? 1 : 0;
            //    }

            //}
            if (roomsSeen > 0)
            {
                for (size_t i = 0; i < scores.size(); ++i)
                {
                    scores[i] = mPresenceRooms[i] ? log(GetNumRoom() * 1.0 / roomsSeen) / log(GetNumRoom()) : 0;
                }
            }
            return scores;
        }

    };

    float DiffusionDistance(const Mat& x, const Mat& y, float sigma = 1.6)
    {
        Mat d = x - y;
        float dist = 0;
        dist += norm(d, NORM_L1);
        while (d.cols > 1)
        {
            GaussianBlur(d, d, Size(0, 0), sigma);
            resize(d, d, Size(), 0.5, 1, INTER_NEAREST);
            dist += norm(d, NORM_L1);
        }
        return dist;
    }

    template <class T>
    float CalculateDistance(const T& x, const T& y) {}

    template<> float CalculateDistance(const vector<float>& x, const vector<float>& y)
    {
        assert(x.size() == y.size());
        float norm = 0;
        for (size_t i = 0; i != x.size(); ++i)
        {
            norm += (x[i] - y[i]) * (x[i] - y[i]);
        }
        return sqrt(norm);
    }


    template<> float CalculateDistance(const int& x, const int& y)
    {
        return abs(x - y);
    }

    template<> float CalculateDistance(const Mat& x, const Mat& y)
    {
        if (x.cols == DIM_SIFT || x.cols != DIM_COLOR_HIST) //  Sift features
        {
            //cout << compareHist(x, y, CV_COMP_KL_DIV);
            Mat diff(x.size(), x.type());
            Mat temp(x.size(), x.type());
            if (y.type() != x.type())
            {
                y.convertTo(temp, x.type());
                diff = temp - x;
            }
            else
            {
                diff = y - x;
            }
            float dist = norm(diff, NORM_L2);
            //float dist = compareHist(x, y, CV_COMP_CHISQR);
            if (DISP_DEBUG)
            {
                cout << dist << endl;
            }
            return dist;
        }
        else if (x.cols == DIM_COLOR_HIST)
        {
            //TODO: implement or find diffusion distance as mentioned in Filliat 08 [27]
            //return compareHist(x, y, CV_COMP_CHISQR);  // lower the closer. CV_COMP_BHATTACHARYYA possible.

            float dist = (mNorm == NORM_KL) ? compareHist(x, y, CV_COMP_KL_DIV) : DiffusionDistance(x, y, 1.0);
            //float dist = DiffusionDistance(x, y, 1.2);
            //cout << dist << endl;
            return dist;  // lower the closer. CV_COMP_BHATTACHARYYA possible.

        }
    }

}


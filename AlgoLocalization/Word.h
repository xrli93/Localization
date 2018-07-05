#pragma once
#include<vector>
#include<assert.h>
#include<cmath>
#include <math.h>`
#include<memory>
#include "Constants.h"
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "cereal\archives\portable_binary.hpp"
#include "cereal\types\vector.hpp"
#include "cereal\types\memory.hpp"
#include "cereal\types\map.hpp"
using namespace std;
using namespace cv;

namespace Localization
{
    namespace Angles
    {

        float DegToRad(float iAngleInDegree)
        {
            return 2 * PI * (iAngleInDegree / 360);
        }

        float RadToDeg(float iAngleInRad)
        {
            return 360 * (iAngleInRad / (2 * PI));
        }

        // iArray of angles in degrees
        float CircularMean(const vector<float>& iAnglesInDegree, const vector<float>& iWeights = vector<float>())
        {
            float sumSin = 0;
            float sumCos = 0;
            bool useWeights = (iWeights.size() != 0);
            //for (const float& x : iAnglesInDegree)
            //{
            //    sumCos += cos(DegToRad(x));
            //    sumSin += sin(DegToRad(x));
            //}
            for (size_t i = 0; i < iAnglesInDegree.size(); i++)
            {
                float angle = iAnglesInDegree[i];

                if (useWeights)
                {
                    sumCos += cos(DegToRad(angle)) * iWeights[i];
                    sumSin += sin(DegToRad(angle)) * iWeights[i];
                }
                else
                {
                    sumCos += cos(DegToRad(angle));
                    sumSin += sin(DegToRad(angle));
                }
            }
            sumCos /= iAnglesInDegree.size();
            sumSin /= iAnglesInDegree.size();
            float mean = RadToDeg(atan2(sumSin, sumCos)); // [-pi, pi) -> [0, 360);
            if (mean < 0)
            {
                mean += 360;
            }
            //DEBUG
            if (isnan(mean))
            {
                cout << "mean test" << mean << ", " << sumCos << ", " << sumSin << endl;
            }
            return mean;
        }

        float CircularStdDev(const vector<float>& iAnglesInDegree, const vector<float>& iWeights = vector<float>())
        {
            if (iAnglesInDegree.size() > 0)
            {
                float sumSin = 0;
                float sumCos = 0;
                bool useWeights = (iWeights.size() != 0);
                float lSumWeights = 0;
                //for (const float& x : iAnglesInDegree)
                for (size_t i = 0; i < iAnglesInDegree.size(); i++)
                {
                    float lAngle = iAnglesInDegree[i];
                    if (useWeights)
                    {
                        sumCos += cos(DegToRad(lAngle)) * iWeights[i];
                        sumSin += sin(DegToRad(lAngle)) * iWeights[i];
                    }
                    else
                    {
                        sumCos += cos(DegToRad(lAngle));
                        sumSin += sin(DegToRad(lAngle));
                    }
                }
                if (useWeights)
                {
                    float lSumWeights = 0;
                    for (auto& x : iWeights)
                    {
                        lSumWeights += x;
                    }
                    sumCos /= lSumWeights;
                    sumSin /= lSumWeights;
                }
                else
                {
                    sumCos /= iAnglesInDegree.size();
                    sumSin /= iAnglesInDegree.size();
                }
                float radius = sumSin * sumSin + sumCos * sumCos;
                float stdDev = (abs(radius - 1.0f) < 0.001) ? 0 : sqrt(-log(sumSin * sumSin + sumCos * sumCos));
                //DEBUG
                if (isnan(stdDev))
                {
                    cout << "atan test" << stdDev << ", " << sumCos << ", " << sumSin << ", " << (sumSin * sumSin + sumCos * sumCos) << "," << -log(sumSin * sumSin + sumCos * sumCos) << endl;
                }
                //cout << stdDev << endl;
                return stdDev;
            }
            else
            {
                return THRESHOLD_CIRCULAR_SECOND + 1; // No angles to use
            }
        }

        float AverageAngles(vector<float>& iAngles, vector<float>& iStdDevs, vector<int>& iNMatches)
        {
            vector<float> lWeights;
            vector<float> lAngles;
            for (size_t i = 0; i < iStdDevs.size(); i++)
            {
                if (iAngles[i] < NO_ORIENTATION)
                {
                    // iStdDevs[i] = 0?
                    lWeights.push_back(iNMatches[i] / sqrt(iStdDevs[i]));
                    lAngles.push_back(iAngles[i]);
                }
            }
            if (lAngles.size() == 0 || CircularStdDev(lAngles) > MAX_STD_DEV * TOLERANCE_ANGLE) // More tolereant
            {
                return NO_ORIENTATION;
            }
            else
            {
                return Angles::CircularMean(lAngles, lWeights);
            }
        }




    }
    template <typename T>
    T Average(const vector<T>& iArray)
    {
        T average = 0;
        for (const T& x : iArray)
        {
            average += x;
        }
        return (average /= iArray.size());
    }


    template <typename T>
    T StandarDeviation(const vector<T>& iArray)
    {
        T average = Average(iArray);
        T sigma = 0;
        for (const T& x : iArray)
        {
            sigma += (x - average) * (x - average);
        }
        //cout << "Circular: " << Angles::CircularStdDev(iArray) << " " << "normal" << sqrt(sigma / iArray.size()) << endl;
        return (T)sqrt(sigma / iArray.size());
        //return 0;
    }

    template <class T>
    class Word
    {
    private:
        T mCenter; // word center in feature space
        float mRadius = RADIUS; // radius of word 
        vector<bool> mPresenceRooms; // seen in which rooms
        map<int, vector<float>> mOrientation; // orientation(float) in ith landmark(int). Attention! one word can have multiple orientations at one landmark
        map<int, vector<T>> mSeenFeatures; // features seen with their room index

        friend class cereal::access;
        template<class Archive>
        void serialize(Archive & archive)
        {
            archive(mCenter, mRadius, mPresenceRooms, mOrientation, mSeenFeatures);
        }


    private:
        void initMPresenceRooms()
        {
            for (int i = 0; i < mConfig.GetRoomCount(); ++i)
            {
                mPresenceRooms.push_back(false);
            }
            //for (auto const& room : mConfig.mRooms)
            //{
            //    mPresenceRooms.insert(make_pair(room, false))
            //}
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
            if (indexRoom < mConfig.GetRoomCount())
            {
                mPresenceRooms[indexRoom] = true;
            }
            else
            {
                //mPresenceRooms.push_back(true);
                //assert(mPresenceRooms.size() - 1 == indexRoom);
                cout << "Error adding new room information!" << endl;
            }
        }

        void AddOrientation(const int& iLandmark, const float& iOrientation)
        {
            auto lIter = mOrientation.find(iLandmark);
            if (lIter == mOrientation.end()) // No existing orientations
            {
                vector<float> lOrientation;
                lOrientation.push_back(iOrientation);
                mOrientation.insert(make_pair(iLandmark, lOrientation));
            }
            else // already exists entry in map
            {
                lIter->second.push_back(iOrientation);
            }
        }

        void AddSeenFeature(const int& iRoomIndex, const T& iFeature)
        {
            auto lIter = mSeenFeatures.find(iRoomIndex);
            if (lIter == mSeenFeatures.end()) // No existing orientations
            {
                vector<T> lFeatureList;
                lFeatureList.push_back(iFeature);
                mSeenFeatures.insert(make_pair(iRoomIndex, lFeatureList));
            }
            else // already exists entry in map
            {
                lIter->second.push_back(iFeature);
            }
        }

        map<int, vector<T>> GetSeenFeatures()
        {
            return mSeenFeatures;
        }

        vector<float> AnalyseWordOrientations()
        {
            vector<float> lStdDevs;
            for (auto& x : mOrientation)
            {
                float lStdDev = Angles::CircularStdDev(x.second);
                if (lStdDev > 0.01)
                {
                    lStdDevs.push_back(lStdDev);
                }
            }
            return lStdDevs;
        }

        float GetOrientation(const int& iLandmark)
        {
            auto lIter = mOrientation.find(iLandmark);
            if (lIter != mOrientation.end())
            {
                vector<float> orientations = lIter->second;
                if (DISP_DEBUG_ORIENTATION)
                {
                    for (auto& x : orientations)
                    {
                        cout << x;
                    }
                    cout << endl;
                }
                //if standard deviation too large, neglect word
                if (USE_CIRCULAR)
                {
                    if (orientations.size() > 0 && Angles::CircularStdDev(orientations) < THRESHOLD_CIRCULAR_FIRST)
                    {
                        return Angles::CircularMean(orientations);
                    }
                    else
                    {
                        //cout << "word stddev" << Angles::CircularStdDev(orientations) << endl;
                        //for (auto& x : orientations)
                        //{
                        //    cout << x << ", ";
                        //}
                        //cout << endl;
                    }
                }
                else
                {
                    if (orientations.size() > 0 && StandarDeviation(orientations) < THRESHOLD_ORIENTATION)
                    {
                        return Average(orientations);
                    }
                }
            }
            return NO_ORIENTATION; // n'importe quoi!
        }


        bool ContainFeature(T feature)
        {
            if (mRadius == RADIUS_FREE && DEBUG)
            {
                //cout << Localization::CalculateDistance(mCenter, feature) << ", " << mRadius << endl;
            }
            return (Localization::CalculateDistance(mCenter, feature) < mRadius) ? true : false;
        }

        bool PresentInAll()
        {
            return all_of(mPresenceRooms.begin(), mPresenceRooms.end(),
                [](bool x) {return x; }) ? true : false;
        }

        void Display()
        {
            for (int i = 0; i < mConfig.GetRoomCount(); ++i)
            {
                cout << mPresenceRooms[i];
            }
            cout << endl;
        }

        void RemoveRoomPresence(const string& room)
        {
            mPresenceRooms.erase(mPresenceRooms.begin() + mConfig.GetRoomIndex(room));
        }

        // Vote using inverse document frequency 
        vector<float> Vote()
        {
            vector<float> scores(mConfig.GetRoomCount(), 0);
            int roomsSeen = 0;
            for (size_t i = 0; i < mConfig.GetRoomCount(); ++i)
            {
                roomsSeen += mPresenceRooms[i] ? 1 : 0;
            }

            if (roomsSeen > 0)
            {
                for (size_t i = 0; i < scores.size(); ++i)
                {
                    scores[i] = mPresenceRooms[i] ? (log(mConfig.GetRoomCount() * 1.0 / (float)roomsSeen) / log(mConfig.GetRoomCount())) : 0;
                }
            }

            //if (roomsSeen == 1)
            //{
            //    for (size_t i = 0; i < scores.size(); i++)
            //    {
            //        scores[i] = mPresenceRooms[i] ? 1 : 0;
            //    }
            //}
            //DEBUG
            //for (auto& x : scores)
            //{
            //    cout << x << ",";
            //}
            //cout << endl;
            //END_DEBUG
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

    static float CosDistance(const cv::Mat &testFeature, const cv::Mat &trainFeature)
    {
        float a = trainFeature.dot(testFeature);
        float b = trainFeature.dot(trainFeature);
        float c = testFeature.dot(testFeature);
        return 1 - a / sqrt(b*c);
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
        if (x.cols == DIM_FREE || x.cols != DIM_COLOR_HIST) //  Sift features
        {
            //cout << compareHist(x, y, CV_COMP_KL_DIV);
            Mat diff(y.size(), y.type());
            Mat temp(y.size(), y.type());
            if (y.type() != x.type())
            {
                x.convertTo(temp, y.type());
                diff = temp - y;
                //cout << temp << endl << x << endl;
            }
            else
            {
                diff = y - x;
            }
            float dist = 0;
            //cout << diff.type() << endl;
            if (USE_FREE)
            {
                if (BINARY_NORM)
                {
                    dist = norm(diff, NORM_HAMMING);
                    //dist = norm(diff, NORM_L2);
                    //DEBUG = true;
                    if (DEBUG)
                    {
                        cout << dist << endl;
                    }
                    //dist = CosDistance(temp, y);
                }
                else
                {
                    dist = norm(diff, NORM_L2);
                }
                //cout << diff << endl << dist << endl << endl;
            }
            else
            {
                dist = norm(diff, NORM_L2);
            }

            return dist;
        }
        else if (x.cols == DIM_COLOR_HIST)
        {
            //TODO: implement or find diffusion distance as mentioned in Filliat 08 [27]
            //return compareHist(x, y, CV_COMP_CHISQR);  
            // lower the closer. CV_COMP_BHATTACHARYYA possible.

            float dist = (mNorm == NORM_KL) ? compareHist(x, y, CV_COMP_KL_DIV) : DiffusionDistance(x, y, 1.0);
            //float dist = DiffusionDistance(x, y, 1.2);
            //cout << dist << endl;
            return dist;  // lower the closer. CV_COMP_BHATTACHARYYA possible.
        }
    }

}


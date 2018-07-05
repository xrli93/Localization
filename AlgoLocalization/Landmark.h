#pragma once
// Referenced https://codereview.stackexchange.com/questions/146423/c-stl-graph-implementation-with-out-and-in-adjacency-list

#include "Node.h"
#include "Constants.h"
#include "Path.h"
//using namespace std;
//using namespace cv;
//using namespace Localization;
namespace Localization
{
    template <typename T>
    // T the key for vertex. Can be just int
    class MapNode
    {
    private:
        map<T, Edge> mInEdges; // Paths to self
        map<T, Edge> mOutEdges; // Paths from self

        friend class cereal::access;
        template<class Archive>
        void serialize(Archive & archive)
        {
            archive(mInEdges, mOutEdges);
        }
    public:
        MapNode() {};

        ~MapNode() {};

        // Add path from source to self
        void AddPathFrom(const T& source, const float& length = 1, const float& orientation = 0)
        {
            Edge lPath = Edge(length, orientation);
            if (mInEdges.find(source) == mInEdges.end())
            {
                mInEdges.insert(std::make_pair(source, lPath));
            }
            else
            {
                std::string errorMessage = std::string("try to insert a link that already exist");
                throw std::invalid_argument(errorMessage);
            }
        }

        // Add path from self to dest
        void AddPathTo(const T& dest, const float& length = 1, const float& orientation = 0)
        {
            Edge lPath = Edge(length, orientation);
            if (mOutEdges.find(dest) == mOutEdges.end())
            {
                mOutEdges.insert(std::make_pair(dest, lPath));
            }
            else
            {
                std::string errorMessage = std::string("try to insert a link that already exist");
                throw std::invalid_argument(errorMessage);
            }
        }

        float GetDistanceTo(T dest)
        {
            if (mOutEdges.find(dest) != mOutEdges.end())
            {
                return mOutEdges[dest].GetLength();
            }
            else
            {
                return -1;
            }
        }

        bool IsAdjacentTo(const T& landmark)
        {
            return (mInEdges.find(landmark) != mInEdges.end()) || (mOutEdges.find(landmark) != mOutEdges.end());
        }

        map<T, Edge> GetInEdges() const
        {
            return mInEdges;
        }

        map<T, Edge> GetOutEdges() const
        {
            return mOutEdges;
        }
    };


    class Landmark : public MapNode<int>
    {
    public:
        // Base members
            //map<int, Edge> mInEdges; // Paths to self
            //map<int, Edge> mOutEdges; // Paths from self

        vector<Mat> mDescriptors;
        vector<float> mAngles;
        string mRoom; // In which room!

        friend class cereal::access;
        template<class Archive>
        void serialize(Archive & archive)
        {
            archive(cereal::base_class<MapNode<int>>(this), mDescriptors, mAngles, mRoom);
        }
    public:
        Landmark()
        {
            mDescriptors = vector<Mat>();
            mAngles = vector<float>();
        }

        Landmark(vector<Mat> iDescriptors, vector<float> iAngles, string iRoom) :
            mDescriptors(iDescriptors), mAngles(iAngles), mRoom(iRoom) {};

        const vector<Mat>& GetDescriptors()
        {
            return mDescriptors;
        }

        const vector<float>& GetAngles()
        {
            return mAngles;
        }

        void AddDescriptor(const Mat& iDescriptors, const float& iAngle)
        {
            mDescriptors.push_back(iDescriptors);
            mAngles.push_back(iAngle);
        }

        float GetDistanceTo(int iLandmark)
        {
            return MapNode<int>::GetDistanceTo(iLandmark);
        }

    };

    class Room : public MapNode<string>
    {
    public:
        map<string, int> mKeyLandmarks; // Landmarks that lead to another room

        void AddKeyLandmark(string iRoom, int iLandmark)
        {
            // Only one keylandmark to one room
            if (mKeyLandmarks.find(iRoom) == mKeyLandmarks.end())
            {
                mKeyLandmarks[iRoom] = iLandmark;
            }
        }

        int GetKeyLandmarkToRoom(string iRoom)
        {
            auto lIter = mKeyLandmarks.find(iRoom);
            if (lIter != mKeyLandmarks.end())
            {
                return lIter->second;
            }
            else
            {
                return -1;
            }
        }
    };
}




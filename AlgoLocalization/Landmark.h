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
    class Landmark
    {
    private:
        map<T, Path> mInEdges; // Paths to self
        map<T, Path> mOutEdges; // Paths from self
    public:
        Landmark() {};

        ~Landmark() {};

        // Add path from source to self
        void AddPathFrom(const T& source, const float& length = 1, const float& orientation = 0)
        {
            Path lPath = Path(length, orientation);
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
            Path lPath = Path(length, orientation);
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

        bool IsAdjacentTo(const T& landmark)
        {
            return (mInEdges.find(landmark) != mInEdges.end()) || (mOutEdges.find(landmark) != mOutEdges.end());
        }
    };

    //template <typename T>
    //class Room : Landmark<string>
    //{
    //protected:
    //    string mName;
    //public:
    //    Room(string name) : mName(name) {};

    //    ~Room() {};
    //};


}

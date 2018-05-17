#pragma once
#include "Landmark.h"
#include "Constants.h"
namespace Localization
{
    class TopoMap
    {
    private:
        map<int, Landmark<int>> mLandmarks; // number indexed. perhaps better to be contained in rooms? No!
        map<string, Landmark<string>> mRooms; // name indexed

    public:
        void AddLandmark(int label, Landmark<int> landmark)
        {
            mLandmarks.insert(make_pair(label, landmark));
        }
        
        void RemoveLandmark(int label)
        {
            mLandmarks.erase(label);
        }
        
        void AddRoom(string label, Landmark<string> room)
        {
            mRooms.insert(make_pair(label, room)); // handles existing case automatically
        }
        
        void RemoveRoom(string roomName)
        {
            mRooms.erase(roomName);
        }


        void AddRoom(string roomName)
        {
            Landmark<string> lRoom;
            AddRoom(roomName, lRoom);
        }

        void AddRoomConnection(string iRoom, string oRoom)
        {
            auto iRoomIter = mRooms.find(iRoom);
            if (iRoomIter == mRooms.end())
            {
                AddRoom(iRoom);
                iRoomIter = mRooms.find(iRoom);
            }
            auto oRoomIter = mRooms.find(oRoom);
            if (oRoomIter == mRooms.end())
            {
                AddRoom(oRoom);
                oRoomIter = mRooms.find(oRoom);
            }
            try
            {
                iRoomIter->second.AddPathTo(oRoom);
                oRoomIter->second.AddPathFrom(iRoom);
            }
            catch (const std::exception& e)
            {
                cout << e.what() << endl;
            }
        }

        bool IsConnected(const string& room1, const string& room2)
        {
            auto roomIter1 = mRooms.find(room1);
            return roomIter1->second.IsAdjacentTo(room2);
        }
        
    public:

        TopoMap() {};

        ~TopoMap() {};

    };
}

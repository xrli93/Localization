#pragma once
#include "Landmark.h"
#include "Constants.h"
namespace Localization
{

    using vMatches = vector<vector<DMatch>>;
    class Path
    {
    public:
        vector<int> mLandmarks; // landmarks in the path
        vector<bool> mActive; // if they are active
        int mCurrentLandmark = 0;
        int mCurrentStep = 0;

        friend class cereal::access;
        template<class Archive>
        void serialize(Archive & archive)
        {
            archive(mLandmarks, mActive, mCurrentLandmark);
        }
    public:
        Path() {};

        // Add landmark to end
        int AddLandmark(int iLandmark)
        {
            // Adding two times same landmark
            if (find(mLandmarks.begin(), mLandmarks.end(), iLandmark) != mLandmarks.end())
            {
                return -1;
            }
            mLandmarks.push_back(iLandmark);
            mActive.push_back(false);
            return 0;
        }

        // Add landmark to beginning
        int InsertLandmark(int iLandmark)
        {
            // Adding two times same landmark
            if (find(mLandmarks.begin(), mLandmarks.end(), iLandmark) != mLandmarks.end())
            {
                return -1;
            }
            mLandmarks.insert(mLandmarks.begin(), iLandmark);
            mActive.insert(mActive.begin(), false);
            return 0;
        }

        int SetActive(int iStep)
        {
            if (iStep < mActive.size())
            {
                mActive[iStep] = true;
                return 0;
            }
            else
            {
                return -1;
            }
        }

        int SetInactive(int iStep)
        {
            if (iStep < mActive.size())
            {
                mActive[iStep] = false;
                return 0;
            }
            else
            {
                return -1;
            }
        }

        int CurrentStep()
        {
            return mCurrentStep;
        }

        int CurrentLandmark()
        {
            if (mCurrentStep >= 0 && mCurrentStep <= mLandmarks.size())
            {
                return mLandmarks[mCurrentStep];
            }
            return -1;
        }


        int NextLandmark()
        {
            return mCurrentLandmark + 1;
        }

        // SetActive iStep et iStep + 1
        // When starting, iStep = 0
        void UpdatePosition(int iStep)
        {
            mCurrentLandmark = iStep;
            if (iStep < mLandmarks.size())
            {
                //mActive[iStep] = true;
                //mActive[iStep + 1] = true;
                SetActive(iStep);
                SetActive(iStep + 1);
            }
            if (iStep > 0)
            {
                //mActive[iStep - 1] = false;
                SetInactive(iStep - 1);
            }
        }

        bool ContainsLandmark(int iLandmark)
        {
            return (find(mLandmarks.begin(), mLandmarks.end(), iLandmark) != mLandmarks.end());
        }

        bool ContainsPath(int iStart, int iEnd)
        {
            int lLength = mLandmarks.size();
            if (lLength > 0)
            {
                if (mLandmarks[0] == iStart && mLandmarks[lLength - 1] == iEnd)
                {
                    return true;
                }

            }
            return false;
        }

        bool ContainsPart(int iPrevious, int iNext)
        {
            for (size_t i = 0; i < mLandmarks.size() - 1; i++)
            {
                if (mLandmarks[i] == iPrevious && mLandmarks[i + 1] == iNext)
                {
                    return true;
                }
            }
            return false;
        }

    };

    class TopoMap
    {
    public:
        map<string, Room> mRooms; // name indexed
        map<int, Landmark> mLandmarks; // Landmarks in the room
        map<int, Path> mPaths; // Paths
        map<int, float> mTotalMatches; // Number of matches to landmarks
        //string mCurrentRoom = "";

        Ptr<Feature2D> f2d;
        Ptr<Feature2D> extract;
        BFMatcher matcher;

        friend class cereal::access;
        template<class Archive>
        void serialize(Archive & archive)
        {
            archive(mRooms, mLandmarks, mTotalMatches, mPaths);
        }

    public:
        TopoMap()
        {
            //f2d = BRISK::create(15, 5);
            f2d = BRISK::create(10, 5); // Better for localization
            extract = xfeatures2d::FREAK::create();
            matcher = BFMatcher(NORM_HAMMING);
        };

        ~TopoMap() {};

    public:
        auto AddRoom(string label, Room room)
        {
            return mRooms.insert(make_pair(label, room)); // handles existing case automatically
        }

        auto AddRoom(string roomName)
        {
            Room lRoom;
            return AddRoom(roomName, lRoom);
        }

        // Returns the landmark as well as the room index
        int FindRoomThenLandmark(const Mat& iImg, int* oRoom, int iNumImges = 6)
        {
            int lNumRooms = mConfig.GetRoomCount();
            static vector<float> lRoomVotes(lNumRooms, 0);
            static int trys = 0;

            std::vector<KeyPoint> keypoints;
            Mat lDescriptors;
            f2d->detect(iImg, keypoints);
            extract->compute(iImg, keypoints, lDescriptors);

            // TODO: potentially can be improved
            for (auto& x : mLandmarks)
            {
                auto lNumMatches = GetTotalMatches(lDescriptors, x.first);
                lRoomVotes[mConfig.GetRoomIndex(x.second.mRoom)] += lNumMatches;
                auto lItLdmkMatching = mTotalMatches.find(x.first);
                if (lItLdmkMatching != mTotalMatches.end())
                {
                    lItLdmkMatching->second = lNumMatches;
                }
            }
            if (++trys >= iNumImges)
            {
                trys = 0;
                int lRoom = distance(begin(lRoomVotes), max_element(begin(lRoomVotes), end(lRoomVotes)));
                for (auto& x : mTotalMatches)
                {
                    auto lLandmarkIter = mLandmarks.find(x.first);
                    if (mConfig.GetRoomName(lRoom).compare(lLandmarkIter->second.mRoom) != 0)
                    {
                        x.second = 0;
                    }
                }
                auto lMaxIter = std::max_element(mTotalMatches.begin(), mTotalMatches.end(),
                    [](const pair<int, float>& p1, const pair<int, float>& p2) {return p1.second < p2.second; });
                int lLandmark = (*lMaxIter).first;
                fill(lRoomVotes.begin(), lRoomVotes.end(), 0);
                *oRoom = lRoom;
                return lLandmark;
            }
            *oRoom = -1;
            return -2;
        }

        // simple test (and last test before use bag of words) // works with new parameters!
        int IdentifyRoom(const Mat& iImg, int iNumImges = NUM_MAX_IMAGES)
        {
            int lNumRooms = mConfig.GetRoomCount();
            static vector<float> lRoomVotes(lNumRooms, 0);
            static int trys = 0;
            shared_ptr<float> quality = make_shared<float>(0.);

            std::vector<KeyPoint> keypoints;
            Mat lDescriptors;
            f2d->detect(iImg, keypoints);
            extract->compute(iImg, keypoints, lDescriptors);

            // TODO: potentially can be improved
            for (auto& x : mLandmarks)
            {
                auto lNumMatches = GetTotalMatches(lDescriptors, x.first);
                cout << "Adding " << x.second.mRoom << " To " << mConfig.GetRoomIndex(x.second.mRoom) << " n matches " << lNumMatches << endl;
                lRoomVotes[mConfig.GetRoomIndex(x.second.mRoom)] += lNumMatches;
            }
            for (auto& x : lRoomVotes)
            {
                cout << x << endl;
            }
            cout << endl;
            //int lResult = CountVotes(lVotes, quality, 0.2);
            //if ((*quality >= 1) || ++trys >= iNumImges)
            if (++trys >= iNumImges)
            {
                trys = 0;
                int lResult = distance(begin(lRoomVotes), max_element(begin(lRoomVotes), end(lRoomVotes)));
                fill(lRoomVotes.begin(), lRoomVotes.end(), 0);
                return lResult;
            }
            return -2;
        }


        // Learn image in room iRoom at landmark iIndexLandmark
        void LearnOrientation(const Mat& img, float iOrientation,
            string iRoom, int iIndexLandmark)
        {
            // Calculate features
            std::vector<KeyPoint> lKeypoints;
            Mat lDescriptors;
            f2d->detect(img, lKeypoints);
            extract->compute(img, lKeypoints, lDescriptors);

            AddRoom(iRoom);

            auto lIterLandmark = mLandmarks.find(iIndexLandmark);
            // Creating landmark
            if (lIterLandmark == mLandmarks.end())
            {
                AddLandmark(iIndexLandmark, Landmark(vector<Mat> {lDescriptors}, vector<float> {iOrientation}, iRoom));
                mTotalMatches.insert(make_pair(iIndexLandmark, 0));
            }
            // Existing landmark
            else
            {
                lIterLandmark->second.AddDescriptor(lDescriptors, iOrientation);
            }
        }

        void AddLandmarkToPath(int iPath, int iLandmark)
        {
            auto lIterPath = mPaths.find(iPath);
            if (lIterPath == mPaths.end())
            {
                Path lPath = Path();
                lPath.AddLandmark(iLandmark);
                mPaths.insert(make_pair(iPath, lPath));
            }
            else
            {
                lIterPath->second.AddLandmark(iLandmark);
            }
        }

        void AddLandmark(int iIndexLandmark, Landmark landmark)
        {
            //int lIndex = mLandmarks.size() + 1;
            mLandmarks.insert(make_pair(iIndexLandmark, landmark));
        }

        // Add path from iLandmart start to end
        int UpdateLandmark(int iLandmarkStart, int iLandmarkEnd, float distance = 0, float orientation = 0)
        {
            auto lIterStart = mLandmarks.find(iLandmarkStart);
            auto lIterEnd = mLandmarks.find(iLandmarkEnd);
            if (lIterStart != mLandmarks.end() && lIterEnd != mLandmarks.end())
            {
                lIterStart->second.AddPathTo(iLandmarkEnd, distance, orientation);
                lIterEnd->second.AddPathFrom(iLandmarkStart, distance, -orientation);
                // inverse chemin
                lIterEnd->second.AddPathTo(iLandmarkStart, distance, -orientation);
                lIterStart->second.AddPathFrom(iLandmarkEnd, distance, -orientation);

            }
            else
            {
                return -1;
            }
            return 0;
        }

        int GetLandmarkCount() const
        {
            return mLandmarks.size();
        }

        void RemoveLandmark(int iLandmark)
        {
            mLandmarks.erase(iLandmark);
        }


        // Test using odometry if the iStep of iPath has been completed.
        // All should start from zero!
        bool ReachedNextLandmark(int iPath, int iStep, float iOdometry)
        {
            if (mPaths.find(iPath) != mPaths.end()) // If exists Path
            {
                Path lPath = mPaths[iPath];
                if (iStep < lPath.mLandmarks.size() - 1) // If step legal // No Last step!
                {
                    int lStart = lPath.mLandmarks[iStep];
                    int lEnd = lPath.mLandmarks[iStep + 1];

                    if (mLandmarks.find(lStart) != mLandmarks.end()) // If exists Landmark
                    {
                        return abs(iOdometry - mLandmarks[lStart].GetDistanceTo(lEnd)) < iOdometry * 0.25;
                    }
                }

            }
            return false;
        }

        void UpdatePathProgress(int iPath, int iStep)
        {
            if (mPaths.find(iPath) != mPaths.end())
            {
                mPaths[iPath].UpdatePosition(iStep);
            }
        }

        int GetLandmarkAtStep(int iPath, int iStep)
        {
            auto lIterPath = mPaths.find(iPath);
            if (lIterPath != mPaths.end())
            {
                if (iStep < lIterPath->second.mLandmarks.size())
                {
                    return lIterPath->second.mLandmarks[iStep];
                }
            }
            else
            {
                return -1;
            }
        }

        int FindLandmarkInPath(int iPath, int iLandmark)
        {
            auto lIterPath = mPaths.find(iPath);
            if (lIterPath != mPaths.end())
            {
                vector<int> lLandmarks = lIterPath->second.mLandmarks;
                auto lIterLandmark = find(lLandmarks.begin(), lLandmarks.end(), iLandmark);
                return distance(lLandmarks.begin(), lIterLandmark);
            }
            // Return -1 if not found 
            return -1;
        }

        // Find nearest landmark in path using une image
        int FindNearestLandmark(int iPath, const Mat& img, int nImgs = 1)
        {
            std::vector<KeyPoint> keypoints;
            Mat lDescriptors;
            f2d->detect(img, keypoints);
            extract->compute(img, keypoints, lDescriptors);

            Path lPath = mPaths[iPath];
            static int lNumImgs = 0;
            ++lNumImgs;
            if (mTotalMatches.size() == 0)
            {
                return 0;
            }
            else
            {
                for (auto& x : mTotalMatches)
                {
                    int lLandmark = x.first;
                    if (lPath.ContainsLandmark(lLandmark))
                    {
                        x.second = GetTotalMatches(lDescriptors, lLandmark);
                    }
                    else
                    {
                        x.second = 0;
                    }
                }
                if (lNumImgs == nImgs)
                {
                    lNumImgs = 0;
                    auto lMaxIter = std::max_element(mTotalMatches.begin(), mTotalMatches.end(),
                        [](const pair<int, float>& p1, const pair<int, float>& p2) {return p1.second < p2.second; });
                    return (*lMaxIter).first;
                }
                else
                {
                    return -1;
                }
            }
            // No need to flush mTotalMatches
        }


        // find the nearest landmark in room
        int FindNearestLandmark(string& iRoom, const Mat& img, int nImgs = 6)
        {
            std::vector<KeyPoint> keypoints;
            Mat lDescriptors;
            f2d->detect(img, keypoints);
            extract->compute(img, keypoints, lDescriptors);

            static int lNumImgs = 0;
            ++lNumImgs;
            if (mTotalMatches.size() == 0)
            {
                return 0;
            }
            else
            {
                for (auto& x : mTotalMatches)
                {
                    auto lRoomIter = mLandmarks.find(x.first);
                    if (iRoom.compare(lRoomIter->second.mRoom) == 0) // landmark in room
                    {
                        x.second = GetTotalMatches(lDescriptors, x.first);
                    }
                    else
                    {
                        x.second = 0;
                    }
                }
                if (lNumImgs == nImgs)
                {
                    lNumImgs = 0;
                    auto lMaxIter = std::max_element(mTotalMatches.begin(), mTotalMatches.end(),
                        [](const pair<int, float>& p1, const pair<int, float>& p2) {return p1.second < p2.second; });
                    return (*lMaxIter).first;
                }
                else
                {
                    return -1;
                }
            }
        }

        int FindNearestLandmarkDescriptors(int iPath, const Mat& iDescriptors, int nImgs = 1)
        {
            Path lPath = mPaths[iPath];
            static int lNumImgs = 0;
            ++lNumImgs;
            if (mTotalMatches.size() == 0)
            {
                return 0;
            }
            else
            {
                for (auto& x : mTotalMatches)
                {
                    int lLandmark = x.first;
                    if (lPath.ContainsLandmark(lLandmark))
                    {
                        x.second = GetTotalMatches(iDescriptors, lLandmark);
                    }
                    else
                    {
                        x.second = 0;
                    }
                }
                if (lNumImgs == nImgs)
                {
                    lNumImgs = 0;
                    auto lMaxIter = std::max_element(mTotalMatches.begin(), mTotalMatches.end(),
                        [](const pair<int, float>& p1, const pair<int, float>& p2) {return p1.second < p2.second; });
                    return (*lMaxIter).first;
                }
                else
                {
                    return -1;
                }
            }
        }


        float GetOrientation(const Mat& img, int iPath, int iStep, int* oConfidence = nullptr, int* oStep = nullptr)
        {
            // Calculate features
            std::vector<KeyPoint> keypoints;
            Mat lDescriptors;
            f2d->detect(img, keypoints);
            extract->compute(img, keypoints, lDescriptors);
            // visual landmark
            if (oStep != NULL)
            {
                *oStep = FindLandmarkInPath(iPath, FindNearestLandmarkDescriptors(iPath, lDescriptors, 1));
            }

            // Find path and the current landmark, update step
            auto lItPath = mPaths.find(iPath);
            if (lItPath == mPaths.end())
            {
                return -1;
            }
            auto lPath = lItPath->second;
            lPath.UpdatePosition(iStep);
            auto lCurrentLandmark = lPath.CurrentLandmark();
            auto lItLandmark = mLandmarks.find(lCurrentLandmark);

            // update landmark current angles for calculation
            if (lItLandmark == mLandmarks.end())
            {
                return -1;
            }
            else
            {
                for (auto& edge : lItLandmark->second.GetOutEdges())
                {
                    if (lPath.ContainsPart(lCurrentLandmark, edge.first))
                    {
                        lItLandmark->second.UpdateAngles(edge.second.GetOrientation());
                        break;
                    }
                }
            }

            auto lNMatch = 0;
            float lStdDev = 0;
            auto lAngle = GetOrientationToLandmark(lDescriptors, lCurrentLandmark, &lStdDev, &lNMatch);
            auto lCountFeatures = lItLandmark->second.CountFeatures();
            //auto lConfidence = (int)(lNMatch * 1500.0 / (lStdDev + 0.001F) / lCountFeatures);
            auto lConfidence = (int)(1000 * min(1.0f, (lNMatch * 1.8f / (lStdDev + 0.001F) / lCountFeatures)));


            if (oConfidence != nullptr)
            {
                *oConfidence = lConfidence;
            }
            return lAngle;
        }

        // Return the number of matches between an image and 
        // All the images in iLandmark
        int GetTotalMatches(const Mat& iDescriptors, int iLandmark = 0)
        {
            const float nnRatio = 0.75f;
            // Get the landmark
            auto lIter = mLandmarks.find(iLandmark);
            vector<Mat> lLandmarkDescriptors;
            if (lIter != mLandmarks.end())
            {
                lLandmarkDescriptors = lIter->second.GetDescriptors();
            }
            else
            {
                cout << "Not in dictionary" << endl;
                return NO_ORIENTATION;
            }

            // compare with each image in the landmark
            vector<int> nMatchList(lLandmarkDescriptors.size());
            for (int i = 0; i < lLandmarkDescriptors.size(); ++i)
            {
                double a = 0;
                int b = a;
                int nMatches = 0;
                vMatches nnMatches;
                Mat refDescriptor = lLandmarkDescriptors[i];
                if (refDescriptor.cols != 0)
                {
                    matcher.knnMatch(iDescriptors, refDescriptor, nnMatches, 2);
                    for (auto& x : nnMatches)
                    {
                        if (x[0].distance < nnRatio * x[1].distance)
                        {
                            ++nMatches;
                        }
                    }
                    nMatchList[i] = nMatches;
                }
                else
                {
                    nMatchList[i] = 0;
                }
            }

            // return the sum of matches 
            int lResult = 0;
            for (auto& x : nMatchList)
            {
                lResult += x;
            }
            return lResult;
        }

        // Calculate orientation wrt to a single landmark
        float GetOrientationToLandmark(const Mat& iDescriptors, int iLandmark, float* pStdDev, int* pNMatches)
        {
            // Find in map
            vector<Mat> lLandmarkDescriptors;
            vector<float> lLandmarkAngles;
            auto lIterLandmarks = mLandmarks.find(iLandmark);
            if (lIterLandmarks != mLandmarks.end())
            {
                lLandmarkDescriptors = lIterLandmarks->second.GetDescriptors();
                lLandmarkAngles = lIterLandmarks->second.GetAngles(); // assured to be parallel
            }
            else
            {
                return NO_ORIENTATION;
            }

            // Calculate local descriptors

            const float nnRatio = 0.75f;
            vector<int> nMatchList(lLandmarkDescriptors.size());
            for (int i = 0; i < lLandmarkDescriptors.size(); ++i)
            {
                int nMatches = 0;
                vMatches nnMatches;
                Mat refDescriptor = lLandmarkDescriptors[i];
                if (refDescriptor.cols != 0)
                {
                    matcher.knnMatch(iDescriptors, lLandmarkDescriptors[i], nnMatches, 2);
                    for (auto& x : nnMatches)
                    {
                        if (x[0].distance < nnRatio * x[1].distance)
                        {
                            ++nMatches;
                        }
                    }
                    nMatchList[i] = nMatches;
                }
                else
                {
                    nMatchList[i] = 0;
                }
            }

            // Find two best angles and their mean
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

            float bestAngle = lLandmarkAngles[bestIndex];
            float secondAngle = lLandmarkAngles[secondIndex];
            vector<float> lAngles{ bestAngle, secondAngle };

            float midAngle = Angles::CircularMean(lAngles, vector<float> {factor1, factor2});
            float stdDev = Angles::CircularStdDev(vector<float> {bestAngle, secondAngle}, vector<float> {(float)maxMatch, (float)secondMatch});
            cout << bestAngle << ", " << maxMatch << ", " << secondAngle << ',' << secondMatch << endl;
            cout << stdDev << endl;
            if (stdDev > MAX_STD_DEV || maxMatch < MIN_MATCH)
            {
                midAngle = NO_ORIENTATION;
            }

            *pStdDev = stdDev;
            *pNMatches = maxMatch + secondMatch;
            return midAngle;
        }

        //void SetCurrentRoom(string& room)
        //{
        //    mCurrentRoom = room;
        //}

        void AddLandmarkToRoom(string iRoom, int iIndexLandmark, Landmark iLandmark)
        {
            auto lRoomIter = mRooms.find(iRoom);
            if (lRoomIter == mRooms.end()) // No existing room
            {
                AddRoom(iRoom);
                lRoomIter = mRooms.find(iRoom);
            }
            iLandmark.mRoom = iRoom;
            AddLandmark(iIndexLandmark, iLandmark);
        }


        void RemoveRoom(string roomName)
        {
            mRooms.erase(roomName);
            //if (mCurrentRoom.compare(roomName) == 0)
            //{
            //    mCurrentRoom == "";
            //}
        }

        void AddRoomConnection(string iRoom, string oRoom)
        {
            auto lRoomIter = mRooms.find(iRoom);
            if (lRoomIter == mRooms.end())
            {
                AddRoom(iRoom);
                lRoomIter = mRooms.find(iRoom);
            }
            auto oRoomIter = mRooms.find(oRoom);
            if (oRoomIter == mRooms.end())
            {
                AddRoom(oRoom);
                oRoomIter = mRooms.find(oRoom);
            }
            try
            {
                lRoomIter->second.AddPathTo(oRoom);
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
            if (roomIter1 != mRooms.end())
            {
                return roomIter1->second.IsAdjacentTo(room2);
            }
            else return false;
        }

        // add landmark in iInRoom that leands to iDestRoom
        void AddKeyLandmark(string iInRoom, int iLandmark, string iDestRoom)
        {
            auto lRoomIter = mRooms.find(iInRoom);
            if (lRoomIter == mRooms.end())
            {
                lRoomIter = AddRoom(iInRoom).first;
            }
            auto oRoomIter = mRooms.find(iDestRoom);
            if (oRoomIter == mRooms.end())
            {
                oRoomIter = AddRoom(iDestRoom).first;
            }
            try
            {
                lRoomIter->second.AddKeyLandmark(oRoomIter->first, iLandmark);
            }
            catch (const std::exception& e)
            {
                cout << e.what() << endl;
            }
        }

        // from current position (iLandmark) to the room
        int FindPathToRoom(int iLandmark, string iRoom)
        {
            auto lItLandmark = mLandmarks.find(iLandmark);
            if (lItLandmark != mLandmarks.end()) // found landmark
            {
                string lCurrentRoom = lItLandmark->second.mRoom;
                auto lItRoom = mRooms.find(iRoom);
                if (lItRoom != mRooms.end()) // found room
                {
                    int iEndLandmark = lItRoom->second.GetKeyLandmarkToRoom(lCurrentRoom);
                    if (iEndLandmark != -1)
                    {
                        return FindPathBetweenLandmarks(iLandmark, iEndLandmark);
                    }
                    else
                    {
                        for (auto& x : mLandmarks)
                        {
                            if (x.second.mRoom.compare(iRoom) == 0)
                            {
                                return FindPathBetweenLandmarks(iLandmark, x.first);
                            }
                        }
                        //return FindPathBetweenLandmarks(iLandmark, )
                    }
                }
                else
                {
                    return -1; // No room
                }
            }
            else
            {
                return -1; // No iLandmark
            }
        }

        // A simple Dijkstra like path-finding
        int FindPathBetweenLandmarks(int iStart, int iEnd)
        {
            auto lItStart = mLandmarks.find(iStart);
            auto lItEnd = mLandmarks.find(iEnd);

            // check existing paths
            for (auto& path : mPaths)
            {
                if (path.second.ContainsPath(iStart, iEnd))
                {
                    return path.first;
                }
            }

            // construct a new path
            if (lItStart != mLandmarks.end() && lItEnd != mLandmarks.end())
            {
                Path lPath;
                lPath.AddLandmark(iEnd);
                // Can be optimized
                map<int, float> lFrontierLandmarks{ make_pair(lItStart->first, 0) };
                map<int, float> lLandmarkDists{ make_pair(lItStart->first, 0) };
                // shortest path parents
                map<int, int> lPathNodes;
                bool lFound = false;
                while (!lFound)
                {
                    for (auto& node : lFrontierLandmarks)
                    {
                        Landmark lLandmark = mLandmarks[node.first];
                        float lPathLength = lLandmarkDists[node.first]; // path length up to this node
                        for (auto& edge : lLandmark.GetOutEdges())
                        {
                            float lEdgeLength = edge.second.GetLength(); // edge length
                            float lNewPathLength = lEdgeLength + lPathLength;
                            auto lItDest = lLandmarkDists.find(edge.first);
                            if (lItDest == lLandmarkDists.end()) // Node first visited
                            {
                                //cout << "Path to " << edge.first << " is " << lNewPathLength << endl;
                                lLandmarkDists.insert(make_pair(edge.first, lNewPathLength));
                                lFrontierLandmarks.insert(make_pair(edge.first, lNewPathLength));
                                lPathNodes[edge.first] = node.first;
                            }
                            else // possible updates
                            {
                                if (lItDest->second > lNewPathLength)
                                {
                                    lItDest->second = lNewPathLength;
                                    lFrontierLandmarks.insert(make_pair(edge.first, lItDest->second));
                                    lPathNodes[edge.first] = node.first;
                                    //cout << "Path to " << lItDest->first << " is " << lNewPathLength << endl;
                                }
                            }
                        }
                        lFrontierLandmarks.erase(node.first);
                    }
                    auto lIterEnd = lLandmarkDists.find(iEnd);

                    // Found shortest path
                    if (lIterEnd != lLandmarkDists.end())
                    {
                        int lNode = iEnd;
                        int lNextNode = lPathNodes[iEnd];
                        while (lNextNode != iStart)
                        {
                            lPath.InsertLandmark(lNextNode);
                            lNextNode = lPathNodes[lNextNode];
                        }
                        lPath.InsertLandmark(iStart);
                        //return lPath;

                        int lNPaths = mPaths.size();
                        mPaths.insert(make_pair(lNPaths, lPath));
                        return lNPaths;
                    }
                    // No path found
                    if (lFrontierLandmarks.size() == 0)
                    {
                        return -1;
                    }
                }
            }
            else
            {
                return -1;
            }
        }
    };
}

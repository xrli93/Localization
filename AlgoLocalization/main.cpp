#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include "Node.h"
#include "serial.h"
#include "TreeDict.h"
#include "Constants.h"
#include "Localizer.h"
#include "ImageLearner.h"
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "cereal\archives\portable_binary.hpp"
#include "cereal\archives\xml.hpp"
#include "matcerealisation.hpp"
#include "TopoMap.h"
using namespace std;
using namespace Localization;
using namespace cv;


string Convert(float number) {
    ostringstream buff;
    buff << number;
    return buff.str();
}

string GetFileName(string dir, int i)
{
    string filename;
    if (i < 10)
    {
        filename = dir + "00" + Convert(i) + ".jpg";
    }
    else if (i < 100)
    {
        filename = dir + "0" + Convert(i) + ".jpg";
    }
    else
    {
        filename = dir + Convert(i) + ".jpg";
    }
    return filename;
}

class Tester
{
public:
    string root = "C:\\Users\\Hoth\\Pictures\\Buddy\\";
    string video = "C:\\Users\\Hoth\\Pictures\\Videos\\";
    string salon = "C:\\Users\\Hoth\\Pictures\\Buddy\\salon\\";
    string cuisine = "C:\\Users\\Hoth\\Pictures\\Buddy\\cuisine\\";
    string reunion = "C:\\Users\\Hoth\\Pictures\\Buddy\\reunion\\";
    vector<Mat> salonImgs;
    vector<Mat> cuisineImgs;
    vector<Mat> reunionImgs;
    vector<Mat> mangerImgs;
    vector<Mat> salonTest;
    vector<Mat> cuisineTest;
    vector<Mat> reunionTest;
    vector<Mat> mangerTest;
    int nLearning = 40;
    int nTest = 20;
    int nImgs = 1;
    int nExperiments = 1;
    Localizer mLocalizer{};
    TopoMap mMap{};
    Tester() {}

    Tester(int numLearning, int numTest, int numImgs) :
        nLearning(numLearning), nTest(numTest), nImgs(numImgs)
    {
    }

    void Train(vector<Mat>* imgs, string room)
    {
        //srand(time(0));
        //random_shuffle(imgs->begin(), imgs->end());
        cout << "Training No. ";
        for (size_t i = 0; i < nLearning; ++i)
        {
            mLocalizer.LearnImage((*imgs)[i], room, 0, (float)(30 + i));
            cout << i << ", ";
        }
        cout << endl;
        ReportDict();
    }
    void LearnImage(const Mat& img, string room)
    {

    }
    void ReportDict()
    {
        vector<int> wordsCount = mLocalizer.CountWords();
        vector<int> nodesCount = mLocalizer.CountNodes();
        vector<int> featuresCount = mLocalizer.CountFeatures();
        cout << "Words FREE " << wordsCount[0] << " color " << wordsCount[1] << endl;
        cout << "Nodes FREE " << nodesCount[0] << " color " << nodesCount[1] << endl;
        cout << "Features learnt FREE " << featuresCount[0] << " color " << featuresCount[1] << endl;
        cout << endl;

        if (mConfig.GetRoomCount() == 3)
        {
            vector<int> FREEAnalysis = mLocalizer.AnalyseDict(USE_FREE);
            vector<int> ColorAnalysis = mLocalizer.AnalyseDict(USE_COLOR);
            cout << "FREE Dict analysis: " << endl;
            for (auto i : FREEAnalysis)
            {
                cout << setw(4) << i << ", ";
            }
            cout << endl << endl;

            cout << "Color Dict analysis: " << endl;
            for (auto i : ColorAnalysis)
            {
                cout << setw(4) << i << ", ";
            }
            cout << endl << endl;
        }
    }

    void ReportIncremental(vector<Mat>& imgs, string room, vector<float>* stats, int offset = 0)
    {

        srand(time(0));
        random_shuffle(imgs.begin(), imgs.end());
        int correct = 0;
        int unIdentified = 0;
        float timings = 0;
        size_t i = 0;
        string result = "";
        int count = 0;
        while (i < imgs.size())
        {
            //cout << "Localisation start " << endl;
            auto t1 = std::chrono::high_resolution_clock::now();
            int trys = 0;
            bool halt = false;
            while (result == "" && i < imgs.size() && !halt)
            {
                trys++;
                result = mLocalizer.IdentifyRoom(imgs[i++], &halt);
            }
            auto t2 = std::chrono::high_resolution_clock::now();
            timings += (float)std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
            count++;

            cout << "Trys " << trys << endl;
            if (result == room)
            {
                cout << ((VERBOSE) ? "**** Correct **** \n" : "");
                ++correct;
            }
            else if (result == "")
            {
                cout << ((VERBOSE) ? "**** Unidentified **** \n" : "");
                ++unIdentified;
            }
            else
            {
                cout << ((VERBOSE) ? "result is: " + result + "\n" : "");
                cout << ((VERBOSE) ? "**** Wrong **** \n" : "");
            }
            result = "";
        }
        float percentCorrect = correct * 100.0 / count;
        float percentUnidentified = unIdentified * 100.0 / count;

        cout << "***********  Average timing: " << timings / count << endl;
        cout << "** ** ** **  Correct: " << percentCorrect << "%" << endl;
        cout << "* * * * * *  Unidentified: " << percentUnidentified << "%" << endl;
        cout << "*    *    *  Average imgs taken: " << (float)imgs.size() / count <<
            " with SECOND_VOTE_SEUIL " << THRESHOLD_SECOND_VOTE << endl;
        cout << endl;
    }



    Mat Resize(Mat img)
    {
        Mat lImg = Mat(240, 320, CV_8UC1);
        resize(img, lImg, lImg.size(), 0, 0, INTER_LINEAR);

        //GaussianBlur(lImg, lImg, Size(0, 0), 3);
        //addWeighted(lImg, 1.5, lImg, -0.5, 0, lImg);
        return lImg;
    }
    void ReadImages()
    {
        for (size_t i = 1; i <= TRAIN_SIZE; ++i)
        {
            salonImgs.push_back(imread(salonTrainPath + to_string(i) + ".jpg", IMREAD_COLOR));
            cuisineImgs.push_back(imread(cuisineTrainPath + to_string(i) + ".jpg", IMREAD_COLOR));
            reunionImgs.push_back(imread(reunionTrainPath + to_string(i) + ".jpg", IMREAD_COLOR));
            mangerImgs.push_back(imread(mangerTrainPath + to_string(i) + ".jpg", IMREAD_COLOR));
        }
        for (size_t i = 1; i <= TEST_SIZE; ++i)
        {
            salonTest.push_back(imread(salonTestPath + to_string(i) + ".jpg", IMREAD_COLOR));
            cuisineTest.push_back(imread(cuisineTestPath + to_string(i) + ".jpg", IMREAD_COLOR));
            //salonTest.push_back(imread(salonTrainPath + to_string(i) + ".jpg", IMREAD_COLOR));
            //cuisineTest.push_back(imread(cuisineTrainPath + to_string(i) + ".jpg", IMREAD_COLOR));
            reunionTest.push_back(imread(reunionTestPath + to_string(i) + ".jpg", IMREAD_COLOR));
            mangerTest.push_back(imread(mangerTestPath + to_string(i) + ".jpg", IMREAD_COLOR));
        }
    }

    void ReportParams()
    {
        cout << "RADIUS_FREE: " << RADIUS_FREE << endl;
        cout << "FREE" << USE_FREE << endl;
        cout << "SYM" << USE_SYMMETRY << endl;
    }

    void Run(vector<float>* results, bool enableCorrection, vector<float>* secondResults)
    {
        stringstream ss;
        string filename;
        Mat img;

        ReadImages();

        string salon = "Salon";
        string cuisine = "Cuisine";
        string reunion = "Reunion";
        //mMap.AddRoom(salon);
        //mMap.AddRoom(cuisine);

        //mMap.AddRoomConnection(salon, cuisine);
        //mMap.AddRoomConnection(salon, reunion);

        cout << "Read all imgs" << endl;
        for (size_t i = 0; i < nExperiments; ++i)
        {

            string filename = "D:\\WorkSpace\\03_Resources\\Dataset\\v2\\dict.bin";
            string configPath = "D:\\WorkSpace\\03_Resources\\Dataset\\v2\\config.bin";

            if (READ_CEREAL)
            {
                // Read data
                auto t1 = std::chrono::high_resolution_clock::now();
                {
                    ifstream ifs(filename, ios::binary);
                    cereal::PortableBinaryInputArchive iarchive(ifs);
                    iarchive(mLocalizer);
                }
                {
                    ifstream ifs(configPath, ios::binary);
                    cereal::PortableBinaryInputArchive iarchive(ifs);
                    iarchive(mConfig);
                }
                auto t2 = std::chrono::high_resolution_clock::now();
                double timings = (float)std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
                cout << "Model loaded " << timings << endl;
            }
            else
            {
                // Training and saving dict
                string salon = "Salon";
                string cuisine = "Cuisine";
                string reunion = "Reunion";
                string manger = "Manger";
                mLocalizer.AddRoom(salon);
                mLocalizer.AddRoom(cuisine);
                mLocalizer.AddRoom(reunion);
                //mLocalizer.AddRoom(manger);

                //Train(&salonImgs, salon);
                //Train(&salonImgs, salon);
                //Train(&salonImgs, salon);
                Train(&cuisineImgs, cuisine);
                Train(&reunionImgs, reunion);
                DEBUG = false;
                //Train(&reunionImgs, reunion);
                //Train(&cuisineImgs, cuisine);
                //Train(&reunionImgs, reunion);
                //Train(&cuisineImgs, cuisine);
                //Train(&reunionImgs, reunion);
                //Train(&mangerImgs, manger);
                //Train(&reunionImgs, reunion);

                //ReportDict();
                cout << " Training done " << endl;
                cout << endl << endl;

                auto t1 = std::chrono::high_resolution_clock::now();
                {
                    ofstream ofs(filename, ios::binary);
                    cereal::PortableBinaryOutputArchive oarchive(ofs);
                    oarchive(mLocalizer);
                }
                {
                    ofstream ofs(configPath, ios::binary);
                    cereal::PortableBinaryOutputArchive oArchive(ofs);
                    oArchive(mConfig);
                }
                auto t2 = std::chrono::high_resolution_clock::now();
                double timings = (float)std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
                cout << "Model saved" << timings << endl;
            }

            // Testing results
            {
                string salon = "Salon";
                string cuisine = "Cuisine";
                string reunion = "Reunion";
                //string manger = "Manger";
                //mLocalizer.RemoveRoom(salon);
                ////ReportIncremental(salonTest, salon, results);
                //DEBUG = true;
                //ReportIncremental(salonTest, salon, results);

                ReportIncremental(cuisineTest, cuisine, results);
                //Train(&salonImgs, salon);
                //ReportIncremental(salonTest, salon, results);
                //Train(&reunionImgs, reunion);
                ReportIncremental(reunionTest, reunion, results);

                //ReportIncremental(mangerTest, manger, results);
                //Train(&reunionImgs, reunion);
                //ReportIncremental(reunionTest, reunion, results);

                //Train(&salonImgs, salon);
                //ReportIncremental(reunionTest, reunion, results);
                //ReportIncremental(mangerTest, manger, results);
            }
            //ReportResults(mangerTest, MANGER, results);
            //cout << mLocalizer.IsConnected(cuisine, salon);
            //cout << mLocalizer.IsConnected(cuisine, reunion);
            //cout << mLocalizer.IsConnected(salon, reunion);
            ReportDict();
            cout << endl << endl;

        }

    }

};

class NewTester
{
public:
    vector<Mat> salonImgs;
    vector<Mat> cuisineImgs;
    vector<Mat> reunionImgs;
    vector<Mat> mangerImgs;
    vector<Mat> salonTest;
    vector<Mat> cuisineTest;
    vector<Mat> reunionTest;
    vector<Mat> mangerTest;
    int nLearning = 40;
    int nTest = 20;
    int nImgs = 1;
    int nExperiments = 1;
    Localizer mLocalizer{};
    TopoMap mMap{};
    Ptr<Feature2D> f2d;
    Ptr<Feature2D> extract;
    BFMatcher matcher;
    map<string, vector<Mat>> mDescriptors;


    NewTester(int numLearning, int numTest, int numImgs) :
        nLearning(numLearning), nTest(numTest), nImgs(numImgs)
    {
        f2d = BRISK::create(15, 5);
        extract = xfeatures2d::FREAK::create();
        matcher = BFMatcher(NORM_HAMMING);
    }

    void Train(vector<Mat>* imgs, string room)
    {
        //srand(time(0));
        //random_shuffle(imgs->begin(), imgs->end());
        cout << "Training No. ";
        for (size_t i = 0; i < salonImgs.size(); ++i)
        {
            //mLocalizer.LearnImage((*imgs)[i], room, 0, (float)(30 + i));
            Train((*imgs)[i], room);
            cout << i << ", ";
        }
        cout << endl;
    }

    void Train(const Mat& img, string room)
    {
        Mat lDescriptors;
        vector<KeyPoint> lKeypoints;
        f2d->detect(img, lKeypoints);
        if (lKeypoints.size() != 0)
        {
            extract->compute(img, lKeypoints, lDescriptors);
        }
        auto lIter = mDescriptors.find(room);
        if (lIter == mDescriptors.end())
        {
            mDescriptors.insert(make_pair(room, vector<Mat> {lDescriptors}));
        }
        else
        {
            lIter->second.push_back(lDescriptors);
        }

    }
    string TestIdentify(const Mat& img, bool* finished)
    {
        return IdentifyRoom(img, finished);
    }

    string IdentifyRoom(const Mat& img, bool* finished)
    {
        static int trys = 0;
        std::vector<KeyPoint> keypoints;
        Mat lDescriptors;
        f2d->detect(img, keypoints);
        extract->compute(img, keypoints, lDescriptors);

        const float nnRatio = 0.75f;
        static vector<float> lVotes(mDescriptors.size(), 0);
        for (auto& x : mDescriptors) // Rooms
        {
            vector<Mat> lRoomDescriptors = x.second;
            string lRoom = x.first;
            for (size_t i = 0; i < lRoomDescriptors.size(); i++) // Features in room
            {
                int nMatches = 0;
                vector<vector<DMatch>> nnMatches;
                Mat refDescriptor = lRoomDescriptors[i];
                if (refDescriptor.cols != 0)
                {
                    matcher.knnMatch(lDescriptors, refDescriptor, nnMatches, 2);
                    for (auto& y : nnMatches)
                    {
                        if (y[0].distance < nnRatio * y[1].distance)
                        {
                            ++nMatches;
                        }
                    }
                    lVotes[mConfig.GetRoomIndex(lRoom)] += nMatches;
                }
                else
                {
                    lVotes[mConfig.GetRoomIndex(lRoom)] += 0;
                }
            }
        }
        for (size_t i = 0; i < lVotes.size(); i++)
        {
            cout << lVotes[i] << endl;
            //cout << mConfig.GetRoomName(i) << ", " << lVotes[i] << endl;
        }
        cout << endl;
        shared_ptr<float> quality = make_shared<float>(0);
        int lResult = CountVotes(lVotes, quality, 0.1);
        if (*quality >= THRESHOLD_SECOND_VOTE_NEW || ++trys >= NUM_MAX_IMAGES)
        {
            fill(lVotes.begin(), lVotes.end(), 0);
            trys = 0;
            *finished = true;
            return mConfig.GetRoomName(lResult);
        }
        return "";
    }



    void ReportIncremental(vector<Mat>& imgs, string room, vector<float>* stats, int offset = 0)
    {

        srand(time(0));
        random_shuffle(imgs.begin(), imgs.end());
        int correct = 0;
        int unIdentified = 0;
        float timings = 0;
        size_t i = 0;
        string result = "";
        int count = 0;
        while (i < imgs.size())
        {
            //cout << "Localisation start " << endl;
            auto t1 = std::chrono::high_resolution_clock::now();
            int trys = 0;
            bool halt = false;
            while (result == "" && i < imgs.size() && !halt)
            {
                trys++;
                result = TestIdentify(imgs[i++], &halt);
            }
            auto t2 = std::chrono::high_resolution_clock::now();
            timings += (float)std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
            count++;

            cout << "Trys " << trys << endl;
            if (result == room)
            {
                cout << ((VERBOSE) ? "**** Correct **** \n" : "");
                ++correct;
            }
            else if (result == "")
            {
                cout << ((VERBOSE) ? "**** Unidentified **** \n" : "");
                ++unIdentified;
            }
            else
            {
                cout << ((VERBOSE) ? "result is: " + result + "\n" : "");
                cout << ((VERBOSE) ? "**** Wrong **** \n" : "");
            }
            result = "";
        }
        float percentCorrect = correct * 100.0 / count;
        float percentUnidentified = unIdentified * 100.0 / count;

        cout << "***********  Average timing: " << timings / count << endl;
        cout << "** ** ** **  Correct: " << percentCorrect << "%" << endl;
        cout << "* * * * * *  Unidentified: " << percentUnidentified << "%" << endl;
        cout << "*    *    *  Average imgs taken: " << (float)imgs.size() / count <<
            " with SECOND_VOTE_SEUIL " << THRESHOLD_SECOND_VOTE << endl;
        cout << endl;
    }


    void ReadImages()
    {
        //string root = "D:/WorkSpace/03_Resources/Dataset/Angle/";
        string root = "D:/WorkSpace/03_Resources/Dataset/Odometry/";
        for (size_t i = 1; i <= 16; ++i)
        {
            salonImgs.push_back(imread(root + "SalonTrain1/" + to_string(i) + ".jpg", IMREAD_GRAYSCALE));
            cuisineImgs.push_back(imread(root + "CuisineTrain1/" + to_string(i) + ".jpg", IMREAD_GRAYSCALE));
            reunionImgs.push_back(imread(root + "HallTrain1/" + to_string(i) + ".jpg", IMREAD_GRAYSCALE));
            //mangerImgs.push_back(imread(root + "manger + to_string(i) + ".jpg", IMREAD_GRAYSCALE));
        }
        //for (size_t i = 1; i <= 12; ++i)
        //{
        //    salonImgs.push_back(imread(root + "SalonTrain2/" + to_string(i) + ".jpg", IMREAD_COLOR));
        //    cuisineImgs.push_back(imread(root + "CuisineTrain2/" + to_string(i) + ".jpg", IMREAD_GRAYSCALE));
        //    reunionImgs.push_back(imread(root + "ReunionTrain2/" + to_string(i) + ".jpg", IMREAD_GRAYSCALE));
        //    //mangerImgs.push_back(imread(root + "manger + to_string(i) + ".jpg", IMREAD_GRAYSCALE));
        //}
        for (size_t i = 1; i <= 10; ++i)
        {
            //salonTest.push_back(imread(root + "SalonTest2/" + to_string(i) + ".jpg", IMREAD_GRAYSCALE));
            //cuisineTest.push_back(imread(root + "CuisineTest2/" + to_string(i) + ".jpg", IMREAD_COLOR));
            //reunionTest.push_back(imread(root + "ReunionTest2/" + to_string(i) + ".jpg", IMREAD_GRAYSCALE));

            salonTest.push_back(imread(root + "SalonTest1/" + to_string(i) + ".jpg", IMREAD_GRAYSCALE));
            cuisineTest.push_back(imread(root + "CuisineTest1/" + to_string(i) + ".jpg", IMREAD_GRAYSCALE));
            reunionTest.push_back(imread(root + "HallTest1/" + to_string(i) + ".jpg", IMREAD_GRAYSCALE));
        }
    }


    //void ReadImages()
    //{
    //    for (size_t i = 1; i <= TRAIN_SIZE; ++i)
    //    {
    //        salonImgs.push_back(imread(salonTrainPath + to_string(i) + ".jpg", IMREAD_COLOR));
    //        cuisineImgs.push_back(imread(cuisineTrainPath + to_string(i) + ".jpg", IMREAD_COLOR));
    //        reunionImgs.push_back(imread(reunionTrainPath + to_string(i) + ".jpg", IMREAD_COLOR));
    //        //mangerImgs.push_back(imread(mangerTrainPath + to_string(i) + ".jpg", IMREAD_COLOR));
    //    }
    //    for (size_t i = 1; i <= TEST_SIZE; ++i)
    //    {
    //        salonTest.push_back(imread(salonTestPath + to_string(i) + ".jpg", IMREAD_COLOR));
    //        cuisineTest.push_back(imread(cuisineTestPath + to_string(i) + ".jpg", IMREAD_COLOR));
    //        //salonTest.push_back(imread(salonTrainPath + to_string(i) + ".jpg", IMREAD_COLOR));
    //        //cuisineTest.push_back(imread(cuisineTrainPath + to_string(i) + ".jpg", IMREAD_COLOR));
    //        reunionTest.push_back(imread(reunionTestPath + to_string(i) + ".jpg", IMREAD_COLOR));
    //        //mangerTest.push_back(imread(mangerTestPath + to_string(i) + ".jpg", IMREAD_COLOR));
    //    }
    //}
    void ReportParams()
    {
        cout << "RADIUS_FREE: " << RADIUS_FREE << endl;
        cout << "FREE" << USE_FREE << endl;
        cout << "SYM" << USE_SYMMETRY << endl;
    }

    void Run(vector<float>* results, bool enableCorrection, vector<float>* secondResults)
    {
        stringstream ss;
        string filename;
        Mat img;

        ReadImages();

        string salon = "Salon";
        string cuisine = "Cuisine";
        string reunion = "Reunion";
        string hall = "Hall";
        //mMap.AddRoom(salon);
        //mMap.AddRoom(cuisine);

        //mMap.AddRoomConnection(salon, cuisine);
        //mMap.AddRoomConnection(salon, reunion);

        cout << "Read all imgs" << endl;
        for (size_t i = 0; i < nExperiments; ++i)
        {

            string filename = "D:\\WorkSpace\\03_Resources\\Dataset\\v2\\new_dict.bin";
            string configPath = "D:\\WorkSpace\\03_Resources\\Dataset\\v2\\new_config.bin";

            if (READ_CEREAL)
            {
                // Read data
                auto t1 = std::chrono::high_resolution_clock::now();
                {
                    ifstream ifs(filename, ios::binary);
                    cereal::PortableBinaryInputArchive iarchive(ifs);
                    iarchive(mLocalizer);
                }
                {
                    ifstream ifs(configPath, ios::binary);
                    cereal::PortableBinaryInputArchive iarchive(ifs);
                    iarchive(mConfig);
                }
                auto t2 = std::chrono::high_resolution_clock::now();
                double timings = (float)std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
                cout << "Model loaded " << timings << endl;
            }
            else
            {
                // Training and saving dict
                string salon = "Salon";
                string cuisine = "Cuisine";
                string reunion = "Reunion";
                string manger = "Manger";
                string hall = "Hall";
                mLocalizer.AddRoom(salon);
                mLocalizer.AddRoom(cuisine);
                //mLocalizer.AddRoom(reunion);
                mLocalizer.AddRoom(hall);
                //mLocalizer.AddRoom(manger);

                //Train(&salonImgs, salon);
                //Train(&salonImgs, salon);
                Train(&salonImgs, salon);
                Train(&cuisineImgs, cuisine);
                //Train(&reunionImgs, reunion);
                Train(&reunionImgs, hall);
                DEBUG = false;

                //ReportDict();
                cout << " Training done " << endl;
                cout << endl << endl;

                auto t1 = std::chrono::high_resolution_clock::now();
                {
                    ofstream ofs(filename, ios::binary);
                    cereal::PortableBinaryOutputArchive oarchive(ofs);
                    oarchive(mLocalizer);
                }
                {
                    ofstream ofs(configPath, ios::binary);
                    cereal::PortableBinaryOutputArchive oArchive(ofs);
                    oArchive(mConfig);
                }
                auto t2 = std::chrono::high_resolution_clock::now();
                double timings = (float)std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
                cout << "Model saved" << timings << endl;
            }

            // Testing results
            {
                string salon = "Salon";
                string cuisine = "Cuisine";
                string reunion = "Reunion";
                string hall = "Hall";
                ReportIncremental(salonTest, salon, results);
                ReportIncremental(cuisineTest, cuisine, results);
                ReportIncremental(reunionTest, hall, results);

            }
            cout << endl << endl;

        }

    }

};

class OrientationTester
{
public:
    string root = "D:/WorkSpace/03_Resources/Dataset/Angle/";
    //string root = "D:/WorkSpace/03_Resources/Dataset/Odometry/";
    //string root = "D:/WorkSpace/03_Resources/Dataset/Odometry3/";
    //vector<string> mRooms{ "Salon", "SalonNew", "Cuisine", "Hall" };
    //vector<string> mRooms{ "SalonNew", "Cuisine", "Hall" };
    vector<string> mRooms{ "Salon", "Hall" };
    vector<string> mTypes{ "Train", "Test" };
    vector<string> mSets{ "1","2" };
    map<string, vector<float>> mRefs;
    const double pi = 3.1415926; // radian or degree?
    map<int, vector<Mat>> mDescriptors;
    map<int, vector<float>> mAngles;
    map<int, vector<float>> mTotalMatches;

    Ptr<Feature2D> f2d;
    Ptr<Feature2D> extract;
    BFMatcher matcher;

    friend class cereal::access;
    template<class Archive>
    void serialize(Archive & archive)
    {
        archive(mTotalMatches);
    }

public:

    // Odometry3
    float GetAngle(vector<float>& iAngles, vector<float>& iStdDevs, vector<int>& iNMatches)
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
        cout << "If no filtering" << Angles::CircularMean(lAngles, lWeights) << endl;
        if (lAngles.size() == 0 || Angles::CircularStdDev(lAngles) > MAX_STD_DEV * 3) // More tolereant
        {
            return NO_ORIENTATION;
        }
        else
        {
            return Angles::CircularMean(lAngles, lWeights);
        }
    }

    // test matching results
    // To find distances between landmarks
    void TestLocalLandmark()
    {
        mRooms = vector<string>{ "Hall" };
        //root = "D:/WorkSpace/03_Resources/Dataset/Odometry3/";
        root = "D:/WorkSpace/03_Resources/Dataset/Landmarks/";
        root = "D:/WorkSpace/03_Resources/Dataset/Landmarks2/";
        std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
        std::cout.precision(3);

        f2d = BRISK::create(15, 5);
        //f2d = AgastFeatureDetector::create();
        extract = xfeatures2d::FREAK::create();
        //extract = f2d;
        matcher = BFMatcher(NORM_HAMMING);

        Localizer mLocalizer;
        //vector<int> nTrains{ 12,12,6 };
        int nTrain = 6;
        int nSet = 6;
        for (const string& room : mRooms)
        {
            cout << "Room: " << room << endl;
            mLocalizer.AddRoom(room);
            for (int set = 1; set <= nSet; set++)
            {
                LearnImgs("", nTrain, to_string(set), set);
            }
        }

        //vector<int> nTotalMatches(nSet * 3, 0);
        //root = "D:/WorkSpace/03_Resources/Dataset/Odometry3bis/";
        for (size_t i = 1; i <= nSet; i++)
        {
            mTotalMatches.insert(make_pair(i, vector<float>(nSet, 0)));
        }
        //nTotalMatches.insert(make_pair(1, vector<int>(nSet, 0)));
        //nTotalMatches.insert(make_pair(2, vector<int>(nSet, 0)));
        //nTotalMatches.insert(make_pair(3, vector<int>(nSet, 0)));
        //nTotalMatches.insert(make_pair(4, vector<int>(nSet, 0)));
        //nTotalMatches.insert(make_pair(5, vector<int>(nSet, 0)));
        for (int imageSet = 1; imageSet <= nSet; imageSet++)
        {
            for (size_t i = 1; i <= nTrain; i++)
            {
                cout << "Image: " << i << " Set: " << imageSet << endl;
                for (size_t iLandmark = 1; iLandmark <= nSet; iLandmark++)
                {
                    int lNMatches = GetTotalMatches("", i, "Train", to_string(imageSet), iLandmark);
                    //cout << "matches: " << lNMatches << endl;
                    mTotalMatches[imageSet][iLandmark - 1] += lNMatches;
                }

                ////Division by number of keypoints in the comparaison class
                //for (size_t iLandmark = 1; iLandmark <= nSet; iLandmark++)
                //{
                //    int nKeypoints = 0;
                //    for (auto& x : mDescriptors[iLandmark])
                //    {
                //        nKeypoints += x.rows;
                //    }
                //    mTotalMatches[imageSet][iLandmark - 1] /= (float)nKeypoints;
                //}
            }
            cout << endl;
        }
        cout << endl;
        for (auto& x : mTotalMatches)
        {
            for (auto& y : x.second)
            {
                cout << y << endl;
            }
            cout << endl;
        }
        mDescriptors.clear();
        mAngles.clear();
    }

    void GetLandmarkDistances(int nSet)
    {
        for (size_t imageSet = 1; imageSet <= (int)nSet / 2; imageSet++)
        {
            vector<float> lMatches = mTotalMatches[imageSet];
            auto maxIter = max_element(lMatches.begin() + (int)nSet / 2, lMatches.end());
            int maxIndex = distance(lMatches.begin(), maxIter) + 1;

            *maxIter = 0;
            auto secondLargestIter = max_element(lMatches.begin() + (int)nSet / 2, lMatches.end());
            int secondIndex = distance(lMatches.begin(), secondLargestIter) + 1;
            cout << "first: " << maxIndex << ", second: " << secondIndex << endl;
        }
        cout << endl;

        for (size_t imageSet = (int)nSet / 2 + 1; imageSet <= nSet; imageSet++)
        {
            vector<float> lMatches = mTotalMatches[imageSet];
            auto maxIter = max_element(lMatches.begin(), lMatches.begin() + (int)nSet / 2);
            int maxIndex = distance(lMatches.begin(), maxIter) + 1;

            *maxIter = 0;
            auto secondLargestIter = max_element(lMatches.begin(), lMatches.begin() + (int)nSet / 2);
            int secondIndex = distance(lMatches.begin(), secondLargestIter) + 1;

            cout << "first: " << maxIndex << ", second: " << secondIndex << endl;
        }
        cout << endl;

        cout << endl;
        for (auto& x : mTotalMatches)
        {
            for (auto& y : x.second)
            {
                cout << y << endl;
            }
            cout << endl;
        }
    }

    void TestMulti(int iLandmark)
    {
        mRooms = vector<string>{ "Salon", "Hall" };
        root = "D:/WorkSpace/03_Resources/Dataset/Odometry3/";
        std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
        std::cout.precision(3);

        f2d = BRISK::create(15, 5);
        //f2d = AgastFeatureDetector::create();
        extract = xfeatures2d::FREAK::create();
        //extract = f2d;
        matcher = BFMatcher(NORM_HAMMING);

        Localizer mLocalizer;
        vector<int> nTrains{ 12,12,6 };
        int nSet = 3;
        for (const string& room : mRooms)
        {
            cout << "Room: " << room << endl;
            mLocalizer.AddRoom(room);
            for (int set = 1; set <= 3; set++)
            {
                LearnImgs(room, nTrains[set - 1], to_string(set), set);
            }
            for (size_t i = 1; i <= 9; i++)
            {
                vector<int> nMatches(nSet, 0);
                vector<float> nStdDevs(nSet, 0);
                vector<float> lAngles(nSet, 0);
                for (int set = 1; set <= nSet; set++)
                {
                    cout << "Image: " << i << " Set: " << set << endl;
                    lAngles[set - 1] = GetOrientation(room, i, "Test", "1", set, &(nStdDevs[set - 1]), &(nMatches[set - 1]), true);
                }
                cout << "The Result of GetAngle is: " << GetAngle(lAngles, nStdDevs, nMatches) << endl;
                cout << endl;
            }
            cout << endl;
            mDescriptors.clear();
            mAngles.clear();
        }
    }

    void Run()
    {
        root = "D:/WorkSpace/03_Resources/Dataset/Angle/";
        mRooms = vector<string>{ "SalonNew", "Cuisine", "Hall", "Reunion" };
        std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
        std::cout.precision(3);
        //Test("1");
        //Test("2");

        //f2d = MSER::create();
        //extract = xfeatures2d::DAISY::create();

        //f2d = AgastFeatureDetector::create(15);
        f2d = BRISK::create(15, 5);
        extract = xfeatures2d::FREAK::create();

        //f2d = AKAZE::create();
        //extract = f2d;
        for (const string& room : mRooms)
        {
            matcher = BFMatcher(NORM_HAMMING);
            //matcher = BFMatcher(NORM_L2);
            cout << "Room: " << room << endl;
            LearnImgs(room, 12, "2");
            //LearnImgs(room, 12, "2");
            //LearnImgs(room, 6, "3");
            for (size_t i = 1; i <= 6; i++)
            {
                cout << "Image: " << i << endl;
                cout << "Angle: " << GetOrientation(room, i, "Train", "1") << endl;
                cout << endl;
            }
            mDescriptors.clear();
            mAngles.clear();
            cout << endl;
        }

    }
    int GetIndex(string room)
    {
        return distance(mRooms.begin(), find(mRooms.begin(), mRooms.end(), room));
    }

    float GetDegrees(float radian)
    {
        return radian / pi * 180;
    }

    void ReportDict(Localizer& mLocalizer)
    {
        vector<int> wordsCount = mLocalizer.CountWords();
        vector<int> nodesCount = mLocalizer.CountNodes();
        vector<int> featuresCount = mLocalizer.CountFeatures();
        cout << "Words FREE " << wordsCount[0] << " color " << wordsCount[1] << endl;
        cout << "Nodes FREE " << nodesCount[0] << " color " << nodesCount[1] << endl;
        cout << "Features learnt FREE " << featuresCount[0] << " color " << featuresCount[1] << endl;
        cout << endl;

        if (mConfig.GetRoomCount() == 3)
        {
            vector<int> FREEAnalysis = mLocalizer.AnalyseDict(USE_FREE);
            vector<int> ColorAnalysis = mLocalizer.AnalyseDict(USE_COLOR);
            cout << "FREE Dict analysis: " << endl;
            for (auto i : FREEAnalysis)
            {
                cout << setw(4) << i << ", ";
            }
            cout << endl << endl;

            cout << "Color Dict analysis: " << endl;
            for (auto i : ColorAnalysis)
            {
                cout << setw(4) << i << ", ";
            }
            cout << endl << endl;
        }
    }

    void LearnImgs(string room, int nTrain = 12, string set = "1", int iLandmark = 1)
    {
        vector<Mat> lDescriptors;
        vector<float> lAngles;
        for (size_t i = 1; i <= nTrain; i++)
        {
            string filename = root + room + "Train" + set + "/" + to_string(i) + ".jpg";
            Mat img = imread(filename, IMREAD_GRAYSCALE); // Read the file
            std::vector<KeyPoint> keypoints;
            Mat descriptors;
            f2d->detect(img, keypoints);
            extract->compute(img, keypoints, descriptors);
            lDescriptors.push_back(descriptors);
            lAngles.push_back((float)(i - 1) / nTrain * 360);
        }
        auto lIter = mDescriptors.find(iLandmark);
        auto lAngleIter = mAngles.find(iLandmark);
        if (lIter == mDescriptors.end()) // new landmark
        {
            mDescriptors.insert(make_pair(iLandmark, lDescriptors));
            mAngles.insert(make_pair(iLandmark, lAngles));
        }
        else
        {
            lIter->second.insert(lIter->second.end(), lDescriptors.begin(), lDescriptors.end());
            lAngleIter->second.insert(lAngleIter->second.end(), lAngles.begin(), lAngles.end());
        }
    }
    Mat Resize(Mat img)
    {
        Mat lImg = Mat(240, 320, CV_8UC1);
        resize(img, lImg, lImg.size(), 0, 0, INTER_LINEAR);

        //GaussianBlur(lImg, lImg, Size(0, 0), 3);
        //addWeighted(lImg, 1.5, lImg, -0.5, 0, lImg);
        return lImg;
    }

    // For image index in set, find the number of total matches in class K
    int GetTotalMatches(string room, int index, string type = "Test", string set = "1", int iLandmark = 1)
    {
        string filename = root + room + type + set + "/" + to_string(index) + ".jpg";
        Mat img = imread(filename, IMREAD_GRAYSCALE); // Read the file
        //img = Resize(img);
        std::vector<KeyPoint> keypoints;
        Mat lDescriptors;
        f2d->detect(img, keypoints);
        extract->compute(img, keypoints, lDescriptors);
        const float nnRatio = 0.75f;
        auto lIter = mDescriptors.find(iLandmark);
        vector<Mat> lLandmarkDescriptors;
        vector<float> lLandmarkAngles;
        if (lIter != mDescriptors.end())
        {
            lLandmarkDescriptors = lIter->second;
            lLandmarkAngles = mAngles[iLandmark];
        }
        else
        {
            cout << "Not in dictionary" << endl;
            return NO_ORIENTATION;
        }

        vector<int> nMatchList(lLandmarkDescriptors.size());
        for (int i = 1; i <= lLandmarkDescriptors.size(); ++i)
        {
            int nMatches = 0;
            vector<vector<DMatch>> nnMatches;
            Mat refDescriptor = lLandmarkDescriptors[i - 1];
            if (refDescriptor.cols != 0)
            {
                matcher.knnMatch(lDescriptors, refDescriptor, nnMatches, 2);
                for (auto& x : nnMatches)
                {
                    if (x[0].distance < nnRatio * x[1].distance)
                    {
                        ++nMatches;
                    }
                }
                nMatchList[i - 1] = nMatches;
            }
            else
            {
                nMatchList[i - 1] = 0;
            }
        }
        int lResult = 0;
        for (auto& x : nMatchList)
        {
            lResult += x;
        }
        return lResult;
    }

    float GetOrientation(string room, int index, string type = "Test", string set = "1", int iLandmark = 1,
        float* pStdDev = NULL, int* pNMatches = NULL, bool verbose = true)
    {
        string filename = root + room + type + set + "/" + to_string(index) + ".jpg";
        Mat img = imread(filename, IMREAD_GRAYSCALE); // Read the file
        //img = Resize(img);
        std::vector<KeyPoint> keypoints;
        Mat lDescriptors;
        f2d->detect(img, keypoints);
        extract->compute(img, keypoints, lDescriptors);
        const float nnRatio = 0.75f;
        auto lIter = mDescriptors.find(iLandmark);
        vector<Mat> lLandmarkDescriptors;
        vector<float> lLandmarkAngles;
        if (lIter != mDescriptors.end())
        {
            lLandmarkDescriptors = lIter->second;
            lLandmarkAngles = mAngles[iLandmark];
        }
        else
        {
            cout << "Not in dictionary" << endl;
            return NO_ORIENTATION;
        }

        vector<int> nMatchList(lLandmarkDescriptors.size());
        for (int i = 1; i <= lLandmarkDescriptors.size(); ++i)
        {
            int nMatches = 0;
            vector<vector<DMatch>> nnMatches;
            Mat refDescriptor = lLandmarkDescriptors[i - 1];
            if (refDescriptor.cols != 0)
            {
                matcher.knnMatch(lDescriptors, refDescriptor, nnMatches, 2);
                for (auto& x : nnMatches)
                {
                    if (x[0].distance < nnRatio * x[1].distance)
                    {
                        ++nMatches;
                    }
                }
                nMatchList[i - 1] = nMatches;
            }
            else
            {
                nMatchList[i - 1] = 0;
            }
        }

        // Find two best results
        vector<int>::iterator maxIter = max_element(nMatchList.begin(), nMatchList.end());
        int maxMatch = *maxIter;
        int bestIndex = distance(nMatchList.begin(), maxIter);
        *maxIter = 0; // set max vote to zero
        auto secondIter = max_element(nMatchList.begin(), nMatchList.end());
        int secondMatch = *secondIter;
        int secondIndex = distance(nMatchList.begin(), secondIter);
        *maxIter = maxMatch; // restore
        if (maxMatch == 0)
        {
            if (pStdDev != NULL)
            {
                *pStdDev = MAX_STD_DEV * 2;
            }
            return NO_ORIENTATION;
        }

        int sumMatches = maxMatch * maxMatch + secondMatch * secondMatch;
        float factor1 = (maxMatch * 1.0) / sumMatches;
        float factor2 = (secondMatch * 1.0) / sumMatches; // or square

        //float diffAngle = Angles::CircularMean(vector<float> {mAngles[bestIndex], -1 * mAngles[secondIndex]});
        //vector<float> lAngles{ factor1 * mAngles[bestIndex], factor2 * mAngles[secondIndex] };
        float bestAngle = lLandmarkAngles[bestIndex];
        float secondAngle = lLandmarkAngles[secondIndex];
        float diff = bestAngle - secondAngle;
        vector<float> lAngles{ bestAngle, secondAngle };
        float midAngle = Angles::CircularMean(lAngles);
        float midAngle2 = Angles::CircularMean(lAngles, vector<float> {factor1, factor2});
        float stdDev = Angles::CircularStdDev(vector<float> {bestAngle, secondAngle});
        float stdDevNew = Angles::CircularStdDev(vector<float> {bestAngle, secondAngle}, vector<float> {(float)maxMatch, (float)secondMatch});
        if (stdDevNew > MAX_STD_DEV || maxMatch < MIN_MATCH)
        {
            midAngle2 = NO_ORIENTATION;
        }
        if (pStdDev != NULL)
        {
            *pStdDev = stdDev;
            *pNMatches = maxMatch + secondMatch;
        }
        //cout << maxMatch << ", " << secondMatch <<  endl;
        if (verbose)
        {
            cout << "Total matchings: " << maxMatch + secondMatch << ", " << maxMatch << ", " << secondMatch << endl;
            cout << "StdDev: " << stdDev << ", " << bestAngle << ", " << secondAngle << ", " << midAngle << endl;
            cout << "StdDev new: " << stdDevNew << endl;
            cout << "Confidence: " << 1 / (stdDevNew + 0.001) << endl;
            //cout << "Confidence alt: " << (maxMatch + secondMatch) * sqrt((float)(maxMatch - secondMatch) / (maxMatch + secondMatch)) / (stdDevNew + 0.001) << endl;
            cout << "Confidence alt: " << (maxMatch + secondMatch) / (stdDevNew + 0.001) << endl;

        }
        //cout << "Total matchings: " << maxMatch + secondMatch << endl;
        //cout << stdDev << ", " << bestAngle << ", " << secondAngle << ", " << midAngle << "," << midAngle2 << endl;

        return midAngle2;
        //cout << "Mid: " << midAngle << ", weighted: " << midAngle2  <<  endl;
        //cout << mAngles[bestIndex];

            //if (nMatches > maxMatch)
            //{
            //    maxMatch = nMatches;
            //    bestMatch = (i);
            //}
            ////cout << nMatches << endl;
    }

};

class TopoMapTester
{

public:
    Localizer mLocalizer;
    TopoMap mMap;
    string root = "D:/WorkSpace/03_Resources/Dataset/Angle/";
    vector<string> mRooms{ "Salon", "Hall" };
    vector<string> mTypes{ "Train", "Test" };
    vector<string> mSets{ "1","2" };

    TopoMapTester() {};

    void Run()
    {
        string lRoot = "D:/WorkSpace/03_Resources/Dataset/Odometry/";
        Mat lMat = imread(root + "SalonTrain1/" + "1" + ".jpg", IMREAD_GRAYSCALE);
        Mat lMat2 = imread(root + "SalonTrain1/" + "2" + ".jpg", IMREAD_GRAYSCALE);
        cout << lMat.rows << ", " << lMat.cols << endl;
        string salon = "Salon";
        string hall = "Hall";
        string cuisine = "Cuisine";
        mLocalizer.LearnOrientation(lMat2, 30, mRooms[0], 0);
        mLocalizer.LearnOrientation(lMat2, 30, mRooms[0], 1);
        mLocalizer.LearnOrientation(lMat2, 30, mRooms[0], 2);
        mLocalizer.LearnOrientation(lMat2, 30, mRooms[0], 3);
        mLocalizer.LearnOrientation(lMat2, 30, mRooms[0], 4);
        mLocalizer.LearnOrientation(lMat2, 30, mRooms[1], 5);
        mLocalizer.LearnOrientation(lMat2, 30, mRooms[1], 6);
        mLocalizer.AddKeyLandmark(mRooms[1], 5, mRooms[0]);
        mLocalizer.AddKeyLandmark(mRooms[1], 6, mRooms[0]);
        mLocalizer.UpdateLandmarkDistance(0, 1, 2, 0);

        //Path lMPath = mLocalizer.mMap.mPaths[mLocalizer.FindPath(1, 0)];

        //for (auto& x : lMPath.mLandmarks)
        //{
        //    cout << x << endl;
        //}
        mLocalizer.UpdateLandmarkDistance(1, 0, 2, 0);
        mLocalizer.UpdateLandmarkDistance(0, 2, 1, 0);
        mLocalizer.UpdateLandmarkDistance(2, 3, 1, 0);
        mLocalizer.UpdateLandmarkDistance(3, 4, 1, 0);
        mLocalizer.UpdateLandmarkDistance(1, 4, 3, 0);
        mLocalizer.UpdateLandmarkDistance(4, 5, 3, 0);

        //Path lPath = mLocalizer.FindPath(0, 4);

        auto t1 = std::chrono::high_resolution_clock::now();
        int lIdPath = mLocalizer.FindPathToRoom(0, mRooms[1]);
        auto t2 = std::chrono::high_resolution_clock::now();
        double timings = (float)std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
        cout << "Id" << lIdPath << "first timing" << timings << endl;

        t1 = std::chrono::high_resolution_clock::now();
        lIdPath = mLocalizer.FindPathToRoom(0, mRooms[1]);
        t2 = std::chrono::high_resolution_clock::now();
        timings = (float)std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
        cout << "Id " << lIdPath << " second timing " << timings << endl;

        Path lPath = mLocalizer.mMap.mPaths[lIdPath];
        for (auto& x : lPath.mLandmarks)
        {
            cout << x << endl;
        }
    }

    void TestLocalization()
    {
        root = "D:/WorkSpace/03_Resources/Dataset/NewLocal/";
        mRooms = vector<string>{ "Salon", "Cuisine", "Hall" };
        for (auto& room : mRooms)
        {
            mConfig.AddRoomName(room);
            for (size_t i = 1; i <= 12; i++)
            {
                string filename = root + room + "Train/" + to_string(i) + ".jpg";
                Mat img = imread(filename, IMREAD_GRAYSCALE); // Read the file
                mMap.LearnOrientation(img, 0, room, mConfig.GetRoomIndex(room) * 7 + (int)(i) / 7);
            }
        }

        for (auto& x : mMap.mLandmarks)
        {
            cout << "Landmark nr: " << x.first << " in room " << x.second.mRoom << endl;
        }

        for (size_t i = 1; i <= 6; i++)
        {
            cout << "test nr. " << i << endl;
            string filename = root + "Test/" + to_string(i) + ".jpg";
            Mat img = imread(filename, IMREAD_GRAYSCALE); // Read the file
            int lRoom = -1;
            cout << "Nearest landmark is: " << mMap.FindRoomThenLandmark(img, &lRoom, 6);
            cout << ", in room " << mConfig.GetRoomName(lRoom) << endl;
        }

        for (size_t i = 7; i <= 11; i++)
        {
            cout << "test nr. " << i << endl;
            string filename = root + "Test/" + to_string(i) + ".jpg";
            Mat img = imread(filename, IMREAD_GRAYSCALE); // Read the file
            //cout << mMap.IdentifyRoom(img, 5) << endl;
            int lRoom;
            cout << "Nearest landmark is: " << mMap.FindRoomThenLandmark(img, &lRoom, 5);
            cout << ", in room " << mConfig.GetRoomName(lRoom) << endl;
        }

        for (size_t i = 12; i <= 16; i++)
        {
            cout << "test nr. " << i << endl;
            string filename = root + "Test/" + to_string(i) + ".jpg";
            Mat img = imread(filename, IMREAD_GRAYSCALE); // Read the file
            //cout << mMap.IdentifyRoom(img, 5) << endl;
            int lRoom = -1;
            cout << "Nearest landmark is: " << mMap.FindRoomThenLandmark(img, &lRoom, 5);
            cout << ", in room " << mConfig.GetRoomName(lRoom) << endl;
        }
        for (size_t i = 17; i <= 21; i++)
        {
            cout << "test nr. " << i << endl;
            string filename = root + "Test/" + to_string(i) + ".jpg";
            Mat img = imread(filename, IMREAD_GRAYSCALE); // Read the file
            //cout << mMap.IdentifyRoom(img, 5) << endl;
            int lRoom = -1;
            cout << "Nearest landmark is: " << mMap.FindRoomThenLandmark(img, &lRoom, 5);
            cout << ", in room " << mConfig.GetRoomName(lRoom) << endl;
        }



    }

};

int main() {
    initParameters();

    TopoMapTester lTester = TopoMapTester();
    lTester.TestLocalization();
    //lTester.Run();

    // New Localization tester
    //{
    //    int nExperiments = N_EXPERIMENTS;
    //    int nRooms = 3;
    //    vector<float> stats(nRooms * 2 + 1, 0); // accuracy and unidentified
    //    vector<float> correctStats(nRooms * 2, 0); // accuracy and unidentified
    //    for (size_t i = 0; i < nExperiments; ++i)
    //    {
    //        cout << "---------------- Run " << i + 1 << " -----------------" << endl;
    //        NewTester mTester = NewTester(N_LEARNING, N_TEST, N_IMGS);
    //        //Tester mTester = Tester();
    //        mTester.Run(&stats, ENABLE_CORRECTION, &correctStats);
    //    }

    //    cout << "Percentage for correct and unidentified in 3 rooms: " << endl;
    //    for (size_t i = 0; i < stats.size(); ++i)
    //    {
    //        cout << stats[i] / nExperiments << ", ";
    //    }
    //    cout << endl;
    //}

    //// Old localization tester based on bag of words
    ////{
    //int nExperiments = N_EXPERIMENTS;
    //int nRooms = 3;
    //vector<float> stats(nRooms * 2 + 1, 0); // accuracy and unidentified
    //vector<float> correctStats(nRooms * 2, 0); // accuracy and unidentified
    //for (size_t i = 0; i < nExperiments; ++i)
    //{
    //    cout << "---------------- Run " << i + 1 << " -----------------" << endl;
    //    Tester mTester = Tester(N_LEARNING, N_TEST, N_IMGS);
    //    //Tester mTester = Tester();
    //    mTester.Run(&stats, ENABLE_CORRECTION, &correctStats);
    //}

    //cout << "Percentage for correct and unidentified in 3 rooms: " << endl;
    //for (size_t i = 0; i < stats.size(); ++i)
    //{
    //    cout << stats[i] / nExperiments << ", ";
    //}
    //cout << endl;
    ////}

//Orientation
    //{
    //    OrientationTester lTester;
    //    float timings = 0;
    //    auto t1 = std::chrono::high_resolution_clock::now();
    //    lTester.Run();
    //    lTester.TestMulti(1);
    //    string filename = "D:\\WorkSpace\\03_Resources\\Dataset\\v2\\nMatches.bin";

    //    //{
    //    //    ifstream ifs(filename, ios::binary);
    //    //    cereal::PortableBinaryInputArchive iarchive(ifs);
    //    //    iarchive(lTester);
    //    //}

    //    lTester.TestLocalLandmark();
    //    {
    //        ofstream ofs(filename, ios::binary);
    //        cereal::PortableBinaryOutputArchive oarchive(ofs);
    //        oarchive(lTester);
    //    }
    //    lTester.Run();

    //    lTester.GetLandmarkDistances(6);
    //    auto t2 = std::chrono::high_resolution_clock::now();
    //    timings += (float)std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    //    cout << "Timing par image" << timings / 18 << endl;
    //    cout << "THRESHOLD 1: " << THRESHOLD_CIRCULAR_FIRST << endl;
    //    cout << "THRESHOLD 2: " << THRESHOLD_CIRCULAR_SECOND << endl;
    //    cout << "RADIUS FREE: " << RADIUS_FREE << endl;
    //}
    cin.get();

    //float timings = 0;
    //size_t i = 0;
    //string result = "";
    //int count = 0;
    //while (i < imgs.size())
    //{
    //    //cout << "Localisation start " << endl;
    //    auto t1 = std::chrono::high_resolution_clock::now();
    //    int trys = 0;
    //    bool halt = false;
    //    while (result == "" && i < imgs.size() && !halt)
    //    {
    //        trys++;
    //        result = mLocalizer.IdentifyRoom(imgs[i++], &halt);
    //    }
    //    auto t2 = std::chrono::high_resolution_clock::now();
    //    timings += (float)std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

    return 0;
}


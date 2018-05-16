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
        nLearning(numLearning), nTest(numTest), nImgs(numImgs) {}

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
    }

    void ReportDict()
    {
        vector<int> wordsCount = mLocalizer.CountWords();
        vector<int> nodesCount = mLocalizer.CountNodes();
        vector<int> featuresCount = mLocalizer.CountFeatures();
        cout << "Words SIFT " << wordsCount[0] << " color " << wordsCount[1] << endl;
        cout << "Nodes SIFT " << nodesCount[0] << " color " << nodesCount[1] << endl;
        cout << "Features learnt SIFT " << featuresCount[0] << " color " << featuresCount[1] << endl;
        cout << endl;

        if (mConfig.GetRoomCount() == 3)
        {
            //vector<int> SIFTAnalysis = mLocalizer.AnalyseDict(USE_SIFT);
            //vector<int> ColorAnalysis = mLocalizer.AnalyseDict(USE_COLOR);
            //cout << "SIFT Dict analysis: " << endl;
            //for (auto i : SIFTAnalysis)
            //{
            //    cout << setw(4) << i << ", ";
            //}
            //cout << endl << endl;

            //cout << "Color Dict analysis: " << endl;
            //for (auto i : ColorAnalysis)
            //{
            //    cout << setw(4) << i << ", ";
            //}
            //cout << endl << endl;
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

    void ReportResults(vector<Mat>& imgs, int label, vector<float>* stats, int offset = 0)
    {
        srand(time(0));
        random_shuffle(imgs.begin(), imgs.end());
        int correct = 0;
        int unIdentified = 0;
        //for (size_t i = nLearning; i < nLearning + nTest; ++i)
        float timings = 0;
        for (size_t i = 0; i < nTest; ++i)
        {
            vector<Mat> lImgs;
            for (size_t j = 0; j < nImgs; j++)
            {
                Mat lImg = imgs[offset + i * nImgs + j];
                lImgs.push_back(lImg);
            }

            shared_ptr<float> quality = make_shared<float>(0);
            auto t1 = std::chrono::high_resolution_clock::now();
            int result;

            if (ENABLE_CORRECTION)
            {
                // modify words + delete words in all words
                result = mLocalizer.IdentifyRoom(lImgs, quality, VERBOSE, label);
            }
            else
            {
                // no feedback, no modification  
                // TODO: add posteriori modification
                result = mLocalizer.IdentifyRoom(lImgs, quality, VERBOSE);
            }

            //cout << "Room detected: " << result << " For " << i;
            if (result == label)
            {
                cout << ((VERBOSE) ? "**** Correct **** \n" : "");
                //cout << "correct" << endl;
                correct++;
            }
            else if (result == -1)
            {
                cout << ((VERBOSE) ? "**** Unidentified **** \n" : "");
                unIdentified++;
            }
            else
            {
                cout << ((VERBOSE) ? "result is: " + to_string(result) + "\n" : "");
                cout << ((VERBOSE) ? "**** Wrong **** \n" : "");
                if (DISP_IMAGE)
                {
                    imshow(to_string(i), lImgs[0]);
                    waitKey(0);
                }
            }

            auto t2 = std::chrono::high_resolution_clock::now();
            timings += (float)std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
            //std::cout << "took "
            //    << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
            //    << " milliseconds\n" << endl;
        }
        if (ENABLE_CORRECTION)
        {
            mLocalizer.RemoveCommonWords();
        }
        float percentCorrect = correct * 100.0 / nTest;
        float percentUnidentified = unIdentified * 100.0 / nTest;

        cout << "***********  Average timing: " << timings / nTest << endl;
        cout << "** ** ** **  Correct: " << percentCorrect << "%" << endl;
        cout << "* * * * * *  Unidentified: " << percentUnidentified << "%";
        cout << endl;
        (*stats)[label * 2] += percentCorrect;
        (*stats)[label * 2 + 1] += percentUnidentified;
        (*stats).back() += timings;
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
            reunionTest.push_back(imread(reunionTestPath + to_string(i) + ".jpg", IMREAD_COLOR));
            mangerTest.push_back(imread(mangerTestPath + to_string(i) + ".jpg", IMREAD_COLOR));
        }
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


            {
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

            // Training and saving dict
            //{
            //    string salon = "Salon";
            //    string cuisine = "Cuisine";
            //    string reunion = "Reunion";
            //    mLocalizer.AddRoom(salon);
            //    mLocalizer.AddRoom(cuisine);
            //    mLocalizer.AddRoom(reunion);
            //    Train(&salonImgs, salon);
            //    ReportDict();
            //    Train(&cuisineImgs, cuisine);
            //    ReportDict();
            //    Train(&reunionImgs, reunion);
            //    ReportDict();
            //    //Train(&mangerImgs, MANGER);
            //    //ReportDict();
            //    cout << " Training done " << endl;
            //    cout << endl << endl;

            //    auto t1 = std::chrono::high_resolution_clock::now();
            //    {
            //        ofstream ofs(filename, ios::binary);
            //        cereal::PortableBinaryOutputArchive oarchive(ofs);
            //        oarchive(mLocalizer);
            //    }
            //    {
            //        ofstream ofs(configPath, ios::binary);
            //        cereal::PortableBinaryOutputArchive oArchive(ofs);
            //        oArchive(mConfig);
            //    }
            //    auto t2 = std::chrono::high_resolution_clock::now();
            //    double timings = (float)std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
            //    cout << "Model saved" << timings << endl;
            //}

            // Testing results
            {
                string salon = "Salon";
                string cuisine = "Cuisine";
                string reunion = "Reunion";
                //mLocalizer.RemoveRoom(salon);
                //ReportIncremental(salonTest, salon, results);
                //ReportIncremental(cuisineTest, cuisine, results);
                //ReportIncremental(reunionTest, reunion, results);
                cout << mLocalizer.GetOrientationToLandmark(salonTest[0], 0) << endl;
                cout << mLocalizer.GetOrientationToLandmark(salonTest[1], 0) << endl;
                //ReportIncremental(mangerTest, MANGER, results);
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

class OrientationTester
{
public:
    string root = "D:/WorkSpace/03_Resources/Dataset/Angle/";
    vector<string> mRooms{ "Salon", "SalonNew"};
    vector<string> mTypes{ "Train", "Test" };
    vector<string> mSets{ "1","2" };
    map<string, vector<float>> mRefs;
    const double pi = 3.1415926; // radian or ?

public:

    void Run()
    {
        std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
        std::cout.precision(0);
        Test("1");
        Test("2");
    }
    int GetIndex(string room)
    {
        return distance(mRooms.begin(), find(mRooms.begin(), mRooms.end(), room));
    }

    float GetDegrees(float radian)
    {
        return radian / pi * 180;
    }

    void Test(string set)
    {
        Localizer mLocalizer;
        int nTrain = (set.compare("1") == 0) ? 6 : 12;
        int nTest = 6;
        cout << "In " << set << endl;
        for (const string& room : mRooms)
        {
            mLocalizer.AddRoom(room);
            for (size_t i = 1; i <= nTrain; i++)
            {
                string filename = root + room + "Train" + set + "/" + to_string(i) + ".jpg";
                Mat img = imread(filename, IMREAD_COLOR); // Read the file
                mLocalizer.LearnImage(img, room, GetIndex(room), (i-1) * 2 * pi / (float)nTrain);
            }

            cout << room << " angles ";
            for (string type : mTypes)
            {
                for (string set : mSets)
                {
                    cout << " on " << type << " " << set << endl;
                    if (type.compare("Train") == 0 && set.compare("2") == 0)
                    {
                        nTest = 12;
                    }
                    else
                    {
                        nTest = 6;
                    }
                    for (size_t i = 1; i <= nTest; i++)
                    {
                        string filename = root + room + type + set + "/" + to_string(i) + ".jpg";
                        Mat img = imread(filename, IMREAD_COLOR); // Read the file
                        cout << GetDegrees(mLocalizer.GetOrientationToLandmark(img, GetIndex(room))) << ", ";
                    }
                    cout << endl;
                }
            }
            cout << endl;
        }
        cout << endl << endl;
    }

};

int main() {
    //initParameters();
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
    OrientationTester lTester;
    lTester.Run();

    std::cin.get();
    return 0;
}


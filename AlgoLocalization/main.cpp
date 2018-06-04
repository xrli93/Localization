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
        ReportDict();
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
            vector<int> SIFTAnalysis = mLocalizer.AnalyseDict(USE_SIFT);
            vector<int> ColorAnalysis = mLocalizer.AnalyseDict(USE_COLOR);
            cout << "SIFT Dict analysis: " << endl;
            for (auto i : SIFTAnalysis)
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

        //srand(time(0));
        //random_shuffle(imgs.begin(), imgs.end());
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
        //srand(time(0));
        //random_shuffle(imgs.begin(), imgs.end());
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
            //salonTest.push_back(imread(salonTrainPath + to_string(i) + ".jpg", IMREAD_COLOR));
            //cuisineTest.push_back(imread(cuisineTrainPath + to_string(i) + ".jpg", IMREAD_COLOR));
            reunionTest.push_back(imread(reunionTestPath + to_string(i) + ".jpg", IMREAD_COLOR));
            mangerTest.push_back(imread(mangerTestPath + to_string(i) + ".jpg", IMREAD_COLOR));
        }
    }

    void ReportParams()
    {
        cout << "RADIUS_SIFT: " << RADIUS_SIFT << endl;
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

class OrientationTester
{
public:
    //string root = "D:/WorkSpace/03_Resources/Dataset/Angle/";
    string root = "D:/WorkSpace/03_Resources/Dataset/Odometry/";
    string root = "D:/WorkSpace/03_Resources/Dataset/Odometry3/";
    //vector<string> mRooms{ "Salon", "SalonNew"};
    //vector<string> mRooms{ "Salon", "SalonNew", "Cuisine", "Hall" };
    vector<string> mRooms{ "Salon", "Cuisine", "Hall" };
    vector<string> mTypes{ "Train", "Test" };
    vector<string> mSets{ "1","2" };
    map<string, vector<float>> mRefs;
    const double pi = 3.1415926; // radian or ?
    vector<Mat> mDescriptors;
    vector<float> mAngles;

    Ptr<Feature2D> f2d;
    Ptr<Feature2D> extract;
    BFMatcher matcher;

public:

    void Run()
    {
        std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
        std::cout.precision(3);
        //Test("1");
        //Test("2");

        //f2d = MSER::create();
        //extract = xfeatures2d::DAISY::create();

        f2d = AgastFeatureDetector::create();
        extract = xfeatures2d::FREAK::create();

        //f2d = AKAZE::create();
        //extract = f2d;
        for (const string& room : mRooms)
        {
            matcher = BFMatcher(NORM_HAMMING);
            //matcher = BFMatcher(NORM_L2);
            cout << "Room: " << room << endl;
            LearnImgs(room, 12, "1");
            for (size_t i = 1; i <= 6; i++)
            {
                cout << "Image: " << i << endl;
                GetOrientation(room, i, "Test", "1");
            }
            mDescriptors.clear();
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
        cout << "Words SIFT " << wordsCount[0] << " color " << wordsCount[1] << endl;
        cout << "Nodes SIFT " << nodesCount[0] << " color " << nodesCount[1] << endl;
        cout << "Features learnt SIFT " << featuresCount[0] << " color " << featuresCount[1] << endl;
        cout << endl;

        if (mConfig.GetRoomCount() == 3)
        {
            vector<int> SIFTAnalysis = mLocalizer.AnalyseDict(USE_SIFT);
            vector<int> ColorAnalysis = mLocalizer.AnalyseDict(USE_COLOR);
            cout << "SIFT Dict analysis: " << endl;
            for (auto i : SIFTAnalysis)
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

    void LearnImgs(string room, int nTrain = 12, string set = "1")
    {
        for (size_t i = 1; i <= nTrain; i++)
        {
            string filename = root + room + "Train" + set + "/" + to_string(i) + ".jpg";
            Mat img = imread(filename, IMREAD_GRAYSCALE); // Read the file
            std::vector<KeyPoint> keypoints;
            Mat descriptors;
            f2d->detect(img, keypoints);
            extract->compute(img, keypoints, descriptors);
            mDescriptors.push_back(descriptors);
            mAngles.push_back((float)(i - 1) / nTrain * 360);
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

    void GetOrientation(string room, int index, string type = "Test", string set = "1")
    {
        string filename = root + room + type + set + "/" + to_string(index) + ".jpg";
        Mat img = imread(filename, IMREAD_GRAYSCALE); // Read the file
        img = Resize(img);
        std::vector<KeyPoint> keypoints;
        Mat lDescriptors;
        f2d->detect(img, keypoints);
        extract->compute(img, keypoints, lDescriptors);
        const float nnRatio = 0.75f;
        vector<int> nMatchList(mDescriptors.size());
        for (int i = 1; i <= mDescriptors.size(); ++i)
        {
            int nMatches = 0;
            vector<vector<DMatch>> nnMatches;
            matcher.knnMatch(lDescriptors, mDescriptors[i - 1], nnMatches, 2);
            for (auto& x : nnMatches)
            {
                if (x[0].distance < nnRatio * x[1].distance)
                {
                    ++nMatches;
                }
            }
            nMatchList[i - 1] = nMatches;
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

        //int sumMatches = maxMatch + secondMatch;
        //float factor1 = (maxMatch * 1.0) / sumMatches;
        //float factor2 = (secondMatch * 1.0) / sumMatches;

        int sumMatches = maxMatch * maxMatch + secondMatch * secondMatch;
        float factor1 = (maxMatch * maxMatch * 1.0) / sumMatches;
        float factor2 = (secondMatch * secondMatch * 1.0) / sumMatches;

        //float diffAngle = Angles::CircularMean(vector<float> {mAngles[bestIndex], -1 * mAngles[secondIndex]});
        //vector<float> lAngles{ factor1 * mAngles[bestIndex], factor2 * mAngles[secondIndex] };
        float bestAngle = mAngles[bestIndex];
        float secondAngle = mAngles[secondIndex];
        float diff = bestAngle - secondAngle;
        vector<float> lAngles{ bestAngle, secondAngle };
        float midAngle = Angles::CircularMean(lAngles);
        float midAngle2 = Angles::CircularMean(lAngles, vector<float> {factor1, factor2});
        float stdDev = Angles::CircularStdDev(vector<float> {bestAngle, secondAngle});
        if (stdDev > MAX_STD_DEV || maxMatch < MIN_MATCH)
        {
            midAngle2 = NO_ORIENTATION;
        }
        cout << maxMatch << ", " << secondMatch << endl;
        cout << stdDev << ", " << bestAngle << ", " << secondAngle << ", " << midAngle << "," << midAngle2 << endl;
        //cout << "Mid: " << midAngle << ", weighted: " << midAngle2  <<  endl;
        //cout << mAngles[bestIndex];

            //if (nMatches > maxMatch)
            //{
            //    maxMatch = nMatches;
            //    bestMatch = (i);
            //}
            ////cout << nMatches << endl;
    }

    void Test(string set)
    {
        Localizer mLocalizer;
        //int nTrain = (set.compare("1") == 0) ? 6 : 12;
        int nTrain = 12;
        int nTest = 6;
        cout << "In " << set << endl;
        for (const string& room : mRooms)
        {
            mLocalizer.AddRoom(room);
            for (size_t i = 1; i <= nTrain; i++)
            {
                string filename = root + room + "Train" + set + "/" + to_string(i) + ".jpg";
                Mat img = imread(filename, IMREAD_COLOR); // Read the file
                mLocalizer.LearnImage(img, room, GetIndex(room), (float)(i - 1) / nTrain * 360);
            }

            mLocalizer.GetAnalyseOrientations();
            ReportDict(mLocalizer);

            cout << room << " angles ";
            for (string type : mTypes)
            {
                //for (string set : mSets)
                //{
                //    cout << " on " << type << " " << set << endl;
                //    if (type.compare("Train") == 0 && set.compare("2") == 0)
                //    {
                //        nTest = 12;
                //    }
                //    else
                //    {
                //        nTest = 6;
                //    }
                //    for (size_t i = 1; i <= nTest; i++)
                //    {
                //        string filename = root + room + type + set + "/" + to_string(i) + ".jpg";
                //        Mat img = imread(filename, IMREAD_COLOR); // Read the file
                //        //cout << GetDegrees(mLocalizer.GetOrientationToLandmark(img, GetIndex(room))) << ", ";
                //        cout << mLocalizer.GetOrientationToLandmark(img, GetIndex(room)) << ", ";
                //    }
                //    cout << endl;
                //}
                if (type.compare("Train") == 0)
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
                    //cout << GetDegrees(mLocalizer.GetOrientationToLandmark(img, GetIndex(room))) << ", ";
                    cout << mLocalizer.GetOrientationToLandmark(img, GetIndex(room)) << ", ";
                }
                cout << endl;
            }
            cout << endl;
        }
        cout << endl << endl;
    }

};


int main() {
    initParameters();

    //{
        int nExperiments = N_EXPERIMENTS;
        int nRooms = 3;
        vector<float> stats(nRooms * 2 + 1, 0); // accuracy and unidentified
        vector<float> correctStats(nRooms * 2, 0); // accuracy and unidentified
        for (size_t i = 0; i < nExperiments; ++i)
        {
            cout << "---------------- Run " << i + 1 << " -----------------" << endl;
            Tester mTester = Tester(N_LEARNING, N_TEST, N_IMGS);
            //Tester mTester = Tester();
            mTester.Run(&stats, ENABLE_CORRECTION, &correctStats);
        }

        cout << "Percentage for correct and unidentified in 3 rooms: " << endl;
        for (size_t i = 0; i < stats.size(); ++i)
        {
            cout << stats[i] / nExperiments << ", ";
        }
        cout << endl;
    //}

    {
        //Orientation
        //OrientationTester lTester;
        //lTester.Run();
        //cout << "THRESHOLD 1: " <<  THRESHOLD_CIRCULAR_FIRST << endl;
        //cout << "THRESHOLD 2: " <<  THRESHOLD_CIRCULAR_SECOND << endl;
        //cout << "RADIUS SIFT: " << RADIUS_SIFT << endl;
    }
    cin.get();

    return 0;
}


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
                Train(&salonImgs, salon);
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
                ReportIncremental(salonTest, salon, results);

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
        if (*quality >= THRESHOLD_SECOND_VOTE || ++trys >= NUM_MAX_IMAGES)
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
        for (size_t i = 1; i <= 12; ++i)
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
        for (size_t i = 1; i <= 6; ++i)
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

    Ptr<Feature2D> f2d;
    Ptr<Feature2D> extract;
    BFMatcher matcher;

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


    void TestLocalLandmark()
    {
        mRooms = vector<string>{ "Hall" };
        root = "D:/WorkSpace/03_Resources/Dataset/Odometry3/";
        std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
        std::cout.precision(3);

        f2d = BRISK::create(15, 5);
        //f2d = AgastFeatureDetector::create();
        extract = xfeatures2d::FREAK::create();
        //extract = f2d;
        matcher = BFMatcher(NORM_HAMMING);

        Localizer mLocalizer;
        //vector<int> nTrains{ 12,12,6 };
        vector<int> nTrains{ 12,12,12 };
        int nSet = 3;
        for (const string& room : mRooms)
        {
            cout << "Room: " << room << endl;
            mLocalizer.AddRoom(room);
            for (int set = 1; set <= 3; set++)
            {
                LearnImgs(room, nTrains[set - 1], to_string(set), set);
            }
        }

        //vector<int> nTotalMatches(nSet * 3, 0);
        //root = "D:/WorkSpace/03_Resources/Dataset/Odometry3bis/";
        map<int, vector<int>> nTotalMatches;
        nTotalMatches.insert(make_pair(1, vector<int>(3, 0)));
        nTotalMatches.insert(make_pair(2, vector<int>(3, 0)));
        nTotalMatches.insert(make_pair(3, vector<int>(3, 0)));
        nTotalMatches.insert(make_pair(4, vector<int>(3, 0)));
        nTotalMatches.insert(make_pair(5, vector<int>(3, 0)));
        nTotalMatches.insert(make_pair(6, vector<int>(3, 0)));
        for (int imageSet = 1; imageSet <= 3; imageSet++)
        {
            for (size_t i = 1; i <= 6; i++)
            {
                cout << "Image: " << i << " Set: " << imageSet << endl;
                for (size_t iLandmark = 1; iLandmark <= 3; iLandmark++)
                {
                    nTotalMatches[imageSet][iLandmark - 1] += GetTotalMatches("Hall", i, "Train", to_string(imageSet), iLandmark);
                }
            }
            cout << endl;
        }
        cout << endl;
        for (auto& x : nTotalMatches)
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
        mRooms = vector<string>{ "SalonNew", "Cuisine", "Hall" };
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

    int GetTotalMatches(string room, int index, string type = "Test", string set = "1", int iLandmark = 1)
    {
        string filename = root + room + type + set + "/" + to_string(index) + ".jpg";
        Mat img = imread(filename, IMREAD_GRAYSCALE); // Read the file
        img = Resize(img);
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
        float* pStdDev = NULL, int* pNMatches = NULL, bool verbose = false)
    {
        string filename = root + room + type + set + "/" + to_string(index) + ".jpg";
        Mat img = imread(filename, IMREAD_GRAYSCALE); // Read the file
        img = Resize(img);
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
        float factor1 = (maxMatch * maxMatch * 1.0) / sumMatches;
        float factor2 = (secondMatch * secondMatch * 1.0) / sumMatches;

        //float diffAngle = Angles::CircularMean(vector<float> {mAngles[bestIndex], -1 * mAngles[secondIndex]});
        //vector<float> lAngles{ factor1 * mAngles[bestIndex], factor2 * mAngles[secondIndex] };
        float bestAngle = lLandmarkAngles[bestIndex];
        float secondAngle = lLandmarkAngles[secondIndex];
        float diff = bestAngle - secondAngle;
        vector<float> lAngles{ bestAngle, secondAngle };
        float midAngle = Angles::CircularMean(lAngles);
        float midAngle2 = Angles::CircularMean(lAngles, vector<float> {factor1, factor2});
        float stdDev = Angles::CircularStdDev(vector<float> {bestAngle, secondAngle});
        if (stdDev > MAX_STD_DEV || maxMatch < MIN_MATCH)
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
            cout << "Total matchings: " << maxMatch + secondMatch << endl;
            cout << stdDev << ", " << bestAngle << ", " << secondAngle << ", " << midAngle << "," << midAngle2 << endl;
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


int main() {
    initParameters();

    // New Localization tester
    {
        int nExperiments = N_EXPERIMENTS;
        int nRooms = 3;
        vector<float> stats(nRooms * 2 + 1, 0); // accuracy and unidentified
        vector<float> correctStats(nRooms * 2, 0); // accuracy and unidentified
        for (size_t i = 0; i < nExperiments; ++i)
        {
            cout << "---------------- Run " << i + 1 << " -----------------" << endl;
            NewTester mTester = NewTester(N_LEARNING, N_TEST, N_IMGS);
            //Tester mTester = Tester();
            mTester.Run(&stats, ENABLE_CORRECTION, &correctStats);
        }

        cout << "Percentage for correct and unidentified in 3 rooms: " << endl;
        for (size_t i = 0; i < stats.size(); ++i)
        {
            cout << stats[i] / nExperiments << ", ";
        }
        cout << endl;
    }

    // Old localization tester based on bag of words
    //{
    //    int nExperiments = N_EXPERIMENTS;
    //    int nRooms = 3;
    //    vector<float> stats(nRooms * 2 + 1, 0); // accuracy and unidentified
    //    vector<float> correctStats(nRooms * 2, 0); // accuracy and unidentified
    //    for (size_t i = 0; i < nExperiments; ++i)
    //    {
    //        cout << "---------------- Run " << i + 1 << " -----------------" << endl;
    //        Tester mTester = Tester(N_LEARNING, N_TEST, N_IMGS);
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

    {
        //Orientation
        //OrientationTester lTester;
        //float timings = 0;
        //auto t1 = std::chrono::high_resolution_clock::now();
        ////lTester.Run();
        ////lTester.TestMulti(1);
        //lTester.TestLocalLandmark();
        //auto t2 = std::chrono::high_resolution_clock::now();
        //timings += (float)std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
        //cout << "Timing par image" << timings / 18 << endl;
        //cout << "THRESHOLD 1: " << THRESHOLD_CIRCULAR_FIRST << endl;
        //cout << "THRESHOLD 2: " << THRESHOLD_CIRCULAR_SECOND << endl;
        //cout << "RADIUS SIFT: " << RADIUS_SIFT << endl;
    }
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


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
#include "matcerealisation.hpp"
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
    Tester() {}

    Tester(int numLearning, int numTest, int numImgs) :
        nLearning(numLearning), nTest(numTest), nImgs(numImgs) {}

    void Train(vector<Mat>* imgs, int label)
    {
        //srand(time(0));
        //random_shuffle(imgs->begin(), imgs->end());
        cout << "Training No. ";
        for (size_t i = 0; i < nLearning; ++i)
        {
            mLocalizer.LearnImage((*imgs)[i], label);
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

        if (NUM_ROOMS == 3)
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

    void ReportIncremental(vector<Mat>& imgs, int label, vector<float>* stats, int offset = 0)
    {

        srand(time(0));
        random_shuffle(imgs.begin(), imgs.end());
        int correct = 0;
        int unIdentified = 0;
        float timings = 0;
        size_t i = 0;
        int result = -1;
        int count = 0;
        while (i < imgs.size())
        {
            //cout << "Localisation start " << endl;
            auto t1 = std::chrono::high_resolution_clock::now();
            int trys = 0;
            bool halt = false;
            while (result == -1 && i < imgs.size() && !halt)
            {
                trys++;
                result = mLocalizer.IdentityRoom(imgs[i++], &halt);
            }
            auto t2 = std::chrono::high_resolution_clock::now();
            timings += (float)std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
            count++;

            cout << "Trys " << trys << endl;
            if (result == label)
            {
                cout << ((VERBOSE) ? "**** Correct **** \n" : "");
                ++correct;
            }
            else if (result == -1)
            {
                cout << ((VERBOSE) ? "**** Unidentified **** \n" : "");
                ++unIdentified;
            }
            else
            {
                cout << ((VERBOSE) ? "result is: " + to_string(result) + "\n" : "");
                cout << ((VERBOSE) ? "**** Wrong **** \n" : "");
            }
            result = -1;
        }
        float percentCorrect = correct * 100.0 / count;
        float percentUnidentified = unIdentified * 100.0 / count;

        cout << "***********  Average timing: " << timings / count << endl;
        cout << "** ** ** **  Correct: " << percentCorrect << "%" << endl;
        cout << "* * * * * *  Unidentified: " << percentUnidentified << "%" << endl;
        cout << "*    *    *  Average imgs taken: " << (float)imgs.size() / count <<
            " with SECOND_VOTE_SEUIL " << THRESHOLD_SECOND_VOTE << endl;
        cout << endl;
        (*stats)[label * 2] += percentCorrect;
        (*stats)[label * 2 + 1] += percentUnidentified;
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

        cout << "Read all imgs" << endl;
        for (size_t i = 0; i < nExperiments; ++i)
        {

            string filename = "D:\\WorkSpace\\03_Resources\\Dataset\\v2\\cereal.out";
            {
                auto t1 = std::chrono::high_resolution_clock::now();
                ifstream ifs(filename, ios::binary);
                cereal::PortableBinaryInputArchive iarchive(ifs);
                iarchive(mLocalizer);
                auto t2 = std::chrono::high_resolution_clock::now();
                double timings = (float)std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
                cout << "Model loaded " << timings << endl;
            }

            // Training and saving dict
            //{
            //    Train(&salonImgs, SALON);
            //    ReportDict();
            //    Train(&cuisineImgs, CUISINE);
            //    ReportDict();
            //    Train(&reunionImgs, REUNION);
            //    //Train(&mangerImgs, MANGER);
            //    cout << " Training done " << endl;
            //    ReportDict();
            //    cout << endl << endl;

            //    auto t1 = std::chrono::high_resolution_clock::now();
            //    {
            //        ofstream ofs(filename, ios::binary);
            //        cereal::PortableBinaryOutputArchive oarchive(ofs);
            //        oarchive(mLocalizer);
            //    }
            //    auto t2 = std::chrono::high_resolution_clock::now();
            //    double timings = (float)std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
            //    cout << "Model saved"  << timings << endl;
            //}

            // Testing results
            {
                ReportIncremental(salonTest, SALON, results);
                ReportIncremental(cuisineTest, CUISINE, results);
                ReportIncremental(reunionTest, REUNION, results);
            }
            //ReportResults(mangerTest, MANGER, results);
            ReportDict();
            cout << endl << endl;

        }

    }

};

int main() {

    //try
    //{
    initParameters();
    int nExperiments = N_EXPERIMENTS;
    vector<float> stats(NUM_ROOMS * 2 + 1, 0); // accuracy and unidentified
    vector<float> correctStats(NUM_ROOMS * 2, 0); // accuracy and unidentified
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
    std::cin.get();
    return 0;
}


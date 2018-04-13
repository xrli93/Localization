#include <iostream>
#include <iomanip>
#include <chrono>
#include "Node.h"
#include "TreeDict.h"
#include "ImageLearner.h";
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace std;
using namespace Localization;
using namespace cv;

void test(vector<int> vec)
{
    vec.push_back(1);
}

void testAddFeature()
{
    TreeDict<int> myDict = TreeDict<int>();
    myDict.AddFeature(1, 1);
    myDict.AddFeature(3, 2);
    myDict.AddFeature(5, 2);
    myDict.AddFeature(7, 2);
    myDict.AddFeature(9, 2);
}



void testSearch(TreeDict<cv::Mat> *dict)
{
    cv::Mat mat(1, 128, CV_32FC1);
    randu(mat, cv::Scalar(0), cv::Scalar(100));
    dict->Search(dict->GetRootNode(), mat);
}
void testCH()
{
    ColorHistogramLearner learner = ColorHistogramLearner();
    cv::Mat img1 = cv::imread("C:\\Users\\Hoth\\Downloads\\opencv-logo1.png", cv::IMREAD_COLOR); // Read the file
    cv::Mat img2 = cv::imread("C:\\Users\\Hoth\\Downloads\\opencv-logo2.png", cv::IMREAD_COLOR); // Read the file
    cv::Mat img3 = cv::imread("C:\\Users\\Hoth\\Downloads\\opencv-logo3.png", cv::IMREAD_COLOR); // Read the file
    Mat features = learner.CalculateFeatures(img2);
    learner.LearnImage(img1, 0);
    TreeDict<cv::Mat> myDict = learner.mDict;
    cout << "Total Words " << myDict.CountWords(myDict.GetRootNode()) << endl;
    cout << "Total Nodes " << myDict.CountNodes(myDict.GetRootNode()) << endl;
    //cout << features << endl;
    //cout << features << endl;
}

void testSIFT()
{
    SIFTImageLearner learner = SIFTImageLearner();
    cv::Mat img1 = cv::imread("C:\\Users\\Hoth\\Downloads\\opencv-logo1.png", cv::IMREAD_COLOR); // Read the file
    cv::Mat img2 = cv::imread("C:\\Users\\Hoth\\Downloads\\opencv-logo2.png", cv::IMREAD_COLOR); // Read the file
    cv::Mat img3 = cv::imread("C:\\Users\\Hoth\\Downloads\\opencv-logo3.png", cv::IMREAD_COLOR); // Read the file
    cv::Mat img11 = cv::imread("C:\\Users\\Hoth\\Downloads\\opencv-logo11.png", cv::IMREAD_COLOR); // Read the file
    cv::Mat img12 = cv::imread("C:\\Users\\Hoth\\Downloads\\opencv-logo12.png", cv::IMREAD_COLOR); // Read the file
    cv::Mat img21 = cv::imread("C:\\Users\\Hoth\\Downloads\\opencv-logo21.png", cv::IMREAD_COLOR); // Read the file
    cv::Mat img22 = cv::imread("C:\\Users\\Hoth\\Downloads\\opencv-logo22.png", cv::IMREAD_COLOR); // Read the file
    cv::Mat img31 = cv::imread("C:\\Users\\Hoth\\Downloads\\opencv-logo31.png", cv::IMREAD_COLOR); // Read the file
    cv::Mat img32 = cv::imread("C:\\Users\\Hoth\\Downloads\\opencv-logo32.png", cv::IMREAD_COLOR); // Read the file

    auto t1 = std::chrono::high_resolution_clock::now();
    learner.LearnImage(img1, 0);
    //learner.LearnImage(img11, 0);
    //learner.LearnImage(img12, 0);
    TreeDict<cv::Mat> myDict = learner.mDict;

    cout << "Total Words " << myDict.CountWords(myDict.GetRootNode()) << endl;
    cout << "Total Nodes " << myDict.CountNodes(myDict.GetRootNode()) << endl;

    learner.LearnImage(img2, 1);
    //learner.LearnImage(img21, 1);
    //learner.LearnImage(img22, 1);
    cout << "Total Words " << myDict.CountWords(myDict.GetRootNode()) << endl;
    cout << "Total Nodes " << myDict.CountNodes(myDict.GetRootNode()) << endl;

    learner.LearnImage(img3, 2);
    //learner.LearnImage(img31, 2);
    //learner.LearnImage(img32, 2);
    cout << "Total Words " << myDict.CountWords(myDict.GetRootNode()) << endl;
    cout << "Total Nodes " << myDict.CountNodes(myDict.GetRootNode()) << endl;
    cout << "Total features learnt" << learner.mFeatureCount << endl;

    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "took "
        << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
        << " milliseconds\n" << endl << endl;


    t1 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < 10; i++)
    {
        testSearch(&learner.mDict);
    }

    t2 = std::chrono::high_resolution_clock::now();
    std::cout << "took "
        << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
        << " milliseconds\n";
    //cv::Mat features = learner.CalculateFeatures(img);
    //cv::Mat row(features.row(0));

    double quality = 0;
    int res = learner.IdentifyImage(img21, &quality);
    cout << "Room is " << res << endl;
}

int main() {
    //SIFTImageLearner<cv::Mat> learner = SIFTImageLearner<cv::Mat>();
    //testCH();
    testSIFT();
    std::cin.get();
    return 0;
}
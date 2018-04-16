#include <iostream>
#include <iomanip>
#include <chrono>
#include "Node.h"
#include "TreeDict.h"
#include "Constants.h"
#include "Localizer.h"
#include "ImageLearner.h"
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
    learner.LearnImage(img11, 0);
    learner.LearnImage(img12, 0);
    TreeDict<cv::Mat> myDict = learner.mDict;

    cout << "Total Words " << myDict.CountWords(myDict.GetRootNode()) << endl;
    cout << "Total Nodes " << myDict.CountNodes(myDict.GetRootNode()) << endl;

    learner.LearnImage(img2, 1);
    learner.LearnImage(img21, 1);
    learner.LearnImage(img22, 1);
    cout << "Total Words " << myDict.CountWords(myDict.GetRootNode()) << endl;
    cout << "Total Nodes " << myDict.CountNodes(myDict.GetRootNode()) << endl;

    learner.LearnImage(img3, 2);
    learner.LearnImage(img31, 2);
    learner.LearnImage(img32, 2);
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
    //Mat features = learner.CalculateFeatures(img);
    //Mat row(features.row(0));

    double quality = 0;
    int res = learner.IdentifyImage(img21, &quality);
    cout << "Room is " << res << endl;
}


void testLocalizer()
{
    Mat img1 = imread("C:\\Users\\Hoth\\Downloads\\opencv-logo1.png", IMREAD_COLOR); // Read the file
    Mat img2 = imread("C:\\Users\\Hoth\\Downloads\\opencv-logo2.png", IMREAD_COLOR); // Read the file
    Mat img3 = imread("C:\\Users\\Hoth\\Downloads\\opencv-logo3.png", IMREAD_COLOR); // Read the file
    Mat img11 = imread("C:\\Users\\Hoth\\Downloads\\opencv-logo11.png", IMREAD_COLOR); // Read the file
    Mat img12 = imread("C:\\Users\\Hoth\\Downloads\\opencv-logo12.png", IMREAD_COLOR); // Read the file
    Mat img21 = imread("C:\\Users\\Hoth\\Downloads\\opencv-logo21.png", IMREAD_COLOR); // Read the file
    Mat img22 = imread("C:\\Users\\Hoth\\Downloads\\opencv-logo22.png", IMREAD_COLOR); // Read the file
    Mat img31 = imread("C:\\Users\\Hoth\\Downloads\\opencv-logo31.png", IMREAD_COLOR); // Read the file
    Mat img32 = imread("C:\\Users\\Hoth\\Downloads\\opencv-logo32.png", IMREAD_COLOR); // Read the file

    Mat blue1 = imread("C:\\Users\\Hoth\\Downloads\\blue1.jpg", IMREAD_COLOR); // Read the file
    Mat blue2 = imread("C:\\Users\\Hoth\\Downloads\\blue2.jpg", IMREAD_COLOR); // Read the file
    Mat red1 = imread("C:\\Users\\Hoth\\Downloads\\red1.jpg", IMREAD_COLOR); // Read the file
    Mat mix1 = imread("C:\\Users\\Hoth\\Downloads\\mix1.jpg", IMREAD_COLOR); // Read the file
    Mat mix2 = imread("C:\\Users\\Hoth\\Downloads\\mix2.jpg", IMREAD_COLOR); // Read the file
    Mat green1 = imread("C:\\Users\\Hoth\\Downloads\\green1.jpg", IMREAD_COLOR); // Read the file
    Mat green2 = imread("C:\\Users\\Hoth\\Downloads\\green2.jpg", IMREAD_COLOR); // Read the file

    // Reading
    //learner.LearnImage(img1, 0);
    //learner.LearnImage(img11, 0);
    //learner.LearnImage(img12, 0);
    //learner.LearnImage(img2, 1);
    //learner.LearnImage(img21, 1);
    //learner.LearnImage(img22, 1);
    //learner.LearnImage(img3, 2);
    //learner.LearnImage(img31, 2);
    //learner.LearnImage(img32, 2);


    Localizer localizer = Localizer();

    //localizer.AddImage(img1, 0);
    //localizer.AddImage(img11, 0);
    //localizer.AddImage(img12, 0);
    //localizer.AddImage(img2, 1);
    //localizer.AddImage(img21, 1);
    //localizer.AddImage(img22, 1);
    //localizer.AddImage(img3, 2);
    //localizer.AddImage(img31, 2);
    //localizer.AddImage(img32, 2);

    localizer.AddImage(blue1, 0);
    localizer.AddImage(blue2, 0);
    localizer.AddImage(green1, 1);
    localizer.AddImage(green2, 1);
    localizer.AddImage(mix1, 2);
    localizer.AddImage(mix2, 2);


    auto t1 = std::chrono::high_resolution_clock::now();
    localizer.LearnCollection();

    vector<int> wordsCount = localizer.CountWords();
    vector<int> nodesCount = localizer.CountNodes();
    vector<int> featuresCount = localizer.CountFeatures();
    cout << "SIFT Words " << wordsCount[0] << " color " << wordsCount[1] << endl;
    cout << "SIFT Nodes " << nodesCount[0] << " color " << nodesCount[1] << endl;
    cout << "SIFT features learnt" << featuresCount[0] << " color " << featuresCount[1] << endl;
    vector<Mat> imgs;
    imgs.push_back(red1);
    //imgs.push_back(img12);
    cout << localizer.IdentifyRoom(imgs);

    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "took "
        << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
        << " milliseconds\n";
    
    //testSIFT();

}
int main() {

    //testSIFT();
    testLocalizer();

    std::cin.get();
    return 0;
}
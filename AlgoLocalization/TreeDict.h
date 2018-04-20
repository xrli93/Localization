#pragma once
#include"Node.h"
#include<vector>
#include<limits>
#include"Word.h"
#include"Constants.h"
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
using namespace std;

namespace Localization
{
    template <class T>
    class TreeDict
    {
    private:
        Node<T> mRootNode{};
        int mFeatureMethod;
        double mRadius = RADIUS;
        //double mFrontier = RADIUS * 5;
        int mNMaxWords = NUM_MAX_WORDS;
        // DEBUG
        int v1 = 0;
        int v2 = 0;
        int v12 = 0;
    public:
        TreeDict()
        {
            mRootNode = Node<T>();
        }

        TreeDict(T origin)
        {
            mRootNode = Node<T>(origin);
        }

        //TreeDict(int featureMethod) : mFeatureMethod(featureMethod)
        //{
        //    mRootNode = Node<T>();
        //}

        ~TreeDict() {}

        void SetRootNodeCenter(const T& feature)
        {
            mRootNode.SetCenter(feature);
        }

        Node<T>* GetRootNode()
        {
            return &mRootNode;
        }

        void SetRadius(double radius)
        {
            mRadius = radius;
        }
        void SetRadius()
        {
            mRadius = (mFeatureMethod == USE_COLOR) ? RADIUS_COLOR : RADIUS_SIFT;
        }


        void SetNMaxWords(int nMaxWords)
        {
            mNMaxWords = nMaxWords;
        }

        void SetFeatureMethod(int featureMethod)
        {
            mFeatureMethod = featureMethod;
        }

        string GetFeatureMethod()
        {
            return (mFeatureMethod == USE_SIFT) ? "SIFT" : "Color";
        }

        // Count words in dict
        int CountWords() { return CountWords(&mRootNode); }

        int CountWords(Node<T>* node)
        {
            if (!node->IsLeafNode())
            {
                int sumWords = 0;
                vector<Node<T> *> childList = node->GetChildNodes();
                for (size_t i = 0; i < childList.size(); i++)
                {
                    Node<T> *lNode = childList[i];
                    sumWords += CountWords(lNode);
                }
                return sumWords;
            }
            else
            {
                return node->GetWordsCount();
            }
        }
        vector<int> AnalyseWords() { return AnalyseWords(&mRootNode); }

        vector<int> AnalyseWords(Node<T>* node)
        {
            if (!node->IsLeafNode())
            {
                vector<int> total(WORD_TYPES, 0);
                vector<Node<T> *> childList = node->GetChildNodes();
                for (size_t i = 0; i < childList.size(); i++)
                {
                    Node<T> *lNode = childList[i];
                    vector<int> lCount = AnalyseWords(lNode);
                    transform(total.begin(), total.end(), lCount.begin(), total.begin(), plus<int>());
                }
                return total;
            }
            else
            {
                return node->AnalyseWords();
            }
        }



        // Count nodes in dict

        int CountNodes() { return CountNodes(&mRootNode); }
        int CountNodes(Node<T>* node)
        {
            if (!node->IsLeafNode())
            {
                int sumNodes = 0;
                vector<Node<T> *> childList = node->GetChildNodes();
                for (size_t i = 0; i < childList.size(); i++)
                {
                    Node<T> *lNode = childList[i];
                    sumNodes += CountNodes(lNode);
                }
                return sumNodes;

            }
            else
            {
                return 1;
            }
        }

        void RemoveCommonWords() { RemoveCommonWords(&mRootNode); }

        void RemoveCommonWords(Node<T>* node)
        {
            if (!node->IsLeafNode())
            {
                vector<Node<T> *> childList = node->GetChildNodes();
                for (size_t i = 0; i < childList.size(); i++)
                {
                    Node<T> *lNode = childList[i];
                    RemoveCommonWords(lNode);
                }
            }
            else
            {
                node->RemoveCommonWords();
            }
        }

        // Adds a new Word into the dict
        // Returns the Node it belongs
        Node<T>* AddWordToDict(Word<T>* word)
        {
            Node<T> * lParentNode = FindNearestNode(word->GetCenter());
            lParentNode->AddWord(word);
            return lParentNode;
        }

        // Add a feature to the dict
        void AddFeature(T feature, int indexRoom)
        {
            //bool enableFullSearch = (mFeatureMethod == USE_SIFT) ? false : true;
            bool enableFullSearch = true;
            vector<Word<T> *> wordList = Search(feature, enableFullSearch);
            if (!wordList.empty())
            {
                typename vector<Word<T> *>::iterator iter;
                for (iter = wordList.begin(); iter != wordList.end(); iter++)
                {
                    (*iter)->UpdateLabel(indexRoom);
                }
            }
            else
            {
                Word<T> *newWord = new Word<T>(feature, indexRoom, mRadius);
                Node<T> *node = AddWordToDict(newWord);
                if (node->GetWordsCount() > mNMaxWords) // Many words necessary to create new nodes
                {
                    Expand(node);
                }
            }
        }

        ///<summary>
        /// Search in dict for words that contains feature
        ///</summary>
        vector<Word<T> *> Search(T feature, bool fullSearch = false)
        {
            return Search(&mRootNode, feature, fullSearch);
        }
        vector<Word<T> *> Search(Node<T> *node, T feature, bool fullSearch = false)
        {
            vector<Word<T> *> wordList;
            if (node->IsLeafNode())
            {
                vector<Word<T> *> lWordsInNode = node->GetWords();
                for (size_t i = 0; i < lWordsInNode.size(); i++)
                {
                    Word<T> *lWord = lWordsInNode[i];
                    if (lWord->ContainFeature(feature))
                    {
                        wordList.push_back(lWord);
                    }
                }
            }
            else
            {
                if (!fullSearch) // SIFT
                {
                    //vector<double> frontierDistances;
                    CalculateFrontierDistances(node, feature);
                    node->SortChildNodes();

                    vector<Node<T> *> childList = node->GetChildNodes();
                    //for (size_t i = 0; i < node->GetChildCount(); i++)
                    //{
                    //    cout << childList[i]->GetFrontier() << " ";
                    //}
                    //    cout << endl;
                    for (int i = 0; i < MAX_CHILD_NUM; i++)
                    {
                        Node<T> *lNode = childList[i];
                        if (lNode->GetFrontier() < mRadius)
                        {
                            //cout << lNode->GetFrontier() << endl;
                            vector<Word<T> *> childWordList = Search(lNode, feature, fullSearch);
                            wordList.insert(wordList.end(), childWordList.begin(), childWordList.end());
                        }
                    }
                }
                else // Color
                {
                    node->SortChildNodes(feature);
                    vector<Node<T> *> childList = node->GetChildNodes();
                    for (int i = 0; i < MAX_CHILD_NUM; i++)
                    {
                        vector<Word<T> *> childWordList = Search(childList[i], feature, fullSearch);
                        wordList.insert(wordList.end(), childWordList.begin(), childWordList.end());
                    }
                }
                //vector<Node<T> *> childList = node->GetChildNodes();
                //for (int i = 0; i < MAX_CHILD_NUM; i++)
                //{
                //    Node<T> *lNode = childList[i];
                //    //TODO: Optimization. Filliat 08. What is frontier distance?
                //    if (fullSearch)
                //    {
                //        vector<Word<T> *> childWordList = Search(lNode, feature, fullSearch);
                //        wordList.insert(wordList.end(), childWordList.begin(), childWordList.end());
                //    }
                //    else if (lNode->GetFrontier() < mRadius)
                //    {
                //        vector<Word<T> *> childWordList = Search(lNode, feature, fullSearch);
                //        wordList.insert(wordList.end(), childWordList.begin(), childWordList.end());
                //    }
                //}
            }
            return wordList;
        }



        // Find in dict the leaf node whose center is nearest to feature (for insertion)
        Node<T>* FindNearestNode(T feature)
        {
            Node<T> *minNode = &mRootNode;
            while (!minNode->IsLeafNode())
            {
                vector<Node<T> *> lChildNodes = minNode->GetChildNodes();
                double minDist = numeric_limits<double>::max();
                for (size_t i = 0; i < lChildNodes.size(); i++)
                {
                    Node<T> *lNode = lChildNodes[i];
                    T lCenter = lNode->GetCenter();
                    double lDist = Localization::CalculateDistance(lCenter, feature);
                    if (lDist < minDist)
                    {
                        minDist = lDist; // perhaps reuse this variable...
                        minNode = lNode;
                    }
                }
            }
            return minNode;
        }

        // Expand a leafnode when it contains many words
        template <class T>
        void Expand(Node<T>* node) {}

        // using OpenCV and KMeans that comes with
        void Expand(Node<cv::Mat>* node)
        {

            cv::Mat centers;
            cv::Mat labels;
            vector<Word<T> *> lWordList = node->GetWords();

            KMeansCluster(lWordList, &labels, &centers);
            for (size_t i = 0; i < centers.rows; i++)
            {
                Node<cv::Mat> *newNode = new Node<cv::Mat>(centers.row(i));
                for (int wordsIter = 0; wordsIter < lWordList.size(); wordsIter++)
                {
                    if (labels.at<int>(wordsIter, 0) == i)
                    {
                        // adding words to their nodes
                        newNode->AddWord(lWordList[wordsIter]);
                    }
                }
                node->AddChildNode(newNode);
            }
            node->RemoveWords();
        }


    };

    template <class T>
    double KMeansCluster(vector<Word<T> *> wordList, cv::Mat* labels, cv::Mat* centers)
    {
        cv::Mat featureMat = MakeFeatureListFromWords(wordList);
        return cv::kmeans(featureMat, K_SPLIT, *labels, cv::TermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 10, 0.1), 3, cv::KMEANS_PP_CENTERS, *centers);
    }

    template <class T>
    cv::Mat MakeFeatureListFromWords(vector<Word<T> *> wordList)
    {
        vector<T> featureList;
        for (size_t i = 0; i < wordList.size(); i++)
        {
            featureList.push_back(wordList[i]->GetCenter());
        }

        cv::Mat featureMat(featureList.size(), 1, CV_32FC1);  // ** Make 1 column Mat from vector
        return featureMat;
    }
    template<> cv::Mat MakeFeatureListFromWords(vector<Word<cv::Mat> *> wordList)
    {
        int featureDim = wordList[0]->GetCenter().cols;
        cv::Mat featureList(wordList.size(), featureDim, CV_32FC1, cv::Scalar(0));
        for (size_t i = 0; i < wordList.size(); i++)
        {
            wordList[i]->GetCenter().copyTo(featureList.row(i));
        }
        return featureList;
    }

}

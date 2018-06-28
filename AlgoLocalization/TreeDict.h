#pragma once
#include"Node.h"
#include<vector>
#include<limits>
#include"Word.h"
#include"Constants.h"
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "omp.h"
using namespace std;

namespace Localization
{
    template <class T>
    class TreeDict
    {
    private:
        shared_ptr<Node<T> > mRootNode;
        int mFeatureMethod;
        float mRadius = RADIUS;

        friend class cereal::access;
        template<class Archive>
        void serialize(Archive & archive)
        {
            archive(mRootNode, mFeatureMethod, mRadius);
        }
    public:
        TreeDict()
        {
            mRootNode = make_shared<Node<T>>();
        }

        TreeDict(T origin)
        {
            mRootNode = make_shared<Node<T>>(origin);
        }


        ~TreeDict() {}

        void SetRootNodeCenter(const T& feature)
        {
            mRootNode->SetCenter(feature);
        }

        shared_ptr<Node<T> > GetRootNode()
        {
            return mRootNode;
        }

        void SetRadius(float radius)
        {
            mRadius = radius;
        }
        void SetRadius()
        {
            mRadius = (mFeatureMethod == USE_COLOR) ? RADIUS_COLOR : RADIUS_FREE;
        }

        void SetFeatureMethod(int featureMethod)
        {
            mFeatureMethod = featureMethod;
        }

        string GetFeatureMethod()
        {
            return (mFeatureMethod == USE_FREE) ? "FREE" : "Color";
        }

        void RemoveRoom(const string& room) { RemoveRoom(mRootNode, room); }

        void RemoveRoom(shared_ptr<Node<T>> node, const string& room)
        {
            if (!node->IsLeafNode())
            {
                vector<shared_ptr<Node<T> > > childList = node->GetChildNodes();
                for (size_t i = 0; i < childList.size(); ++i)
                {
                    shared_ptr<Node<T> > lNode = childList[i];
                    RemoveRoom(lNode, room);
                }
            }
            else
            {
                node->RemoveRoom(room);
            }
        }

        // For each word, display roomPresences
        void CheckWords() {}

        // Count words in dict
        int CountWords() { return CountWords(mRootNode); }

        int CountWords(shared_ptr<Node<T> > node)
        {
            if (!node->IsLeafNode())
            {
                int sumWords = 0;
                vector<shared_ptr<Node<T> > > childList = node->GetChildNodes();
                for (size_t i = 0; i < childList.size(); ++i)
                {
                    shared_ptr<Node<T> > lNode = childList[i];
                    sumWords += CountWords(lNode);
                }
                return sumWords;
            }
            else
            {
                return node->GetWordsCount();
            }
        }
        vector<int> AnalyseWords() { return AnalyseWords(mRootNode); }

        vector<int> AnalyseWords(shared_ptr<Node<T> > node)
        {
            if (!node->IsLeafNode())
            {
                vector<int> total(WORD_TYPES, 0);
                vector<shared_ptr<Node<T> > > childList = node->GetChildNodes();
                for (size_t i = 0; i < childList.size(); ++i)
                {
                    shared_ptr<Node<T> > lNode = childList[i];
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
        vector<float> AnalyseOrientations()
        {
            return AnalyseOrientations(mRootNode);
        }

        vector<float> AnalyseOrientations(shared_ptr<Node<T> > node)
        {
            //static vector<float> lStdDevs;
            if (!node->IsLeafNode())
            {
                vector<float> lStdDevs;
                vector<shared_ptr<Node<T> > > childList = node->GetChildNodes();
                for (size_t i = 0; i < childList.size(); ++i)
                {
                    shared_ptr<Node<T> > lNode = childList[i];
                    vector<float> lNodeStdDevs = lNode->AnalyseNodeOrientations();
                    lStdDevs.insert(lStdDevs.end(), lNodeStdDevs.begin(), lNodeStdDevs.end());
                }
                return lStdDevs;
            }
            else
            {
                return node->AnalyseNodeOrientations();
            }
        }



        // Count nodes in dict

        int CountNodes() { return CountNodes(mRootNode); }
        int CountNodes(shared_ptr<Node<T> > node)
        {
            if (!node->IsLeafNode())
            {
                int sumNodes = 0;
                vector<shared_ptr<Node<T> > > childList = node->GetChildNodes();
                for (size_t i = 0; i < childList.size(); ++i)
                {
                    shared_ptr<Node<T> > lNode = childList[i];
                    sumNodes += CountNodes(lNode);
                }
                return sumNodes;

            }
            else
            {
                return 1;
            }
        }

        void RemoveCommonWords() { RemoveCommonWords(mRootNode); }

        void RemoveCommonWords(shared_ptr<Node<T> > node)
        {
            if (!node->IsLeafNode())
            {
                vector<shared_ptr<Node<T> > > childList = node->GetChildNodes();
                for (size_t i = 0; i < childList.size(); ++i)
                {
                    shared_ptr<Node<T> > lNode = childList[i];
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
        shared_ptr<Node<T> > AddWordToDict(shared_ptr<Word<T> >  word)
        {
            shared_ptr<Node<T> > lParentNode = FindNearestNode(word->GetCenter());
            lParentNode->AddWord(word);
            return lParentNode;
        }

        // Add a feature to the dict
        vector<shared_ptr<Word<T>>> AddFeature(T feature, int indexRoom)
        {
            //bool enableFullSearch = (mFeatureMethod == USE_FREE) ? false : true;
            //cout << feature;
            vector<shared_ptr<Word<T> > > wordList = Search(feature, FULL_SEARCH);
            if (!wordList.empty())
            {
                typename vector<shared_ptr<Word<T> > > ::iterator iter;
                for (iter = wordList.begin(); iter != wordList.end(); iter++)
                {
                    (*iter)->UpdateLabel(indexRoom);
                    if (USE_SYMMETRY)
                    {
                        (*iter)->AddSeenFeature(indexRoom, feature);
                    }
                }
            }
            else
            {
                // DEBUG
                shared_ptr<Word<T> > newWord = make_shared<Word<T> >(feature, indexRoom, mRadius);
                shared_ptr<Node<T> > node = AddWordToDict(newWord);
                if (USE_SYMMETRY)
                {
                    ReviewFeatures(newWord, node);
                }
                wordList.push_back(newWord);
                if (node->GetWordsCount() > NUM_MAX_WORDS) // Many words necessary to create new nodes
                {
                    Expand(node);
                }
            }
            // Subtle point: if return constant reference, it will not increment the counter
            // So wordList will be empty!
            return wordList;
        }

        // Make symmetrical learnings
        // Repass old features to the new word
        void ReviewFeatures(shared_ptr<Word<T>> iWord, shared_ptr<Node<T>> iNode)
        {
            vector<shared_ptr<Word<T>>> lWordList = iNode->GetWords();
            for (size_t i = 0; i < lWordList.size(); i++)
            {
                Word<T> lWord = *(lWordList[i]);
                if (CalculateDistance(lWord.GetCenter(), iWord->GetCenter()) <= 2 * iWord->GetRadius())
                {
                    map<int, vector<T>> lFeatures = lWord.GetSeenFeatures();
                    for (auto& x : lFeatures)
                    {
                        vector<T> lSeenFeatures = x.second;
                        for (T& lFeature : lSeenFeatures)
                        {
                            if (iWord->ContainFeature(lFeature))
                            {
                                iWord->UpdateLabel(x.first);
                            }
                        }
                    }
                }
            }
        }

        ///<summary>
        /// Search in dict for words that contains feature
        ///</summary>
        vector<shared_ptr<Word<T> > > Search(T feature, bool fullSearch = false)
        {
            return Search(mRootNode, feature, fullSearch);
        }
        vector<shared_ptr<Word<T> > >Search(shared_ptr<Node<T> > node, T feature, bool fullSearch = false)
        {
            vector<shared_ptr<Word<T> > > wordList;
            if (node->IsLeafNode())
            {
                vector<shared_ptr<Word<T> > > lWordsInNode = node->GetWords();
                for (int i = 0; i < lWordsInNode.size(); ++i)
                {
                    //shared_ptr<Word<T> > lWord = lWordsInNode[i];
                    //if (lWord->ContainFeature(feature))
                    //{
                    //    wordList.push_back(lWord);
                    //}
                    if (lWordsInNode[i]->ContainFeature(feature))
                    {
                        wordList.push_back(lWordsInNode[i]);
                    }
                }
            }
            else
            {
                fullSearch = true;
                if (!fullSearch) // FREE
                {
                    cout << "No enter!" << endl;
                    //vector<float> frontierDistances;
                    CalculateFrontierDistances(node, feature);
                    node->SortChildNodes();

                    vector<shared_ptr<Node<T> > > childList = node->GetChildNodes();
                    for (int i = 0; i < MAX_CHILD_NUM; ++i)
                    {
                        shared_ptr<Node<T> > lNode = childList[i];
                        if (lNode->GetFrontier() < mRadius)
                        {
                            //cout << lNode->GetFrontier() << endl;
                            vector<shared_ptr<Word<T> > > childWordList = Search(lNode, feature, fullSearch);
                            wordList.insert(wordList.end(), childWordList.begin(), childWordList.end());
                        }
                    }
                }
                else // Color
                {
                    node->SortChildNodes(feature);
                    vector<shared_ptr<Node<T> > > childList = node->GetChildNodes();
                    for (int i = 0; i < MAX_CHILD_NUM; ++i)
                    {
                        vector<shared_ptr<Word<T> > > childWordList = Search(childList[i], feature, fullSearch);
                        wordList.insert(wordList.end(), childWordList.begin(), childWordList.end());
                    }
                }
            }
            return wordList;
        }



        // Find in dict the leaf node whose center is nearest to feature (for insertion)
        shared_ptr<Node<T> > FindNearestNode(T feature)
        {
            shared_ptr<Node<T> > minNode = mRootNode;
            while (!minNode->IsLeafNode())
            {
                vector<shared_ptr<Node<T> > > lChildNodes = minNode->GetChildNodes();
                float minDist = numeric_limits<float>::max();
                //#pragma omp parallel for
                for (int i = 0; i < lChildNodes.size(); ++i)
                {
                    shared_ptr<Node<T> > lNode = lChildNodes[i];
                    T lCenter = lNode->GetCenter();
                    float lDist = Localization::CalculateDistance(lCenter, feature);
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
        void Expand(shared_ptr<Node<T> > node) {}

        // using OpenCV and KMeans that comes with
        void Expand(shared_ptr<Node<Mat> > node)
        {

            shared_ptr<Mat> centers = make_shared<Mat>();
            shared_ptr<Mat> labels = make_shared<Mat>();
            vector<shared_ptr<Word<T> > > lWordList = node->GetWords();

            KMeansCluster(lWordList, labels, centers);
            //cout << labels.cols;
            for (int i = 0; i < centers->rows; ++i)
            {
                shared_ptr<Node<Mat> >  newNode = make_shared<Node<Mat> >(centers->row(i));
                //cout << newNode->GetCenter() << endl;
                for (int wordsIter = 0; wordsIter < lWordList.size(); ++wordsIter)
                {
                    if (labels->at<int>(wordsIter, 0) == i)
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
    float KMeansCluster(vector<shared_ptr<Word<T> > > wordList, shared_ptr<Mat> labels, shared_ptr<Mat> centers)
    {
        Mat featureMat = MakeFeatureListFromWords(wordList);
        return kmeans(featureMat, K_SPLIT, *labels, TermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 10, 0.1), 3, KMEANS_PP_CENTERS, *centers);
    }

    template <class T>
    Mat MakeFeatureListFromWords(vector<shared_ptr<Word<T> > >& wordList)
    {
        vector<T> featureList;
#pragma omp parallel for
        for (size_t i = 0; i < wordList.size(); ++i)
        {
            featureList.push_back(wordList[i]->GetCenter());
        }

        Mat featureMat(featureList.size(), 1, CV_32FC1);  // ** Make 1 column Mat from vector
        return featureMat;
    }
    template<> Mat MakeFeatureListFromWords(vector<shared_ptr<Word<Mat> > >& wordList)
    {
        int featureDim = wordList[0]->GetCenter().cols;
        Mat featureList(wordList.size(), featureDim, CV_32FC1, Scalar(0)); // Exigee by K-Means
#pragma omp parallel for
        for (int i = 0; i < wordList.size(); ++i)
        {
            wordList[i]->GetCenter().copyTo(featureList.row(i));
        }
        return featureList;
    }

}

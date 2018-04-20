#pragma once

#include<vector>
#include<algorithm>
#include<limits>
#include "Word.h"
#include "Constants.h"

using namespace std;

namespace Localization
{

    template <class T>
    class Node
    {
    private:
        T mCenter{};
        std::vector<Node<T> *> mChildNodes;
        std::vector<Word<T> *> mWords;
        double mDistFrontier = numeric_limits<double>::max(); // store frontier distances in SIFT search to use std::sort
    public:
        Node()
        {
        }

        Node(T center) : mCenter(center) {};

        ~Node()
        {
            mChildNodes.clear();
            mWords.clear();
        }

        void AddWord(Word<T> *word)
        {
            mWords.push_back(word);
        }

        void AddChildNode(Node<T> *node)
        {
            mChildNodes.push_back(node);
        }

        void RemoveWords() // TODO: Change frontier?
        {
            while (!mWords.empty())
                mWords.pop_back();
        }

        vector<Word<T> *> GetWords() const
        {
            return mWords;
        }

        vector<Node<T> *> GetChildNodes() const
        {
            return mChildNodes;
        }

        void SetFrontier(double frontier)
        {
            mDistFrontier = frontier;
        }

        double GetFrontier() const
        {
            return mDistFrontier;
        }


        int GetChildCount() const
        {
            return mChildNodes.size();
        }

        // remove words present in all rooms
        void RemoveCommonWords()
        {
            mWords.erase(remove_if(mWords.begin(), mWords.end(),
                [](Word<T>* pWord) {return pWord->PresentInAll(); }),
                mWords.end());
        }


        // General sorting method, used by Color Histogram.
        // Sort child nodes according to their distances to a feature
        void SortChildNodes(T feature)
        {
            std::sort(mChildNodes.begin(), mChildNodes.end(),
                [feature](const Node<T>* lhs, const Node<T>* rhs)
            {
                double lhsDist = Localization::CalculateDistance(lhs->GetCenter(), feature);
                double rhsDist = Localization::CalculateDistance(rhs->GetCenter(), feature);
                return lhsDist < rhsDist;
            });
        }

        // When we have distances to Voronoi frontiers, SIFT
        void SortChildNodes()
        {
            std::sort(mChildNodes.begin(), mChildNodes.end(),
                [](const Node<T>* lhs, const Node<T>* rhs)
            {
                return (lhs->GetFrontier() < rhs->GetFrontier());
            });
        }
        bool IsLeafNode()
        {
            return ((mChildNodes.size() == 0) ? true : false);
        }

        // Number of words contained in Node
        int GetWordsCount()
        {
            return (int)mWords.size();
        }

        // Return a vector counting words according to their lables
        // {0, 1, 2, 0 + 1, 0 + 2, 1 + 2, 0 + 1 + 2}
        // TODO: make general
        vector<int> AnalyseWords()
        {
            assert(NUM_ROOMS == 3);
            vector<int> count(WORD_TYPES, 0);
            vector<int> total(WORD_TYPES, 0);
            for (size_t i = 0; i < mWords.size(); i++)
            {
                Word<T> *ptrWord = mWords[i];
                vector<bool> labels = ptrWord->GetLabels();
                bool salon = labels[SALON];
                bool cuisine = labels[CUISINE];
                bool reunion = labels[REUNION];

                count[SALON] = (int)salon;
                count[CUISINE] = (int)cuisine;
                count[REUNION] = (int)reunion;
                count[3] = (int)(salon && cuisine);
                count[4] = (int)(salon && reunion);
                count[5] = (int)(cuisine && reunion);
                count[6] = (int)(salon && reunion && cuisine);
                transform(total.begin(), total.end(), count.begin(), total.begin(), plus<int>());
            }
            return total;
        }



        T GetCenter() const
        {
            return mCenter;
        }

        void SetCenter(const T& center)
        {
            mCenter = center;
        }


    };
    template <class T>
    void CalculateFrontierDistances(Node<T>* node, T feature) {}

    template<>
    void CalculateFrontierDistances(Node<Mat>* node, Mat feature)
    {
        int nbChild = node->GetChildCount();
        // distMat(i,j) is the distance from feature to frontier between i and j
        // if negative, on side of i
        // if positive, on side of j
        Mat distMat(nbChild, nbChild, CV_64FC1);
        // distances(i) = max(distMat(i,:)).
        // distances(i) < RADIUS means feature in the cell of i
        vector<double> distances;
        vector<Node<Mat> *> nodeList = node->GetChildNodes();
        for (size_t i = 0; i < nbChild; i++)
        {
            Node<Mat>* lNodeI = nodeList[i];
            //Mat VecIFeature = feature - lNodeI->GetCenter();
            for (size_t j = 0; j < i; j++)
            {
                Node<Mat>* lNodeJ = nodeList[j];
                Mat vecJFeature = feature - lNodeJ->GetCenter();
                Mat vecIJ = lNodeJ->GetCenter() - lNodeI->GetCenter();
                double normIJ = norm(vecIJ, NORM_L2);
                double projection = vecJFeature.dot(vecIJ) / normIJ;
                double distI2J = normIJ / 2 - projection;
                distMat.at<double>(i, j) = distI2J;
                distMat.at<double>(j, i) = distI2J;
            }
        }
        for (size_t i = 0; i < nbChild; i++)
        {
            distMat.at<double>(i, i) = 0;
            double min, max;
            minMaxIdx(distMat.row(i), &min, &max);
            distances.push_back(max);
            nodeList[i]->SetFrontier(max);
        }
        //return vector<double>();
    }
}


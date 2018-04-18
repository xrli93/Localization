#pragma once

#include<vector>
#include<algorithm>
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


        void RemoveWords()
        {
            while (!mWords.empty())
                mWords.pop_back();
        }
        

        void AddChildNode(Node<T> *node)
        {
            mChildNodes.push_back(node);
        }

        vector<Word<T> *> GetWords() const
        {
            return mWords;
        }

        vector<Node<T> *> GetChildNodes() const
        {
            return mChildNodes;
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
}


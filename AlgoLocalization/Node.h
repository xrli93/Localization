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
            for (size_t i = 0; i < mChildNodes.size(); i++)
            {
                delete mChildNodes[i];
            }


            for (size_t i = 0; i < mWords.size(); i++)
            {
                delete mWords[i];
            }
            mWords.clear();
        }

        void AddWord(Word<T> *word)
        {
            mWords.push_back(word);
        }

        void RemoveWords()
        {
            while (!mChildNodes.empty())
                mChildNodes.pop_back();
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


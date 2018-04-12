#pragma once

#include<vector>
#include<algorithm>
#include "Word.h"

using namespace std;
template <class T>
class Node
{
private:
	T mCenter;
	std::vector<Node<T> *> mChildNodes;
	std::vector<Word<T> *> mWords;
public:
	Node()
	{
	}

	Node(T center) : mCenter(center) {};

	~Node()
	{

	}

	void AddWord(Word<T> *word)
	{
		mWords.push_back(word);
	}

	void AddChildNode(Node<T> *node)
	{
		mChildNodes.push_back(node);
	}

	vector<Word<T> *> GetWords()
	{
		return mWords;
	}

	vector<Node<T> *> GetChildNodes()
	{
		return mChildNodes;
	}

	// Sort child nodes according to their distances to a feature


	// TODO: test
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

	T SetCenter(T center)
	{
		mCenter = center;
	}


};
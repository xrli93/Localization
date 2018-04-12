#pragma once
#include"Node.h"
#include<vector>
#include<limits>
#include"Word.h"
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#define NUM_MAX_WORDS 3
#define K_SPLIT 3
#define MAX_CHILD_NUM 3
using namespace std;


template <class T>
class TreeDict
{
private:
	Node<T> mRootNode;

public:
	TreeDict()
	{
		mRootNode = Node<T>(0);
	}
	~TreeDict()
	{

	}

	Node<T>* GetRootNode()
	{
		return &mRootNode;
	}

	// add a new node into the dict (not used!)
	void AddNode(Node<T>* node)
	{
		Node<T> *lParentNode = &mRootNode;
		lParentNode->AddChildNode(node);
	}

	// Adds a new Word into the dict
	// Returns the Node it belongs
	Node<T>* AddWordToDict(Word<T>* word)
	{
		Node<T> * lParentNode = FindNearestNode(word->GetCenter());
		lParentNode->AddWord(word);
		return lParentNode;
	}

	///<summary>
	/// Search in dict for words that contains feature
	///</summary>
	vector<Word<T> *> Search(Node<T> *node, T feature, int maxChildNum = MAX_CHILD_NUM) 
	{
		//TODO
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
			node->SortChildNodes(feature);
			vector<Node<T> *> childList = node->GetChildNodes();
			for (int i = 0; i < maxChildNum; i++)
			{
				//TODO: verify 
				Node<T> *lNode = childList[i];
				if (Localization::CalculateDistance(feature, lNode->GetCenter()) < RADIUS)
				{

					vector<Word<T> *> childWordList = Search(lNode, feature, maxChildNum);
					wordList.insert(wordList.end(), childWordList.begin(), childWordList.end());
				}
			}
			
		}
		return vector<Word<T> *>();
	}


	// Find in dict the leaf node whose center is nearest to feature
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
					minDist = lDist;
					minNode = lNode;
				}
			}
		}
		return minNode;
	}

	// Expand a leafnode when it contains many words
	void Expand(Node<T>* node)
	{
		cout << "Creating new nodes" << endl;
		vector<T> centers;
		vector<int> labels;
		vector<Word<T> *> lWordList = node->GetWords();

		KMeansCluster(lWordList, labels, centers);
		cout << "Centers number" << centers.size() << endl;
		// creating new nodes at centers
		//typename vector<T>::iterator iter;
		//for (iter = centers.begin(); iter != centers.end(); iter++)
		for (size_t centerIter = 0; centerIter != centers.size(); centerIter++)
		{
			Node<T> *newNode = new Node<T>(centers[centerIter]); // use new?
																 // adding words to their nodes
			for (size_t wordsIter = 0; wordsIter < lWordList.size(); wordsIter++)
			{
				if (labels[wordsIter] == centerIter)
				{
					cout << "Word attached to new node" << endl;
					newNode->AddWord(lWordList[wordsIter]);
					// newNode.AddWord(newWord); words that belong to this center
				}
			}
			node->AddChildNode(newNode);
		}
	}

	// Add a feature to the dict
	void AddFeature(T feature, int indexRoom) // 
	{
		vector<Word<T> *> wordList = Search(&mRootNode, feature);
		if (!wordList.empty())
		{
			cout << "Word found" << endl;
			typename vector<Word<T> *>::iterator iter;
			for (iter = wordList.begin(); iter != wordList.end(); iter++)
			{
				(*iter)->UpdateLabel(indexRoom);
			}
		}
		else
		{
			cout << "No word found" << endl;
			Word<T> *newWord = new Word<T>(feature, indexRoom);
			Node<T> *node = AddWordToDict(newWord);

			if (node->GetWordsCount() > NUM_MAX_WORDS) // Many words necessary to create new nodes
			{
				Expand(node);
			}
		}
	}

	double KMeansCluster(vector<Word<T> *> wordList, vector<int> labels, vector<T> centers)
	{
		cv::Mat featureMat = MakeFeatureListFromWords(wordList);
		return cv::kmeans(featureMat, K_SPLIT, labels, cv::TermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 10, 0.1), 3, cv::KMEANS_PP_CENTERS, centers);
	}

	// TODO: To modify
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
};

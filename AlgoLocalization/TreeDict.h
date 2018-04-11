#pragma once
#include"Node.h"
#include<vector>
#include<limits>
#include"Word.h"
#define NUM_MAX_WORDS 4
#define K_SPLIT 3
using namespace std;



template <class T>
class TreeDict
{
private:
	Node<T> mRootNode;

public:
	TreeDict() 
	{
		mRootNode = Node<T>();
	}
	~TreeDict()
	{
		
	}

	// Add a new Node into the dict
	void AddNode(Node<T>& node)
	{
		T center = node.GetCenter();
		//TODO Find parent cl
		//parent.AddChildNode(child);
	}

	// Adds a new Word into the dict
	// Returns the Node it belongs
	Node<T>* AddWordToDict(Word<T>* word)
	{
		Node<T> * lParentNode = FindNearestNode(word->GetCenter());
		//TODO: search the nearest node
		lParentNode->AddWord(word);
		return lParentNode;
	}

	vector<Word<T> *> Search(T feature) // find in dict words that contain feature
	{
		//TODO
		return vector<Word<T> *>();
	}

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
				double lDist = CalculateDistance(lCenter, feature);
				if (lDist < minDist)
				{
					minDist = lDist;
					minNode = lNode;
				}
			}
		}
		return minNode;
	}

	void AddFeature(T feature, int indexRoom) // 
	{
		vector<Word<T> *> wordList = Search(feature);
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
			Node<T> node = *AddWordToDict(newWord);

			if (node.GetWordsCount() > NUM_MAX_WORDS) // Many words necessary to create new nodes
			{
				vector<T> centers = KMeansCluster(node.GetWords());
				typename vector<T>::iterator iter;
				for (iter = centers.begin(); iter != centers.end(); iter++)
				{
					Node<T> *newNode = new Node<T>(*iter); // use new?
					// newNode.AddWord(newWord); words that belong to this center 
					node.AddChildNode(newNode);
				}
			}			
		}
	}

	vector<T> KMeansCluster(vector<Word<T> *> words) 
	{
		return vector<T>(K_SPLIT);
	}
};

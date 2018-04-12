#include <iostream>
#include <iomanip>
#include "Node.h"
#include "TreeDict.h"
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace std;
//using namespace cv;

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
int main() {
	TreeDict<int> myDict = TreeDict<int>();
	myDict.AddNode(new Node<int>(1));
	myDict.AddNode(new Node<int>(3));
	myDict.AddNode(new Node<int>(2));

	myDict.AddFeature(3, 1);

	Node<int>* rootNode = myDict.GetRootNode();
	rootNode ->SortChildNodes(0);

	std::cin.get();
	return 0;
}
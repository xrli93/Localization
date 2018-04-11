#include <iostream>
#include <iomanip>
#include "Node.h"
#include "TreeDict.h"
using namespace std;
int main() {
	TreeDict<int> myDict = TreeDict<int>();
	myDict.AddFeature(1, 1);
	myDict.AddFeature(3, 2);
	myDict.AddFeature(5, 2);
	myDict.AddFeature(7, 2);
	Node<int> *parent = myDict.FindNearestNode(1);
	
	return 0;
}
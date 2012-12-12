#include "test.h"
#include <string>
#include <iostream>

#include "algorithm.h"
using namespace std;

typedef int(*Pf)();	//测试函数指针 返回0成功 否则失败

class TestCase{	
public:
	TestCase(Pf func,string descrip){
		this->func = func;
		this->descrip = descrip;
	}
	Pf func;	//测试函数指针
	string descrip;//用例描述
};

TestCase case_TestTps(TestTps,"test TSP");//tps用例

const TestCase cases[] = {	//用例集合
	case_TestTps,
};
const int cases_len = sizeof(cases)/sizeof(cases[0]);

/*测试所有用例*/
void TestAll(){
	for(int i=0; i < cases_len; ++i){
		TestCase tc = cases[i];
		cout<<tc.descrip<<"---->";
		int r = tc.func();
		if(r != 0){
			cout<<"failed";
		}else{
			cout<<"pass";
		}
		cout<<endl;
	}
}

/*测试TPS是否正确*/
int TestTps(){
	CvMat* src = cvCreateFloatMat(3,2);
	CvMat* des = cvCreateFloatMat(3,1);
	cvmSet(src,0,0,0);
	cvmSet(src,0,1,0);

	cvmSet(src,1,0,1);
	cvmSet(src,1,1,0);

	cvmSet(src,2,0,0);
	cvmSet(src,2,1,1);

	cvmSet(des,0,0,0);
	cvmSet(des,1,0,0);
	cvmSet(des,2,0,1);

	CvMat* coef = TpsTrain(src,des);
	float h = TpsPredict(6,2,src,coef);

	return h == 2 ? 0 : 1;
}
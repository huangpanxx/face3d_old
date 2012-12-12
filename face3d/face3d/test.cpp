#include "test.h"
#include <string>
#include <iostream>

#include "algorithm.h"
using namespace std;

typedef int(*Pf)();	//���Ժ���ָ�� ����0�ɹ� ����ʧ��

class TestCase{	
public:
	TestCase(Pf func,string descrip){
		this->func = func;
		this->descrip = descrip;
	}
	Pf func;	//���Ժ���ָ��
	string descrip;//��������
};

TestCase case_TestTps(TestTps,"test TSP");//tps����

const TestCase cases[] = {	//��������
	case_TestTps,
};
const int cases_len = sizeof(cases)/sizeof(cases[0]);

/*������������*/
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

/*����TPS�Ƿ���ȷ*/
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
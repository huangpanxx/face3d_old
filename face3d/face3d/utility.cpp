#include "utility.h"
#include <fstream>
#include <iostream>
using namespace std;

CvMat* ReadMatFromText(std::string file,int rows,int column){
	CvMat* mat = cvCreateMat(rows,column,CV_32FC1);
	ifstream fin(file);
	if(!fin) Log("读取文件" + file + "出错");
	for(int i = 0;i<rows;++i){
		for(int j=0;j<column;++j){
			double value;
			fin>>value;
			cvmSet(mat,i,j,value);
		}
	}
	return mat;
}

CvMat* ReadMatFromBinary(string file,int rows,int column){
	CvMat* mat = cvCreateMat(rows,column,CV_32FC1);
	ifstream fin(file,ios::binary);
	if(!fin) Log("读取文件" + file + "出错");
	for(int i = 0;i<rows;++i){
		for(int j=0;j<column;++j){
			float value;
			fin.read((char*)(&value),sizeof(float));
			cvmSet(mat,i,j,value);
		}
	}
	return mat;
}


void WriteMatToText(string file,CvMat* mat){
	ofstream fout(file);
	for(int i=0;i<mat->rows;++i){
		for(int j=0;j<mat->cols;++j){
			double v = cvmGet(mat,i,j);
			fout<<v<<"\t";
		}
		fout<<endl;
	}
	fout.close();
}

void Log(std::string msg){
	cout<<msg<<endl;
}

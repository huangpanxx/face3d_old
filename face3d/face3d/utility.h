#ifndef UTILITY_H
#define UTILITY_H

/*
	author		:	snail
	create		:	2012/11/24
	desription	:	���ļ�����һЩ��������
*/



#include <opencv\cv.h>
#include <string>
#include <vector>

/*��־*/
void Log(std::string msg);

/*����д�뵽�ı��ļ�*/
void WriteMatToText(std::string file,CvMat* mat);

/*�ı��ļ���ȡ����*/
CvMat* ReadMatFromText(std::string file,int rows,int column);

/*�������ļ���ȡ����*/
CvMat* ReadMatFromBinary(std::string file,int rows,int column);

#endif
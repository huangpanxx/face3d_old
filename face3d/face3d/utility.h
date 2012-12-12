#ifndef UTILITY_H
#define UTILITY_H

/*
	author		:	snail
	create		:	2012/11/24
	desription	:	该文件包含一些辅助函数
*/



#include <opencv\cv.h>
#include <string>
#include <vector>

/*日志*/
void Log(std::string msg);

/*矩阵写入到文本文件*/
void WriteMatToText(std::string file,CvMat* mat);

/*文本文件读取矩阵*/
CvMat* ReadMatFromText(std::string file,int rows,int column);

/*二进制文件读取矩阵*/
CvMat* ReadMatFromBinary(std::string file,int rows,int column);

#endif
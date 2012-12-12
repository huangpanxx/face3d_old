#ifndef ALGORITHM_H
#define ALGORITHM_H

/*
	author		:	snail
	create		:	2012/11/22
	desription	:	该头文件包含一些通用算法
*/

#include <opencv\cv.h>
#include <vector>

#define esp (1e-7)
#define ZERO(x) (abs(x)<=esp)
#define NaN (-500000)
#define IsNaN(x) (ZERO((x)-NaN))
#define PI (3.14159f)
#define cvCreateFloatMat(rows,cols) (cvCreateMat(rows,cols,CV_32FC1))
#define E (2.71828f)

/*变形参数*/
class TransformParam {
public:
	TransformParam(float ax=1,float ay=0,float tx=0,float ty=0) {
		this->ax = ax;
		this->ay = ay;
		this->tx = tx;
		this->ty = ty;
	}
	float ax; //s*cos(alpha)
	float ay; //s*sin(alpha)
	float tx;
	float ty;
};

/* 计算两点距离 */
float GetDistance(float x1,float y1,float x2,float y2);

/*矩阵变形*/
CvMat* Transform(const CvMat *src,const TransformParam& param);

/*图像变形*/
IplImage* TransformImage(IplImage *image,const int width,const int height,const TransformParam param);

/*对齐shape到_template*/
TransformParam Alignment(const CvMat* _template,const CvMat* shape);

/*在image上绘制点*/
void drawPoints(IplImage *image,CvMat* pts,CvScalar color = CV_RGB(255,0,0));

/*合并变形参数*/
TransformParam CombineTransformParam(const TransformParam &param1,const TransformParam &param2);

/*变形参数求逆*/
TransformParam InvTransformParam(const TransformParam &param);

/*薄板样条系数计算 输出插值系数向量*/
CvMat* TpsTrain(CvMat* src,/* 源点n*2矩阵 */
				CvMat* des /* 目标值 n*1 矩阵 */
				);

/*薄板样条函数预测 输出预测值*/
float TpsPredict(
	float x,	/* 横坐标 */
	float y,	/* 纵坐标 */
	CvMat* src,	/* 用于计算插值系数的源点n*2矩阵 */
	CvMat* coefficient/* 插值系数向量 */
	);

/*转为为灰度图像矩阵*/
CvMat* ImageToMat(IplImage* image);

/*向量转化为矩阵*/
CvMat* VecToMat(const std::vector<float>& v,const int rows,const int cols);

/* rgb转化到灰度 */
int inline RgbToGray(int r,int g,int b){
	return 0.3 * r + 0.6 * g + 0.1 * b;
};
#endif
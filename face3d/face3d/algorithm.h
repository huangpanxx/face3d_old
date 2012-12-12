#ifndef ALGORITHM_H
#define ALGORITHM_H

/*
	author		:	snail
	create		:	2012/11/22
	desription	:	��ͷ�ļ�����һЩͨ���㷨
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

/*���β���*/
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

/* ����������� */
float GetDistance(float x1,float y1,float x2,float y2);

/*�������*/
CvMat* Transform(const CvMat *src,const TransformParam& param);

/*ͼ�����*/
IplImage* TransformImage(IplImage *image,const int width,const int height,const TransformParam param);

/*����shape��_template*/
TransformParam Alignment(const CvMat* _template,const CvMat* shape);

/*��image�ϻ��Ƶ�*/
void drawPoints(IplImage *image,CvMat* pts,CvScalar color = CV_RGB(255,0,0));

/*�ϲ����β���*/
TransformParam CombineTransformParam(const TransformParam &param1,const TransformParam &param2);

/*���β�������*/
TransformParam InvTransformParam(const TransformParam &param);

/*��������ϵ������ �����ֵϵ������*/
CvMat* TpsTrain(CvMat* src,/* Դ��n*2���� */
				CvMat* des /* Ŀ��ֵ n*1 ���� */
				);

/*������������Ԥ�� ���Ԥ��ֵ*/
float TpsPredict(
	float x,	/* ������ */
	float y,	/* ������ */
	CvMat* src,	/* ���ڼ����ֵϵ����Դ��n*2���� */
	CvMat* coefficient/* ��ֵϵ������ */
	);

/*תΪΪ�Ҷ�ͼ�����*/
CvMat* ImageToMat(IplImage* image);

/*����ת��Ϊ����*/
CvMat* VecToMat(const std::vector<float>& v,const int rows,const int cols);

/* rgbת�����Ҷ� */
int inline RgbToGray(int r,int g,int b){
	return 0.3 * r + 0.6 * g + 0.1 * b;
};
#endif
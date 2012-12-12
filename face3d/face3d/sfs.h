#ifndef SFS_H
#define SFS_H
#include <opencv/cv.h>
#include "algorithm.h"

const int D = 2;	//�ظ����������
const float ZLIMIT = -40.f;	//�������

/*�����ֵ����*/
CvMat* InterpolateToGrid(const std::vector<float>& vx,
						 const std::vector<float>& vy,
						 const std::vector<float>& vz,
						 int rows,int cols,
						 int offsetX = 0,int offsetY = 0);

/* �����ֵ���� srcΪn*3�㼯�� rows colsΪ����߿� */
CvMat* InterpolateToGrid(CvMat* src,int rows,int cols,int x = 0,int y = 0);

/* �������ϵ�� Z�ο������������ IĿ�������Ҷ�  P�ο������Ҷ�  ���Ϊ1*4����ϵ������ [l0 l1 l2 l3] */
CvMat* EstimateLightCoff(CvMat* Z,CvMat* I,CvMat* P);

/* ��ȹ��� Z�ο������������ IĿ�������Ҷ� P�ο������Ҷ� L����ϵ��*/
CvMat* EstimateZ(CvMat* Z,CvMat* I,CvMat* P,CvMat* L,float segma = 9.0f, float lambda = 25.0f);

#endif
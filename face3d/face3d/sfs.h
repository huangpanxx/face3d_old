#ifndef SFS_H
#define SFS_H
#include <opencv/cv.h>
#include "algorithm.h"

const int D = 2;	//重复点过滤设置
const float ZLIMIT = -40.f;	//深度下限

/*计算插值网格*/
CvMat* InterpolateToGrid(const std::vector<float>& vx,
						 const std::vector<float>& vy,
						 const std::vector<float>& vz,
						 int rows,int cols,
						 int offsetX = 0,int offsetY = 0);

/* 计算插值网格 src为n*3点集合 rows cols为网格高宽 */
CvMat* InterpolateToGrid(CvMat* src,int rows,int cols,int x = 0,int y = 0);

/* 计算光线系数 Z参考人脸深度网格 I目标人脸灰度  P参考人脸灰度  输出为1*4光线系数向量 [l0 l1 l2 l3] */
CvMat* EstimateLightCoff(CvMat* Z,CvMat* I,CvMat* P);

/* 深度估计 Z参考人脸深度网格 I目标人脸灰度 P参考人脸灰度 L光线系数*/
CvMat* EstimateZ(CvMat* Z,CvMat* I,CvMat* P,CvMat* L,float segma = 9.0f, float lambda = 25.0f);

#endif
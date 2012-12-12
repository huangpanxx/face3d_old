#ifndef DEFORM_H
#define DEFORM_H

/*
	author		:	snail
	create		:	2012/11/23
	desription	:	该文件包含整个重建过程相关的函数,并对asm/asm.h的c-style函数进行封装
	重建过程使用cvMat作为基本数据结构
*/

#include <opencv\cv.h>
#include <string>
#include "algorithm.h"



const int FEATURES_NUMBER = 76;	/*使用的特征点数量*/
const int FEATURE_VECTORS_NUMBER = 102; /* 特征向量数量 */
const int ALL_POINTS_NUMBER = 4980; /* 三维人脸点的数量 */
const float TRAIN_FACE_SIZE = 200.0f;	/* 训练数据中的标准人脸大小 */

const int IMAGE_SIZE = 200;	//输出图像边框大小
const int FACE_SIZE = 200;	//输出人脸大小

/* 需要手动销毁传入的mat */
class FaceFatureParams : public TransformParam {
public:
	FaceFatureParams(
		CvMat* feature_params,
		float ax,float ay,
		float tx,float ty,
		bool is_auto_release_feature_params = true
		);
	~FaceFatureParams();
	CvMat *feature_params;
	bool is_auto_release_feature_params;
};

/*初始化系统 包括ASM数据以及人脸训练数据*/
void Initialize(const std::string data_dir);

/*计算特征点到标准人脸对齐参数*/
TransformParam AlignToStandard(const CvMat* features);

/*定位人脸特征点 输出为n*2矩阵 包含每个特征点坐标 */
CvMat* Search(const std::string image_path);

/* 计算特征向量系数 注意输入特征点需要对齐到标准人脸 */
FaceFatureParams ComputeFaceFeatureParams(const CvMat* features);

/* 得到三维人脸所有点 (n*3矩阵) */
CvMat* Compute3dFace(const CvMat *alpha);

/* 校正人脸 */
CvMat* Rectify3dFace(
	const CvMat *face3d, /* 人脸坐标系下的三维人脸 n*3 矩阵 */
	const CvMat *features /* 人脸坐标系下的特征点 m*2 矩阵 */
	);

#endif
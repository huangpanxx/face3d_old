#ifndef DEFORM_H
#define DEFORM_H

/*
	author		:	snail
	create		:	2012/11/23
	desription	:	���ļ����������ؽ�������صĺ���,����asm/asm.h��c-style�������з�װ
	�ؽ�����ʹ��cvMat��Ϊ�������ݽṹ
*/

#include <opencv\cv.h>
#include <string>
#include "algorithm.h"



const int FEATURES_NUMBER = 76;	/*ʹ�õ�����������*/
const int FEATURE_VECTORS_NUMBER = 102; /* ������������ */
const int ALL_POINTS_NUMBER = 4980; /* ��ά����������� */
const float TRAIN_FACE_SIZE = 200.0f;	/* ѵ�������еı�׼������С */

const int IMAGE_SIZE = 200;	//���ͼ��߿��С
const int FACE_SIZE = 200;	//���������С

/* ��Ҫ�ֶ����ٴ����mat */
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

/*��ʼ��ϵͳ ����ASM�����Լ�����ѵ������*/
void Initialize(const std::string data_dir);

/*���������㵽��׼�����������*/
TransformParam AlignToStandard(const CvMat* features);

/*��λ���������� ���Ϊn*2���� ����ÿ������������ */
CvMat* Search(const std::string image_path);

/* ������������ϵ�� ע��������������Ҫ���뵽��׼���� */
FaceFatureParams ComputeFaceFeatureParams(const CvMat* features);

/* �õ���ά�������е� (n*3����) */
CvMat* Compute3dFace(const CvMat *alpha);

/* У������ */
CvMat* Rectify3dFace(
	const CvMat *face3d, /* ��������ϵ�µ���ά���� n*3 ���� */
	const CvMat *features /* ��������ϵ�µ������� m*2 ���� */
	);

#endif
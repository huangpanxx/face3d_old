#include "morphing.h"
#include "asm\asm.h"
#include <fstream>
#include "algorithm.h"
#include "utility.h"

using namespace std;

CvMat *sf_bar = NULL;
CvMat *s_bar  = NULL;
CvMat *pf	  =	NULL;
CvMat *p	  =	NULL;


FaceFatureParams::FaceFatureParams(
	CvMat* feature_params,
	float ax,float ay,
	float tx,float ty,
	bool is_auto_release_feature_params
	) : TransformParam(ax,ay,tx,ty)
{
	this->feature_params = feature_params;
	this->is_auto_release_feature_params = is_auto_release_feature_params;
}

FaceFatureParams::~FaceFatureParams(){
	if(this->is_auto_release_feature_params){
		cvReleaseMat(&this->feature_params);	
	}
}


void Initialize(const std::string data_dir){
	const std::string MODEL1_CONF_FILE = "mu-68-1d.conf";
	const std::string MODEL2_CONF_FILE = "mu-76-2d.conf";
	string conf1 = data_dir + "/" + MODEL1_CONF_FILE;
	string conf2 = data_dir + "/" + MODEL2_CONF_FILE;

	Log("Read ASM data...");
	InitModels(conf1.c_str(),conf2.c_str(),data_dir.c_str());
	
	//��ʼ��ѵ������
	Log("Read mophable data...");
	sf_bar	= ReadMatFromBinary(data_dir + "/sf_bar", 1, FEATURES_NUMBER * 2);	//�������ֵ
	pf		= ReadMatFromBinary(data_dir + "/pf", FEATURE_VECTORS_NUMBER, FEATURES_NUMBER * 2);	//��������������
	s_bar	= ReadMatFromBinary(data_dir + "/s_bar", 1, ALL_POINTS_NUMBER * 3);	//������ֵ
	p		= ReadMatFromBinary(data_dir + "/p", FEATURE_VECTORS_NUMBER, ALL_POINTS_NUMBER * 3);	//������������

}


TransformParam AlignToStandard(const CvMat* points){
	CvMat mat = cvMat(FEATURES_NUMBER,2,CV_32FC1,sf_bar->data.fl);
	return Alignment(&mat,points);
}


CvMat* Search(const std::string image_path){
	double points[FEATURES_NUMBER][2];
	int n = AsmSearch((char*)image_path.c_str(),points);
	CvMat *m = NULL;
	if(n != 0){
		m = cvCreateMat(FEATURES_NUMBER,2,CV_32FC1);
		for(int i=0;i<FEATURES_NUMBER;++i){
			cvmSet(m,i,0,points[i][0]);
			cvmSet(m,i,1,points[i][1]);
		}
	}
	return m;
}

CvMat* Compute3dFace(const CvMat *alpha){
	assert(alpha->rows == 1);
	assert(alpha->cols == FEATURE_VECTORS_NUMBER);
	CvMat *face3d = cvCreateMat(ALL_POINTS_NUMBER,3,CV_32FC1);
	CvMat v_face3d = cvMat(1,ALL_POINTS_NUMBER*3,CV_32FC1,face3d->data.fl);
	cvMatMulAdd(alpha,p,s_bar,&v_face3d);
	return face3d;
}

 FaceFatureParams ComputeFaceFeatureParams(const CvMat *_features){
	assert(_features->rows == FEATURES_NUMBER);
	assert(_features->cols == 2);

	/* �������� */
	TransformParam p1 = AlignToStandard(_features);
	CvMat *features = Transform(_features,p1); //�任����׼��������ϵ

	/* ����ƽ�Ʋ��� */
	float tx_sum = 0;
	float ty_sum = 0;
	for(int i = 0; i < FEATURES_NUMBER; ++i){
		tx_sum += cvmGet(features,i,0);
		ty_sum += cvmGet(features,i,1);
	}

	float tx = tx_sum / FEATURES_NUMBER;
	float ty = ty_sum / FEATURES_NUMBER;

	/* ת��Ϊ���� */
	CvMat* sf = cvCreateMat(1,FEATURES_NUMBER*2,CV_32FC1);  //���� 2n����
	for(int i = 0; i < FEATURES_NUMBER; ++i){//������������ƽ�Ʋ�ת��Ϊ2n����
		cvmSet(sf,0,2*i,cvmGet(features,i,0) - tx);
		cvmSet(sf,0,2*i+1,cvmGet(features,i,1) - ty);
	}

	/* ������ʱ�����ռ䲢����һЩ���� */
	const float lambda = 2.0f;
	CvMat *_sf		= cvCloneMat(sf_bar);	//��ʼֵΪsf_bar
	CvMat *dsf		= cvCreateMat(1, FEATURES_NUMBER * 2, CV_32FC1); //_sf - sf_bar
	CvMat *pf_trans = cvCreateMat(pf->cols,pf->rows,CV_32FC1); cvTranspose(pf,pf_trans);	//pf'
	CvMat *alpha	= cvCreateMat(1,FEATURE_VECTORS_NUMBER,CV_32FC1);	//����������ϲ���
	
	
	float old_c = -1000;
	float c = 1;
	/*���￪ʼ��������*/
	for(int i = 0; i < 10 ; ++i){
		//��������ϵ��
		float a = 0, b = 0;
		for(int i = 0; i < FEATURES_NUMBER * 2; ++i){//����sf,_sf�ڻ��Լ�_sf����
			float p1 = cvmGet(sf,0,i);
			float p2 = cvmGet(_sf,0,i);
			a += p1 * p2;
			b += p2 * p2; 
		}

		c = a / b; //����ϵ��
		float dt = abs((c-old_c)*100/c); //���
		old_c = c;	//����ֵ

		if( i >= 1){
			cout<<"Compute feature vector coefficent,iter:"<<i<<",error"<<dt<<"%"<<endl;
			if(dt < 0.0001){//������0000.1%
				cout<<"break"<<endl;
				break;
			}
		}

		//����_sf
		for(int i = 0;i < FEATURES_NUMBER * 2; ++i){
			float p = cvmGet(sf,0,i) / c;
		}

		//��ֵ
		cvSub(sf,sf_bar,dsf);
		//����alpha
		cvMatMul(dsf,pf_trans,alpha);

		//����_sf
		cvMatMulAdd(alpha,pf,sf_bar,_sf);
	}

	/* �ͷſռ� */
	cvReleaseMat(&sf);
	cvReleaseMat(&_sf);
	cvReleaseMat(&dsf);
	cvReleaseMat(&pf_trans);
	cvReleaseMat(&features);
	/* �ϲ����α仯 */
	TransformParam p2(c,0,-tx,-ty);
	TransformParam p = CombineTransformParam(p1,p2);

	return FaceFatureParams(alpha,p.ax,p.ay,p.tx,p.ty,true);
}


vector<int> GetReliablePoints(const CvMat *face3d,const CvMat *features){
	assert(face3d->rows == ALL_POINTS_NUMBER);
	assert(face3d->cols == 3);
	assert(features->rows == FEATURES_NUMBER);
	assert(features->cols == 2);
	
	float var = 0;	//����
	for(int i = 0; i < FEATURES_NUMBER; ++i){
		float x1 = cvmGet(face3d,i,0);
		float y1 = cvmGet(face3d,i,1);
		float x2 = cvmGet(features,i,0);
		float y2 = cvmGet(features,i,1);
		float d = GetDistance(x1,y1,x2,y2);
		var += d*d;
	}

	var = sqrt(var / FEATURES_NUMBER);

	vector<int> v;
	for(int i = 0; i < FEATURES_NUMBER; ++i){
		float x1 = cvmGet(face3d,i,0);
		float y1 = cvmGet(face3d,i,1);
		float x2 = cvmGet(features,i,0);
		float y2 = cvmGet(features,i,1);
		float d = GetDistance(x1,y1,x2,y2);
		if( d < 10 * var){
			v.push_back(i);
		}
	}

	return v;
}

CvMat* Rectify3dFace(const CvMat *face3d, const CvMat *features){
	assert(face3d->rows == ALL_POINTS_NUMBER);
	assert(face3d->cols == 3);
	assert(features->rows == FEATURES_NUMBER);
	assert(features->cols == 2);

	/*
	����ɿ��ĵ�
	*/
	const vector<int> control_points = GetReliablePoints(face3d,features);
	const int control_points_size = control_points.size();
	
	CvMat* src			= cvCreateFloatMat(control_points_size,2);
	CvMat* des_x		= cvCreateFloatMat(control_points_size,1);
	CvMat* des_y		= cvCreateFloatMat(control_points_size,1);
	CvMat* new_face3d	= cvCloneMat(face3d);

	Log("Rectify control points:");
	/* ������� */
	for(int i = 0; i < control_points_size; ++i){
		int index = control_points[i];
		float x_src = cvmGet(face3d,index,0);
		float y_src = cvmGet(face3d,index,1);
		float x_des = cvmGet(features,index,0);
		float y_des = cvmGet(features,index,1);
		cvmSet(src,i,0,x_src);
		cvmSet(src,i,1,y_src);
		cvmSet(des_x,i,0,x_des);
		cvmSet(des_y,i,0,y_des);
		cout<<index<<"\t";
	}
	cout<<endl;

	/* �����ֵϵ�� */
	CvMat* coff_x = TpsTrain(src,des_x);
	CvMat* coff_y = TpsTrain(src,des_y);
	
	/* �������� */
	for(int i = 0; i < ALL_POINTS_NUMBER; ++i){
		float x = cvmGet(face3d,i,0);
		float y = cvmGet(face3d,i,1);
		float new_x = TpsPredict(x,y,src,coff_x);
		float new_y = TpsPredict(x,y,src,coff_y);
		cvmSet(new_face3d,i,0,new_x);
		cvmSet(new_face3d,i,1,new_y);
	}

	/* �ͷſռ� */
	cvReleaseMat(&src);
	cvReleaseMat(&des_x);
	cvReleaseMat(&des_y);
	cvReleaseMat(&coff_x);
	cvReleaseMat(&coff_y);

	return new_face3d;
}
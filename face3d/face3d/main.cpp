#include <Windows.h>
#include "morphing.h"
#include "utility.h"
#include "sfs.h"
#include <opencv\highgui.h>
using namespace std;

/* 全局变量 */
string GetModulePath();
static string APP_PATH = GetModulePath();

string GetModulePath()
{
	static char name[255];
	GetModuleFileNameA(NULL,name,255);
	string str = name;
	return str.substr(0, str.rfind('\\')); 
}


/* 恢复步骤 */
void Init()
{
	Log("Initialize");
	Initialize(APP_PATH + "/data/");
}

bool MorphingProcess(const string image_path,IplImage*& out_image, CvMat*& out_face3d)
{
	Log("Input:" + image_path);

	Log("Locate feature points");
	CvMat* _features = Search(image_path);
	if(NULL == _features){
		Log("Locating failed!");
		return false;
	}
	
	Log("Compute feature vector coefficient");
	FaceFatureParams feature_param =  ComputeFaceFeatureParams(_features);

	Log("Compute 3d face points(basic)");
	CvMat* _face3d = Compute3dFace(feature_param.feature_params);


	Log("Rectify 3d face points");
	CvMat* features = Transform(_features,feature_param);	//标准大小的特征点
	CvMat* rectified_face3d = Rectify3dFace(_face3d,features);	//用标准大小的特征点来校正三维人脸
	cvReleaseMat(&_features);
	cvReleaseMat(&_face3d);


	Log("Align the face picture");
	IplImage *_image = cvLoadImage(image_path.c_str());	//图片
	IplImage *aligned_image = TransformImage(_image,IMAGE_SIZE,IMAGE_SIZE,
		CombineTransformParam(
		feature_param,
		TransformParam(((float)FACE_SIZE)/TRAIN_FACE_SIZE,0,IMAGE_SIZE/2,IMAGE_SIZE/2)
	));
	cvReleaseImage(&_image);



	Log("Align the 3d face");
	CvMat* aligned_face3d = Transform(rectified_face3d,
		TransformParam(((float)FACE_SIZE)/TRAIN_FACE_SIZE,0,aligned_image->width/2,aligned_image->height/2));
	cvReleaseMat(&rectified_face3d);
	
	out_image = aligned_image;
	out_face3d = aligned_face3d;
}

bool SfsProcess(IplImage* image,CvMat* face_ref,CvMat*& out_face3d,float segma = 9.0f,float lambda = 25.0f)
{
	Log("Read reference skin");
	CvMat *rgb  = ReadMatFromText((APP_PATH + "/data/albedo.txt").c_str(),ALL_POINTS_NUMBER,3);
	vector<float> x,y,z,gray;
	x.reserve(rgb->rows); y.reserve(rgb->rows); z.reserve(rgb->rows);
	for(int i = 0;i<rgb->rows;++i)
	{
		int _x = cvmGet(face_ref,i,0); int _y = cvmGet(face_ref,i,1);float _z = cvmGet(face_ref,i,2);
		float r = cvmGet(rgb,i,0); float g = cvmGet(rgb,i,1); float b = cvmGet(rgb,i,2);
		x.push_back(_x);y.push_back(_y);z.push_back(_z);gray.push_back(RgbToGray(r,g,b));
	}
	cvReleaseMat(&rgb);

	Log("Interpolate reference skin");
	CvMat* P = InterpolateToGrid(x,y,gray,IMAGE_SIZE,IMAGE_SIZE);

	Log("Interpolate reference face");
	CvMat* Z = InterpolateToGrid(x,y,z,IMAGE_SIZE,IMAGE_SIZE);


	Log("Estimate light coefficient");
	CvMat *I = ImageToMat(image);
	CvMat* L = EstimateLightCoff(Z,I,P);
	for(int i = 0; i < 4;++i){
		float v = cvmGet(L,0,i);
		cout<<"l"<<i<<"="<<v<<"\t";
	}
	cout<<endl;

	Log("Construct depth matrix");
	CvMat* D = EstimateZ(Z,I,P,L,segma,lambda);
	cvReleaseMat(&Z); cvReleaseMat(&I); cvReleaseMat(&P); cvReleaseMat(&Z);
	out_face3d = D;
	return true;
}


int main(int argc, char **argv) {
	bool ok = false;
	do
	{
		if(argc < 2)
		{
			Log("Usage: face3d.exe picture_path <segma default=9.0f> <lambda default=25.0f>");
			Log("Output:");
			Log("F:\ta morph 3d model"); 
			Log("A:\ta sparse matrix");
			Log("B:\ta dense matrix");
			Log("BOUND:\t the boundary points");
			Log("I.bmp:\t face picture");
			break;
		}
		//配置参数
		string image_path = APP_PATH + "/data/face.bmp";
		float segma = 9.0f;
		float lambda = 25.0f;	
		if(argc >= 2)	image_path = argv[1];
		if(argc >= 3)	segma = atof(argv[2]);
		if(argc >= 4)	lambda = atof(argv[3]);
		
		//临时变量
		IplImage *image = NULL; 
		CvMat *moph_face  = NULL;		CvMat* sfs_face = NULL;
		//初始化
		Init();	
		//构造形变模型
		ok = MorphingProcess(image_path,image,moph_face);	
		if(!ok) break;
		Log("Write the mophable model(file \"F\")");
		WriteMatToText("F",moph_face);
		
		//光照模型
		ok = SfsProcess(image, moph_face, sfs_face,segma,lambda);
		if(!ok) break;
		cvReleaseMat(&moph_face); cvReleaseMat(&sfs_face); cvReleaseImage(&image);
		Log("Complete!");
	}while (false);
	system("pause");
	return ok ? 0:-1;
}

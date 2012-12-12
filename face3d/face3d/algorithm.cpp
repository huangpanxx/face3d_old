#include "algorithm.h"
#include <cmath>

using namespace std;


TransformParam Alignment(const CvMat* s1,const CvMat* s2){
	float X1 = 0; float Y1 = 0;
	float X2 = 0; float Y2 = 0;
	float C1 = 0; float C2 = 0;
	float Z  = 0; float W  = 0; 
	int len = s1->rows;

	//矩阵参数
	for(int i = 0; i < len; ++i){
		float _w = 1;
		X1 += _w * cvmGet(s1,i,0);
		Y1 += _w * cvmGet(s1,i,1);

		X2 += _w * cvmGet(s2,i,0);
		Y2 += _w * cvmGet(s2,i,1);

		C1 += _w * (cvmGet(s1,i,0) * cvmGet(s2,i,0) + cvmGet(s1,i,1) * cvmGet(s2,i,1));
		C2 += _w * (cvmGet(s1,i,1) * cvmGet(s2,i,0) - cvmGet(s1,i,0) * cvmGet(s2,i,1));
		Z  += _w * (cvmGet(s2,i,0) * cvmGet(s2,i,0) + cvmGet(s2,i,1) * cvmGet(s2,i,1));
		W  += _w;
	}

	float A[] = 
	{
		X2, -Y2,   W,  0 ,
		Y2,  X2,   0,  W ,
		Z,    0,  X2, Y2 ,
		0,    Z, -Y2, X2 , 
	};
	float B[4] = 
	{
		X1,
		Y1,
		C1,
		C2,
	};

	CvMat a = cvMat(4,4,CV_32FC1,A);
	CvMat b = cvMat(4,1,CV_32FC1,B);
	CvMat *x = cvCreateMat(4,1,CV_32FC1);
	cvSolve(&a,&b,x);
	TransformParam p;
	p.ax = cvmGet(x,0,0);
	p.ay = cvmGet(x,1,0);
	p.tx = cvmGet(x,2,0);
	p.ty = cvmGet(x,3,0);
	cvReleaseMat(&x);
	return p;
}

CvMat* Transform(const CvMat *src,const TransformParam& param){
	CvMat *des = cvCloneMat(src);
	for(int i=0;i<src->rows;++i){
		float x = cvmGet(src,i,0)*param.ax - cvmGet(src,i,1)*param.ay + param.tx;
		float y = cvmGet(src,i,0)*param.ay + cvmGet(src,i,1)*param.ax + param.ty;
		cvmSet(des,i,0,x);
		cvmSet(des,i,1,y);
		if(src->cols == 3)
		{
			float r = sqrt( param.ax*param.ax + param.ay*param.ay);
			cvmSet(des,i,2,cvmGet(src,i,2)*r);
		}
	}
	return des;
}

/* 临时使用 目前输出图像和输入图像尺寸一致,存在截断,空白等问题 */
IplImage* TransformImage(IplImage *image,const int width,const int height,const TransformParam param){
	float r = param.ax*param.ax + param.ay*param.ay;
	
	IplImage *pImage = cvCreateImage(cvSize(width,height),image->depth,image->nChannels);

	for(int x = 0;x<width;++x){
		for(int y = 0;y<height;++y){
			int _x = (param.ay*(y-param.ty) + param.ax*(x-param.tx))/r + 0.5;
			int _y = (param.ay*(-x+param.tx) - param.ax*(-y+param.ty))/r + 0.5;
			if( _x >=0 && _x < image->width &&
				_y>=0 && _y < image->height &&
				x >= 0 && x< width &&
				y >=0 && y < height ){
					int des =  y * pImage->widthStep + x*pImage->nChannels;
					int src = _y * image->widthStep + _x*image->nChannels;
					for(int i=0;i<image->nChannels;++i){
						pImage->imageData[des+i] = image->imageData[src+i];
					}
			}
		}
	}
	return pImage;
}


void drawPoints(IplImage *image,CvMat* pts,CvScalar color){
	for(int i = 0; i < pts->rows; ++i){
		CvPoint pt;
		pt.x = cvmGet(pts,i,0);
		pt.y = cvmGet(pts,i,1);
		cvCircle( image, pt ,1 , color);
	}
}


TransformParam CombineTransformParam(const TransformParam &param1,const TransformParam &param2){
	float ax = param2.ax*param1.ax - param2.ay*param1.ay;
	float ay = param2.ay*param1.ax + param2.ax*param1.ay;
	float tx = param2.ax*param1.tx - param2.ay*param1.ty + param2.tx;
	float ty = param2.ay*param1.tx + param2.ax*param1.ty + param2.ty;
	return TransformParam(ax,ay,tx,ty);
}

TransformParam InvTransformParam(const TransformParam &param){
	float r = param.ax*param.ax + param.ay*param.ay;
	float ax = param.ax / r;
	float ay = -param.ay / r;
	float tx = -(param.ax*param.tx + param.ay*param.ty)/r;
	float ty = (param.ay*param.tx - param.ax*param.ty)/r;
	return TransformParam(ax,ay,tx,ty);
}


float TpsBaseFunc(float r){
	if(ZERO(r))	return r;
	else return r * r * log(r);
}

float GetDistance(float x1,float y1,float x2,float y2){
	float dx = x1-x2;
	float dy = y1-y2;
	return sqrt(dx*dx + dy*dy);
}

CvMat* TpsTrain(CvMat* src,CvMat* des){
	assert(src->rows == des->rows);
	assert(src->cols == 2);
	assert(des->cols == 1);
	int n = src->rows;
	/*申请空间*/
	CvMat* A = cvCreateMat(n+3,n+3,CV_32FC1);
	CvMat* B = cvCreateMat(n+3,1,CV_32FC1);
	CvMat* x = cvCreateMat(n+3,1,CV_32FC1);
	
	/*填充矩阵A*/
	//填充左上矩阵K(p*p)
	for(int i=0;i<n;++i){
		for(int j=i;j<n;++j){
			if( i == j) cvmSet(A,i,j,0);
			else{
				float x1 = cvmGet(src,i,0);float y1 = cvmGet(src,i,1);
				float x2 = cvmGet(src,j,0);float y2 = cvmGet(src,j,1);
				float u = TpsBaseFunc(GetDistance(x1,y1,x2,y2));
				cvmSet(A,i,j,u);
				cvmSet(A,j,i,u);
			}
		}
	}
	//填充右上及左下角矩阵P,P'
	for(int i = 0; i < n; ++i){
		float x = cvmGet(src,i,0);
		float y = cvmGet(src,i,1);
		//右上
		cvmSet(A,i,n+0,1);
		cvmSet(A,i,n+1,x);
		cvmSet(A,i,n+2,y);
		//左下
		cvmSet(A,n+0,i,1);
		cvmSet(A,n+1,i,x);
		cvmSet(A,n+2,i,y);
	}
	//填充右下角3*3矩阵
	for(int i=0;i<3;++i){
		for(int j=0;j<3;++j){
			cvmSet(A,n+i,n+j,0);
		}
	}
	/*填充矩阵B*/
	//填充上方v
	for(int i=0;i<n;++i){
		float v = cvmGet(des,i,0);
		cvmSet(B,i,0,v);
	}
	//填充下方0
	cvmSet(B,n+0,0,0);
	cvmSet(B,n+1,0,0);
	cvmSet(B,n+2,0,0);
	/*求解*/
	cvSolve(A,B,x);
	/*释放空间*/
	cvReleaseMat(&A);
	cvReleaseMat(&B);
	return x;
}


float TpsPredict(float x, float y, CvMat* src, CvMat* coefficient){
	assert(src->cols == 2);
	assert(src->rows + 3 == coefficient->rows);
	assert(coefficient->cols == 1);
	int n = src->rows;
	float a1 = cvmGet(coefficient,n+0,0);
	float a2 = cvmGet(coefficient,n+1,0);
	float a3 = cvmGet(coefficient,n+2,0);
	float h = a1 + a2*x + a3*y;
	for(int i=0; i<n ; ++i){
		float w = cvmGet(coefficient,i,0);
		float px = cvmGet(src,i,0);
		float py = cvmGet(src,i,1);
		h += w * TpsBaseFunc(GetDistance(x,y,px,py));
	}
	return h;
}


CvMat* ImageToMat(IplImage* image){
	assert(image->nChannels == 3);//ensure rgb picture
	CvMat* mat = cvCreateFloatMat(image->height,image->width);
	for(int x = 0;x < image->width; ++x){
		for(int y = 0;y < image->height; ++y){
			int idx = y * image->widthStep + x * image->nChannels;
			CvScalar s = cvGet2D(image,y,x);
			uchar b = s.val[0];
			uchar g = s.val[1];
			uchar r = s.val[2];
			uchar gray = RgbToGray(r,g,b);
			cvmSet(mat,y,x,gray);
		}
	}
	return mat;
}


CvMat* VecToMat(const vector<float>& v,const int rows,const int cols){
	CvMat* mat = cvCreateFloatMat(rows,cols);
	for(int r = 0; r < rows; ++r){
		for(int c = 0; c < cols; ++c){
			cvmSet(mat,r,c,v[r*cols+c]);
		}
	}
	return mat;
}

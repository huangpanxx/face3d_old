#include "sfs.h"
#include <cmath>
#include <opencv/highgui.h>	//����ʹ��
#include <fstream>
#include "utility.h"
using namespace std;


CvMat* EstimateLightCoff(CvMat* Z,CvMat* I,CvMat* P)
{
	assert(I->cols == Z->cols && I->cols == P->cols);
	assert(I->rows == Z->rows && I->rows == P->rows);
	const int rows = I->rows;
	const int cols = I->cols;

	float SI = 0, SINx = 0, SINy = 0, SINz = 0;
	float SP = 0, SPNx = 0, SPNy = 0, SPNz = 0;
	float SPNxx = 0, SPNxy = 0, SPNxz = 0;
	float SPNyy = 0, SPNyz = 0, SPNzz = 0;

	for(int y = 1; y < rows-1; ++y){
		for(int x = 1; x < cols-1;++x){
			float d = cvmGet(Z,y,x);
			float dx = cvmGet(Z,y,x+1);
			float dy = cvmGet(Z,y+1,x);
			if(!IsNaN(d) && !IsNaN(dx) && !IsNaN(dy)){
				//I
				float i = cvmGet(I,y,x);
				//P
				float p = cvmGet(P,y,x);
				//Z
				float z  = cvmGet(Z,y,x);float zx = cvmGet(Z,y,x+1);
				float zy = cvmGet(Z,y+1,x);
				//n
				float dx = zx - z;float dy = zy - z;
				float r = sqrt(dx*dx + dy*dy + 1);
				float nx = dx/r;float ny = dy/r;float nz = -1/r;
				//sum
				SI += i;SINx += i*nx;SINy += i*ny;SINz += i*nz;
				SP += p;SPNx += p*nx;SPNy += p*ny;SPNz += p*nz;
				SPNxx += p*nx*nx;SPNxy += p*nx*ny;SPNxz += p*nx*nz;
				SPNyy += p*ny*ny;SPNyz += p*ny*nz;SPNzz += p*nz*nz;
			}
		}
	}

	float MA[] = 
	{
		SP,   SPNx,   SPNy,  SPNz,
		SPNx, SPNxx,  SPNxy, SPNxz,
		SPNy, SPNxy,  SPNyy, SPNyz,
		SPNz, SPNxz,  SPNyz, SPNzz,
	};
	float MB[] = 
	{
		SI,
		SINx,
		SINy,
		SINz,
	};
	CvMat* x = cvCreateFloatMat(4,1);
	CvMat A = cvMat(4,4,CV_32FC1,MA);
	CvMat B = cvMat(4,1,CV_32FC1,MB);
	cvSolve(&A,&B,x);

	CvMat* light =  cvCreateFloatMat(1,4);
	for(int i=0;i<4;++i)
	{
		float v = cvmGet(x,i,0);
		cvmSet(light,0,i,v);
	}
	cvReleaseMat(&x);
	return light;
}


/* ������ظ��� */
CvMat* _interpolateToGrid(const std::vector<float>& vx,
						 const std::vector<float>& vy,
						 const std::vector<float>& vz,
						 int rows,int cols)
{


	CvMat* src = cvCreateFloatMat(vx.size(),2);
	CvMat* des = cvCreateFloatMat(vx.size(),1);
	for(int i = 0;i<vx.size();++i){
		cvmSet(src,i,0,vx[i]);
		cvmSet(src,i,1,vy[i]);
		cvmSet(des,i,0,vz[i]);
	}

	Log("Compute tps coefficient");
	CvMat* coef = TpsTrain(src,des);
	CvMat* grid = cvCreateFloatMat(rows,cols);

	Log("Interpolating...");
	int o = 0;
	for(int y = 0;y<rows;++y){
		for(int x=0;x<cols;++x){
			float v = TpsPredict(x,y,src,coef);
			if(v<=ZLIMIT)
				v = NaN;
			cvmSet(grid,y,x,v);
			int p = (y*cols+x) * 100 / (cols*rows);
			if(p-o>=5)
			{
				cout<<p<<"%\t";
				o = p;
			}
		}
	}
	cout<<"100%"<<endl;

	cvReleaseMat(&src);
	cvReleaseMat(&des);
	cvReleaseMat(&coef);

	return grid;
}


void _mergePoints(const vector<float>& vx,const vector<float>& vy,const vector<float>& vz,
				  vector<float>& out_vx,vector<float>& out_vy,vector<float>& out_vz,
				  const int offsetX,const int offsetY,
				  const int rows,const int cols)
{
	out_vx.clear(); out_vy.clear(); out_vz.clear();	//����������

	const int d = D;	//��֮�����С�������

	//��ʼ����¼��
	char * const table = new char[ (cols+d) * rows];
	memset(table,0,(cols+d)*rows);

	const int sz = vx.size();
	for(int i = 0; i < sz; ++i){ //����ÿ����

		int x = vx[i] - offsetX;	//������
		int y = vy[i] - offsetY;	//������

		if( x<0 || x>=cols || y<0 || y>=rows)	//������
			continue;

		bool ok = true;	//�Ƿ��ظ�
		for(int dx = max(0,x-d); (dx <= x+d) && (dx < cols); ++dx){
			for(int dy = max(0,y-d); (dy <= y+d) && (dy <rows);  ++dy){
				int idx = dx*(cols+d) + dy;
				if(table[idx] != 0){
					ok = false;	//�ظ�
					break;
				}
			}
			if(!ok) break;
		}

		if(!ok) continue;

		int idx = x*(cols+d) + y;
		table[idx] = 1;
		out_vx.push_back(x);
		out_vy.push_back(y);
		out_vz.push_back(vz[i]);
	}

	delete[] table;
}


/* �ú�����Ҫ�޸ĳ������з�+˫������ֵ����ʱ��TPS���� */
CvMat* InterpolateToGrid(const std::vector<float>& vx,
						 const std::vector<float>& vy,
						 const std::vector<float>& vz,
						 int rows,int cols,
						 int offsetX ,int offsetY )
{
	int sz = vx.size();
	assert(sz == vy.size() && sz == vz.size());

	//ȥ���ظ����Լ��߽���ĵ�
	vector<float> newX,newY,newZ;
	_mergePoints(vx,vy,vz,newX,newY,newZ,offsetX,offsetY,rows,cols);

	//��ֵ
	return _interpolateToGrid(newX,newY,newZ,rows,cols);
}



bool inline _surrounded(CvMat* mat,int x,int y)
{
	if(x<1 || x>=mat->cols-1 || y<1 || y>=mat->rows-1) return false;
	for(int dx = x-1; dx <= x+1; ++dx)
	{
		for(int dy = y-1; dy <= y+1; ++dy){
			float v = cvmGet(mat,dy,dx);
			if(IsNaN(v)) return false;
		}
	}
	return true;
}

float inline _gaussCoff(float doubleR,float segma)
{
	const static float c = 1.0f / sqrt(2*PI);
	return (c/segma) * pow(E,-doubleR/(2*segma*segma));
}


/*��ͼ����� ͳ���ڵ��Լ��߽��*/
void _indexPoints(CvMat* Z,
				  map<int,int>& out_mapZToI,map<int,int>& out_mapIToZ,
				  vector<int>& out_innerPoints,vector<int>& out_boundPoints)
{
	const int rows = Z->rows;
	const int cols = Z->cols;
	out_mapZToI.clear();
	out_mapIToZ.clear();
	out_innerPoints.clear();out_innerPoints.reserve(20000);//Ԥ��2W��λ��
	out_boundPoints.clear();out_boundPoints.reserve(20000);//...

	int *table = new int[rows * cols];	//ͼ���±굽�����ŵ�ӳ��
	int *tableBound = new int[rows * cols];	//�߽��
	int di[] = {0,-1,1,cols,-cols,cols+1,cols-1,-cols+1,-cols-1};	//3��3���ӵĵ�

	//��ʼ��
	for(int i = 0;i < rows * cols; ++i)
	{
		table[i] = -1;
		tableBound[i] = -1;
	}

	int allCnt = 0;
	for(int i = 0;i < cols * rows; ++i)
	{
		if(_surrounded(Z,i%cols,i/cols))
		{
			out_innerPoints.push_back(i);	//ͼ���±�
			tableBound[i] = -2;				//��Ϊ�߽��
			for(int j = 0; j < sizeof(di)/sizeof(di[0]); ++j)
			{
				int idx = i + di[j];	//��ǰͼ���±�
				if(table[idx] == -1)	//��һ�������õ�?
				{
					table[idx] = allCnt++;	//����һ��z�±�
					if(tableBound[idx] == -1)
					{
						tableBound[idx] = table[idx];	//��ʱ���Ϊ�߽��
					}
				}
			}
		}
	}

	//tableת����map,�ӿ��������
	for(int i = 0;i < rows * cols; ++i)
	{
		if(table[i] != -1)
		{
			out_mapZToI[table[i]] = i;
			out_mapIToZ[i] = table[i];
		}
		if(tableBound[i] > -1)	//�Ƿ��Ǳ߽��?
		{
			out_boundPoints.push_back(i);
		}
	}
	delete[] table;
	delete[] tableBound;

}


/* ����ʽ */
class Poly
{
public:
	Poly(float value = 0)
	{
		this->Value = value;
	};
	float Value;	//Ŀ��ֵ
	map<int,float> Items;	//�������-ϵ��ӳ��
};

/* ���淽�� (Ԫ���±��1��ʼ)*/
void _saveData(const vector<Poly>& polys,const CvMat* I,
			   map<int,int>& mapZToI, map<int,int>& mapIToZ,
			   const vector<int>& inner_pts, const vector<int>& bound_pts)
{
	
	const int rows = I->rows;
	const int cols = I->cols;

	/* x*A = B */
	Log("Write file \"A\" \"B\"");
	ofstream oa("A");	//ϡ�����A(��;��;ֵ) ÿ1��Ϊһ�����Է���ϵ��
	ofstream ob("B");	//ֵ����B

	//����A B
	int col = 0;
	for(auto it = polys.begin() ; it != polys.end();++it)
	{
		auto p = *it;
		for(auto i = p.Items.begin(); i != p.Items.end();++i)
		{
			int row = i->first;
			float v = i->second;
			oa<<row + 1<<"\t"<<col + 1<<"\t"<<v<<endl;
		}
		col += 1;
		ob<<it->Value<<"\t";
	}

	/* ����Z-I��Ӧ��ϵ */
	Log("Write file \"MAP\"");
	ofstream omap("MAP");
	for(auto it = mapZToI.begin();it != mapZToI.end();++it)
	{
		omap<<it->first+1<< "\t" <<it->second<<endl;
	}

	/* �����߽��(��Ϊ�������±�) */
	Log("Write file \"BOUND\"");
	ofstream bound("BOUND");
	for(auto it = bound_pts.begin();it != bound_pts.end();++it)
	{
		bound<<mapIToZ[*it] + 1<<endl;
	}

	/* ����ͼ�� */
	auto image = cvCreateImage(cvSize(cols,rows),8,3);
	for(int i = 0; i < bound_pts.size(); ++i) //�߽���Ϊ��ɫ
	{
		int idx = bound_pts[i]; int x = idx % cols; int y = idx / cols;
		int t = y * image->widthStep + x * image->nChannels;
		image->imageData[t] = 0;image->imageData[t+1] = 0;image->imageData[t+2] = 255;
	}

	for(int i = 0;i < inner_pts.size(); ++i)	//�ڵ����
	{
		int idx = inner_pts[i];	int x = idx % cols;	int y = idx / cols;
		int t = y * image->widthStep + x * image->nChannels;
		image->imageData[t] = cvmGet(I,y,x);image->imageData[t+1] = cvmGet(I,y,x);image->imageData[t+2] = cvmGet(I,y,x);
	}
	Log("Write file \"I.bmp\"");	
	cvSaveImage("I.bmp",image);	//����
	cvReleaseImage(&image);
}

CvMat* EstimateZ(CvMat* Z,CvMat* I,CvMat* P,CvMat* L,float segma,float lambda)
{
	const int rows = Z->rows;
	const int cols = Z->cols;
	assert(rows == I->rows && cols == I->cols);
	assert(rows == P->rows && cols == P->cols);
	assert(L->rows == 1 && L->cols == 4);

	/*3*3 gaussģ��*/
#define G(x) _gaussCoff((x),segma)

	float GSum = 1;//G(0)+4*G(1)+4*G(2);
	float GaussTemplate[] = 
	{
		G(2)/GSum,G(1)/GSum,G(2)/GSum,
		G(1)/GSum,G(0)/GSum,G(1)/GSum,
		G(2)/GSum,G(1)/GSum,G(2)/GSum,
	};

#undef G(x)


	/* Ѱ�ұ߽���Լ��ڲ��� */
	vector<int> inner_pts;	//���ĵ㼯��
	vector<int> bound_pts;	//�߽�㼯��
	map<int,int> mapIToZ;	//ͼ���±굽z���
	map<int,int> mapZToI;	//z��ŵ�ͼ���±�
	_indexPoints(Z,mapZToI,mapIToZ,inner_pts,bound_pts);	//�Ե��� ͳ�Ʊ߽�������ĵ� ����I��Z�ı�Ź�ϵ

	/* �߽�ƽ���Բ���� */
	int top = 1000000,button = -1,left = 1000000,right = -1;
	for(int i = 0;i < bound_pts.size();++i)
	{
		int idx = bound_pts[i];
		int x = idx % cols,y = idx / cols;
		top = min(y,top);	button = max(y,button);
		left = min(left,x); right = max(x,right);
	}
	//��Բ����ϵ��
	int a = (right - left) / 2;
	int b = (button - top) / 2;
	int centerX = (left + right) / 2;
	int centerY = (top  + button) / 2;


	int innerCnt = inner_pts.size();	//�ڲ�������
	int boundCnt = bound_pts.size();	//�߽������
	int allCnt = innerCnt + boundCnt;
	int funcCnt  = 2*innerCnt + boundCnt + 1;//����������ÿ���ڲ������+gauss2�����̣��ⲿ��߽�Լ��1�����̣�z0=zref0һ�����̣�

	/* ���췽�� */
	vector<Poly> polys;
	polys.reserve(40000);
	/*1.���䷽��*/
	float l0 = cvmGet(L,0,0), l1 = cvmGet(L,0,1), l2 = cvmGet(L,0,2), l3 = cvmGet(L,0,3);	//����ϵ��
	
	for(auto it = inner_pts.begin();it != inner_pts.end();++it)
	{
		int idx = *it;
		int x = idx % cols, y = idx / cols;
		float z = cvmGet(Z,y,x);
		float p = cvmGet(P,y,x);
		float i = cvmGet(I,y,x);

		float zx = cvmGet(Z,y,x+1) - z;
		float zy = cvmGet(Z,y+1,x) - z;

		float n = sqrt(zx*zx + zy*zy + 1);
		float nx = zx/n;
		float ny = zy/n;
		float nz = -1/n;

		float s = p * (l0 + l1*nx + l2*ny + l3*nz);

		float v = i - nz*l3 - p*l0;
		Poly poly(v);


		float cof_zx = p*l1/n;
		float cof_zy = p*l2/n;
		float cof_z = -cof_zx - cof_zy;

		poly.Items[mapIToZ[y*cols+x]] = cof_z;
		poly.Items[mapIToZ[y*cols+x+1]] = cof_zx;
		poly.Items[mapIToZ[(y+1)*cols+x]] = cof_zy;

		polys.push_back(poly);
	}

	/*2.gauss����*/
	for(auto it = inner_pts.begin();it != inner_pts.end();++it)
	{
		int idx = *it;
		int x = idx % cols, y = idx / cols;

		Poly poly(0);

		float sum = 0;
		for(int dx = -1; dx <= 1;++dx)
		{
			for(int dy = -1;dy <= 1;++dy)
			{
				float d = (dx==0 && dy==0) ? 1:0;
				float g = (GaussTemplate[(dy+1)*3+dx+1] - d) * lambda;	//��ǰϵ��
				float zref = cvmGet(Z,y+dy,x+dx);	//�ο�z
				sum += zref * g;					//����ܺ�
				int z_i = (y+dy)*cols+(x+dx);		//Ŀ��z�±�
				poly.Items[mapIToZ[z_i]] = g;
			}
		}
		poly.Value = sum;
		polys.push_back(poly);
	}


	/*3.�߽�Լ��*/
	float _k = -b*b/(a*a);
	for(auto it = bound_pts.begin(); it != bound_pts.end(); ++it)
	{
		int idx = *it;
		int x = idx % cols;
		int y = idx / cols;

		//������
		float nx =  (x - centerX)*_k;
		float ny = -(y - centerY);
		float r = sqrt(nx*nx + ny*ny);
		nx /= r;
		ny /= r;

		//�ݶ�
		float dx = NaN, dy = NaN;
		float z = cvmGet(Z,y,x);
		{	
			//Ѱ�Ҹ�����
			if(x-1>=0 && IsNaN(dx))	//���
			{
				float _z = cvmGet(Z,y,x-1);	
				if(!IsNaN(_z))
					dx = -1;
			}
			if(x+1<cols && IsNaN(dx))	//�ұ�
			{
				float _z = cvmGet(Z,y,x+1);	
				if(!IsNaN(_z))
					dx = 1;
			}
			if(IsNaN(dy) && y-1>=0)
			{
				float _z = cvmGet(Z,y-1,x);	//�ϱ�
				if(!IsNaN(_z))
					dy = -1;
			};
			if(IsNaN(dy) && y+1<rows)
			{
				float _z = cvmGet(Z,y+1,x);	//�±�
				if(!IsNaN(_z))
					dy = 1;
			};
		};

		Poly poly(0);
		poly.Items[mapIToZ[y*cols+x+dx]] = dx*nx;
		poly.Items[mapIToZ[(y+dy)*cols+x]] = dy*ny;
		poly.Items[mapIToZ[y*cols+x]] = -dx*nx - dy*ny;
		polys.push_back(poly);

	}


	/*4.z0=zref0*/
	{
		int idx = inner_pts[inner_pts.size()/2];	//ѡȡ�е�
		int x = idx % cols;
		int y = idx / cols;
		float zref = cvmGet(Z,y,x);
		Poly poly(zref);
		poly.Items[mapIToZ[idx]] = 1.0f;
	}

	/* ��C++ʵ�����֮ǰ����ʱ��������matlab��� */
	Log("Write the sfs model");
	_saveData(polys,I,mapZToI,mapIToZ,inner_pts,bound_pts);

	return 0;
}

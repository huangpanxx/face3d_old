#include "stasm.hpp"

#include "asm.h"
#include <iostream>
#include <string>
#include <vector>
using namespace std;


bool IsLoggerEnabled = true; 

void SetLogging(bool isEnable)
{
	IsLoggerEnabled = isEnable;
}

void InitModels(const char conf1[],const char conf2[],const char data_dir[])
{
	InitAsmModels(conf1,conf2,data_dir);
}

int AsmSearch(
	char imagePath[],
	double points[][2])
{
	RgbImage image(imagePath);
	SHAPE startShape;
	DET_PARAMS detParams;
	double meanTime;
	SHAPE shape = AsmSearch(
		startShape,detParams,meanTime,
		image,imagePath,
		false,NULL,false
		);
	int nRows = shape.nrows();
	for(int i = 0; i < nRows; ++i){
		points[i][0] = image.width/2   + shape(i,0);
		points[i][1] = image.height/2 - shape(i,1);
	};
	return nRows;
}

#ifndef ASM_H
#define ASM_H
/*
	该文件作为asm.lib配套头文件
	不应该直接引用该文件，而应使用deform.h中包装后的函数
*/


/*是否允许输出日志*/
void SetLogging(bool isEnable);

/*初始化模型*/
void InitModels(const char conf1[],const char conf2[],const char data_dir[]);

/*定位特征点*/
int AsmSearch ( char imagePath[], double points[][2]);


#endif
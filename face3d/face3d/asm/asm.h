#ifndef ASM_H
#define ASM_H
/*
	���ļ���Ϊasm.lib����ͷ�ļ�
	��Ӧ��ֱ�����ø��ļ�����Ӧʹ��deform.h�а�װ��ĺ���
*/


/*�Ƿ����������־*/
void SetLogging(bool isEnable);

/*��ʼ��ģ��*/
void InitModels(const char conf1[],const char conf2[],const char data_dir[]);

/*��λ������*/
int AsmSearch ( char imagePath[], double points[][2]);


#endif
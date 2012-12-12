#ifndef ASM_H
#define ASM_H


void SetLogging(bool isEnable);

void InitModels(const char conf1[],const char conf2[],const char data_dir[]);

int AsmSearch ( char imagePath[], double points[][2]);


#endif
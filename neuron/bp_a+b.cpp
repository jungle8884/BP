// neuron.cpp : �������̨Ӧ�ó������ڵ㡣
//

//#include "stdafx.h"
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>

#define Data  1000  	// ѵ������������
#define In 2			// ����ÿ�������ж��ٸ�������� 
#define Out 1			// ����ÿ�������ж��ٸ��������
#define Neuron 5		// ��Ԫ������ 
#define TrainC 20000 	// ��ʾѵ���Ĵ��� 
#define A  0.2			
#define B  0.4
#define a  0.2
#define b  0.3

// d_in[Data][In] �洢 Data ��������ÿ�������� In ������
// d_out[Data][Out] �洢 Data ��������ÿ�������� Out �����
double d_in[Data][In],d_out[Data][Out]; 
	
// w[Neuron][In]  ��ʾĳ�������ĳ����Ԫ��Ȩ�� 
// v[Out][Neuron] ����ʾĳ����Ԫ��ĳ�������Ȩ�� 
// ���� o[Neuron] ��¼������Ԫͨ��������������� 
// ��֮��Ӧ�ı����������������������� dw[Neuron][In] �� dv[Out][Neuron] 
double w[Neuron][In],v[Out][Neuron],o[Neuron];
double dv[Out][Neuron],dw[Neuron][In];

// Data�������� ���������������� �����,��Сֵ 
double Maxin[In],Minin[In],Maxout[Out],Minout[Out];

// OutputData[Out]  �洢BP���������� 
double OutputData[Out];

// e���������� 
double e;

//����ʵ�����ݲ�������Ӧ�ļ���
void writeTest(){      
	FILE *fp1,*fp2;
	double r1,r2;
	int i;

	if((fp1=fopen("E:\\neuron\\in.txt","w"))==NULL){
		printf("can not open the in file\n");
		exit(0);
	}
	if((fp2=fopen("E:\\neuron\\out.txt","w"))==NULL){
		printf("can not open the out file\n");
		exit(0);
	}

	for(i=0;i<Data;i++){
		// ����0~10�����С�� 
		r1=rand()%1000/100.0;
		r2=rand()%1000/100.0; 
		// д���ļ� 
		fprintf(fp1,"%lf  %lf\n",r1,r2);
		fprintf(fp2,"%lf \n",r1+r2);
	}
	fclose(fp1);
	fclose(fp2);
}

// ����ѵ������ 
void readData(){

	FILE *fp1,*fp2;
	int i,j;
	if((fp1=fopen("E:\\neuron\\in.txt","r"))==NULL){
		printf("can not open the in file\n");
		exit(0);
	}
	// �������ݵ� d_in[Data][In] 
	for(i=0;i<Data;i++)
		for(j=0; j<In; j++)
			fscanf(fp1,"%lf",&d_in[i][j]);
	fclose(fp1);
	
	if((fp2=fopen("E:\\neuron\\out.txt","r"))==NULL){
		printf("can not open the out file\n");
		exit(0);
	}
	// �������ݵ� d_in[Data][Out] 
	for(i=0;i<Data;i++)
		for(j=0; j<Out; j++)
			fscanf(fp1,"%lf",&d_out[i][j]);
	fclose(fp2);
}

/*
һ�����ǶԶ�ȡ��ѵ���������ݽ��й�һ������
��һ���������ָ�ľ��ǽ�����ת����0~1֮��; 
��BP�������������棬��û�ж��������Ҫ��
����ʵ��ʵ�������У���һ�������ǲ��ɻ�ȱ�ġ�
��Ϊ����ģ��û���ǵ���BP�������������������⣬
һ����˵��Ԫ���������0~1֮������ݷǳ����У���һ���ܹ��������ѵ��Ч�ʡ�
���������¹�ʽ��������й�һ����
���� �Ӹ�����A ��Ϊ�˷�ֹ���� 0 �������0����Ϊ��ĸ����
��һ���棬���Ƕ���Ԫ��Ȩ�ؽ��г�ʼ���ˣ�
���ݹ�һ���ˣ�0~1��֮�䣬
��ôȨ�س�ʼ��Ϊ��-1~1��֮������ݣ�
�������������ֵΪ0 
*/ 
void initBPNework(){

	int i,j;

	for(i=0; i<In; i++){   //��Data�������� �����ͱ������������Сֵ��
		Minin[i]=Maxin[i]=d_in[0][i];
		for(j=0; j<Data; j++)
		{
			Maxin[i]=Maxin[i]>d_in[j][i]?Maxin[i]:d_in[j][i];
			Minin[i]=Minin[i]<d_in[j][i]?Minin[i]:d_in[j][i];
		}
	}

	for(i=0; i<Out; i++){     //��Data�������к͵������Сֵ��
		Minout[i]=Maxout[i]=d_out[0][i];
		for(j=0; j<Data; j++)
		{
			Maxout[i]=Maxout[i]>d_out[j][i]?Maxout[i]:d_out[j][i];
			Minout[i]=Minout[i]<d_out[j][i]?Minout[i]:d_out[j][i];
		}
	}
	
	//�������ݹ�һ��
	for (i = 0; i < In; i++)
		for(j = 0; j < Data; j++)
			d_in[j][i]=(d_in[j][i]-Minin[i]+1)/(Maxin[i]-Minin[i]+1);
			
	//������ݹ�һ��
	for (i = 0; i < Out; i++)    
		for(j = 0; j < Data; j++)
			d_out[j][i]=(d_out[j][i]-Minout[i]+1)/(Maxout[i]-Minout[i]+1);
	
	//��ʼ����Ԫ
	for (i = 0; i < Neuron; ++i)	
		for (j = 0; j < In; ++j){	
			// rand()����Ҫ���������᷵��һ����0�������������������� 
			// rand()/RAND_MAX Ϊ (0, 1) 
			w[i][j]=rand()*2.0/RAND_MAX-1; // Ȩֵ��ʼ�� 
			dw[i][j]=0;
		}

		for (i = 0; i < Neuron; ++i)	
			for (j = 0; j < Out; ++j){
				v[j][i]=rand()*2.0/RAND_MAX-1; // Ȩֵ��ʼ�� 
				dv[j][i]=0;
			}
}

void computO(int var){   //��var�����������ز��������������o[]��outputdata[]��

	int i,j;
	double sum,y;
	// ��Ԫ��� 
	for (i = 0; i < Neuron; ++i){
		sum=0;
		for (j = 0; j < In; ++j)
			sum+=w[i][j]*d_in[var][j];
		//Sigmoid ����---����� 
		o[i]=1/(1+exp(-1*sum));
	}

	/*  ���ز㵽�������� */
	for (i = 0; i < Out; ++i){
		sum=0;
		for (j = 0; j < Neuron; ++j)
			sum+=v[i][j]*o[j];
		OutputData[i]=sum;
	}	
}

//�Ӻ���ǰ����Ȩֵ��
void backUpdate(int var)
{
	int i,j;
	double t;
	for (i = 0; i < Neuron; ++i)
	{
		t=0;
		for (j = 0; j < Out; ++j){
			t+=(OutputData[j]-d_out[var][j])*v[j][i];
			
			/*
			 �ھ���ʵ�ֶ�����޸��У������ټ���ѧϰ�ʣ�
			 ���Ҷ���ǰѧϰ����������������м̳У�
			 ֱ�׵�˵���Ƕ�����һ��0��1֮�����
			*/ 
			dv[j][i]=A*dv[j][i]+B*(OutputData[j]-d_out[var][j])*o[i];
			v[j][i]-=dv[j][i];
		}

		for (j = 0; j < In; ++j){
			dw[i][j]=a*dw[i][j]+b*t*o[i]*(1-o[i])*d_in[var][j];
			w[i][j]-=dw[i][j];
		}
	}
}

double result(double var1,double var2)
{
	int i,j;
	double sum,y;

	var1=(var1-Minin[0]+1)/(Maxin[0]-Minin[0]+1);
	var2=(var2-Minin[1]+1)/(Maxin[1]-Minin[1]+1);

	for (i = 0; i < Neuron; ++i){
		sum=0;
		sum=w[i][0]*var1+w[i][1]*var2;
		o[i]=1/(1+exp(-1*sum));
	}
	sum=0;
	for (j = 0; j < Neuron; ++j)
		sum+=v[0][j]*o[j];

	return sum*(Maxout[0]-Minout[0]+1)+Minout[0]-1;  //����һ��
}

void writeNeuron()
{
	FILE *fp1;
	int i,j;
	if((fp1=fopen("E:\\neuron\\neuron.txt","w"))==NULL)
	{
		printf("can not open the neuron file\n");
		exit(0);
	}
	for (i = 0; i < Neuron; ++i)	
		for (j = 0; j < In; ++j){
			fprintf(fp1,"%lf ",w[i][j]);
		}
	fprintf(fp1,"\n\n\n\n");

	for (i = 0; i < Neuron; ++i)	
		for (j = 0; j < Out; ++j){
			fprintf(fp1,"%lf ",v[j][i]);
		}

	fclose(fp1);
}


/*��������a��b��10���ڵ�������һ����� c��c=a+b��
���仰˵���ǽ�BP������ӷ����� */
void  trainNetwork(){

	int i,c=0,j;
	do{
		e=0;
		for (i = 0; i < Data; ++i){
			computO(i);//�������ز�������������Ԫ�������
			
			for (j = 0; j < Out; ++j)
				e+=fabs((OutputData[j]-d_out[i][j])/d_out[i][j]);//fabs������float��double�����ֵ 
				
			backUpdate(i);//�����޸�Ȩֵ
		}
		printf("%d  %lf\n",c,e/Data);
		c++;
	}while(c<TrainC && e/Data>0.01);//һֱѵ�����涨������ƽ�����С��0.01ʱ������
}

//int _tmain(int argc, _TCHAR* argv[])
int main(int argc, char* argv[])
{
	writeTest();//�������Data�����ݣ�ÿ�����ݰ���������λ���Լ���������֮�͡�
	readData();//׼�����룬���ѵ�����ݡ�
	initBPNework();//���롢������ݹ�һ�����Լ�����Ȩֵ��ʼ����
	trainNetwork();//ѵ�����硣
	printf("%lf \n",result(6,8) );
	printf("%lf \n",result(2.1,7) );
	printf("%lf \n",result(4.3,8) );
	writeNeuron();//����Ȩֵ��
	getchar();
	return 0;

}


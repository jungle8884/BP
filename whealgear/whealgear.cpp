#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>

#define Data  100  	// ѵ������������
#define TestData 10		// �������������� 
#define In 8			// ����ÿ�������ж��ٸ�������� 
#define Out 2			// ����ÿ�������ж��ٸ��������
#define Neuron 6		// ��Ԫ������ 
#define TrainC 1000 	// ��ʾѵ���Ĵ��� 
#define A  0.2			
#define B  0.4
#define a  0.2
#define b  0.3

// d_in[Data][In] �洢 Data ��������ÿ�������� In ������
// d_out[Data][Out] �洢 Data ��������ÿ�������� Out �����
double d_in[Data][In],d_out[Data][Out]; 

// d_test[TestData][In] �洢 TestData ��������ÿ�������� In ������
double d_test[TestData][In]; 
	
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

// ����ѵ������ 
void readData(){

	FILE *fp1,*fp2;
	int i,j;
	if((fp1=fopen("E:\\whealgear\\in.txt","r"))==NULL){
		printf("can not open the in file\n");
		exit(0);
	}
	// �������ݵ� d_in[Data][In] 
	for(i=0;i<Data;i++)
		for(j=0; j<In; j++)
			fscanf(fp1,"%lf",&d_in[i][j]);
	fclose(fp1);
	
	if((fp2=fopen("E:\\whealgear\\out.txt","r"))==NULL){
		printf("can not open the out file\n");
		exit(0);
	}
	// �������ݵ� d_in[Data][Out] 
	for(i=0;i<Data;i++)
		for(j=0; j<Out; j++)
			fscanf(fp1,"%lf",&d_out[i][j]);
	fclose(fp2);
}


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

void result(double var[In])
{
	int i,j,k;
	double sum;
	double y[Out];
	
	for(i = 0; i < In; ++i){
		var[i]=(var[i]-Minin[i]+1)/(Maxin[i]-Minin[i]+1);
	}

	for (i = 0; i < Neuron; ++i){
		sum=0;
		for(k = 0; k < In; ++k){
			sum += w[i][k] * var[k];
		}
		o[i]=1/(1+exp(-1*sum));
	}
	
	for (k = 0; k < Out; ++k){
		sum=0;
		for (j = 0; j < Neuron; ++j){
			sum += v[k][j] * o[j];
		}
		y[k] = sum;
	}
	
	//����һ��
	for (k = 0; k < Out; ++k){
		y[k] = y[k] * (Maxout[0]-Minout[0]+1)+Minout[0]-1;
	}
	printf("%lf %lf\n", y[0], y[1]);
	
	return;  
}

void writeNeuron()
{
	FILE *fp1;
	int i,j;
	if((fp1=fopen("E:\\whealgear\\whealgear.txt","w"))==NULL)
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
		//printf("%d  %lf\n",c, e/Data);
		c++;
	}while(c<TrainC && e/Data>0.001);//һֱѵ�����涨������ƽ�����С��0.001ʱ������
}

void testNetwork()
{
	FILE *fp1,*fp2;
	int i,j;
	double test[In]; 
	if((fp1=fopen("E:\\whealgear\\test.txt","r"))==NULL){
		printf("can not open the in file\n");
		exit(0);
	}
	// �������ݵ� d_test[Data][In] 
	for(i=0;i<TestData;i++)
		for(j=0; j<In; j++)
			fscanf(fp1,"%lf",&d_test[i][j]);
	fclose(fp1);
	
	for(i=0;i<TestData;i++){
		for(j=0; j<In; j++){
			test[j] = d_test[i][j];
		}
		result(test);	
	}
} 

int main(int argc, char* argv[])
{
	readData();//׼�����룬���ѵ�����ݡ�
	initBPNework();//���롢������ݹ�һ�����Լ�����Ȩֵ��ʼ����
	trainNetwork();//ѵ�����硣
	testNetwork();//�������� 
	writeNeuron();//����Ȩֵ��
	
	getchar();
	return 0;

}

							  

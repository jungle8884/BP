// neuron.cpp : 定义控制台应用程序的入口点。
//

//#include "stdafx.h"
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>

#define Data  1000  	// 训练样本的数量
#define In 2			// 对于每个样本有多少个输入变量 
#define Out 1			// 对于每个样本有多少个输出变量
#define Neuron 5		// 神经元的数量 
#define TrainC 20000 	// 表示训练的次数 
#define A  0.2			
#define B  0.4
#define a  0.2
#define b  0.3

// d_in[Data][In] 存储 Data 个样本，每个样本的 In 个输入
// d_out[Data][Out] 存储 Data 个样本，每个样本的 Out 个输出
double d_in[Data][In],d_out[Data][Out]; 
	
// w[Neuron][In]  表示某个输入对某个神经元的权重 
// v[Out][Neuron] 来表示某个神经元对某个输出的权重 
// 数组 o[Neuron] 记录的是神经元通过激活函数对外的输出 
// 与之对应的保存它们两个修正量的数组 dw[Neuron][In] 和 dv[Out][Neuron] 
double w[Neuron][In],v[Out][Neuron],o[Neuron];
double dv[Out][Neuron],dw[Neuron][In];

// Data个数据中 加数、被加数、和 的最大,最小值 
double Maxin[In],Minin[In],Maxout[Out],Minout[Out];

// OutputData[Out]  存储BP神经网络的输出 
double OutputData[Out];

// e用来监控误差 
double e;

//生成实验数据并存入相应文件中
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
		// 生成0~10的随机小数 
		r1=rand()%1000/100.0;
		r2=rand()%1000/100.0; 
		// 写入文件 
		fprintf(fp1,"%lf  %lf\n",r1,r2);
		fprintf(fp2,"%lf \n",r1+r2);
	}
	fclose(fp1);
	fclose(fp2);
}

// 读入训练数据 
void readData(){

	FILE *fp1,*fp2;
	int i,j;
	if((fp1=fopen("E:\\neuron\\in.txt","r"))==NULL){
		printf("can not open the in file\n");
		exit(0);
	}
	// 读入数据到 d_in[Data][In] 
	for(i=0;i<Data;i++)
		for(j=0; j<In; j++)
			fscanf(fp1,"%lf",&d_in[i][j]);
	fclose(fp1);
	
	if((fp2=fopen("E:\\neuron\\out.txt","r"))==NULL){
		printf("can not open the out file\n");
		exit(0);
	}
	// 读入数据到 d_in[Data][Out] 
	for(i=0;i<Data;i++)
		for(j=0; j<Out; j++)
			fscanf(fp1,"%lf",&d_out[i][j]);
	fclose(fp2);
}

/*
一方面是对读取的训练样本数据进行归一化处理，
归一化处理就是指的就是将数据转换成0~1之间; 
在BP神经网络理论里面，并没有对这个进行要求，
不过实际实践过程中，归一化处理是不可或缺的。
因为理论模型没考虑到，BP神经网络收敛的速率问题，
一般来说神经元的输出对于0~1之间的数据非常敏感，归一化能够显著提高训练效率。
可以用以下公式来对其进行归一化，
其中 加个常数A 是为了防止出现 0 的情况（0不能为分母）。
另一方面，就是对神经元的权重进行初始化了，
数据归一到了（0~1）之间，
那么权重初始化为（-1~1）之间的数据，
另外对修正量赋值为0 
*/ 
void initBPNework(){

	int i,j;

	for(i=0; i<In; i++){   //求Data个数据中 加数和被加数的最大、最小值。
		Minin[i]=Maxin[i]=d_in[0][i];
		for(j=0; j<Data; j++)
		{
			Maxin[i]=Maxin[i]>d_in[j][i]?Maxin[i]:d_in[j][i];
			Minin[i]=Minin[i]<d_in[j][i]?Minin[i]:d_in[j][i];
		}
	}

	for(i=0; i<Out; i++){     //求Data个数据中和的最大、最小值。
		Minout[i]=Maxout[i]=d_out[0][i];
		for(j=0; j<Data; j++)
		{
			Maxout[i]=Maxout[i]>d_out[j][i]?Maxout[i]:d_out[j][i];
			Minout[i]=Minout[i]<d_out[j][i]?Minout[i]:d_out[j][i];
		}
	}
	
	//输入数据归一化
	for (i = 0; i < In; i++)
		for(j = 0; j < Data; j++)
			d_in[j][i]=(d_in[j][i]-Minin[i]+1)/(Maxin[i]-Minin[i]+1);
			
	//输出数据归一化
	for (i = 0; i < Out; i++)    
		for(j = 0; j < Data; j++)
			d_out[j][i]=(d_out[j][i]-Minout[i]+1)/(Maxout[i]-Minout[i]+1);
	
	//初始化神经元
	for (i = 0; i < Neuron; ++i)	
		for (j = 0; j < In; ++j){	
			// rand()不需要参数，它会返回一个从0到最大随机数的任意整数 
			// rand()/RAND_MAX 为 (0, 1) 
			w[i][j]=rand()*2.0/RAND_MAX-1; // 权值初始化 
			dw[i][j]=0;
		}

		for (i = 0; i < Neuron; ++i)	
			for (j = 0; j < Out; ++j){
				v[j][i]=rand()*2.0/RAND_MAX-1; // 权值初始化 
				dv[j][i]=0;
			}
}

void computO(int var){   //第var组数据在隐藏层和输出层的输出结果o[]和outputdata[]。

	int i,j;
	double sum,y;
	// 神经元输出 
	for (i = 0; i < Neuron; ++i){
		sum=0;
		for (j = 0; j < In; ++j)
			sum+=w[i][j]*d_in[var][j];
		//Sigmoid 函数---激活函数 
		o[i]=1/(1+exp(-1*sum));
	}

	/*  隐藏层到输出层输出 */
	for (i = 0; i < Out; ++i){
		sum=0;
		for (j = 0; j < Neuron; ++j)
			sum+=v[i][j]*o[j];
		OutputData[i]=sum;
	}	
}

//从后向前更新权值；
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
			 在具体实现对误差修改中，我们再加上学习率，
			 并且对先前学习到的修正误差量进行继承，
			 直白的说就是都乘上一个0到1之间的数
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

	return sum*(Maxout[0]-Minout[0]+1)+Minout[0]-1;  //返归一化
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


/*两个输入a、b（10以内的数），一个输出 c，c=a+b。
换句话说就是教BP神经网络加法运算 */
void  trainNetwork(){

	int i,c=0,j;
	do{
		e=0;
		for (i = 0; i < Data; ++i){
			computO(i);//计算隐藏层和输出层所有神经元的输出。
			
			for (j = 0; j < Out; ++j)
				e+=fabs((OutputData[j]-d_out[i][j])/d_out[i][j]);//fabs（）对float，double求绝对值 
				
			backUpdate(i);//反向修改权值
		}
		printf("%d  %lf\n",c,e/Data);
		c++;
	}while(c<TrainC && e/Data>0.01);//一直训练到规定次数或平均误差小于0.01时结束。
}

//int _tmain(int argc, _TCHAR* argv[])
int main(int argc, char* argv[])
{
	writeTest();//随机生成Data个数据，每个数据包括两个个位数以及这两个数之和。
	readData();//准备输入，输出训练数据。
	initBPNework();//输入、输出数据归一化，以及网络权值初始化。
	trainNetwork();//训练网络。
	printf("%lf \n",result(6,8) );
	printf("%lf \n",result(2.1,7) );
	printf("%lf \n",result(4.3,8) );
	writeNeuron();//保存权值。
	getchar();
	return 0;

}


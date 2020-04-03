//This is an implementation of two layered fully connected neural network.
//Author: Ajay Singh Pawar
#include<iostream>
#include<time.h>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/opencv.hpp>
#include<dirent.h>
#include<unistd.h>
#include<cmath>
#include<cstdlib>
#include<string>
#include <boost/lexical_cast.hpp>// for lexical_cast() 
#include<sstream>
#include<random>
#define batch_size 100
using namespace std;
using namespace cv;
#define Hidden_layer_1 200
#define Hidden_layer_2 200

long double sigmoid(long double a)
{
	return 1/(1+exp(-a));
}

int read_image( long double *a, int *num, int count, int *y)
{
 Mat input;
 int c, r;
 int temp;
 string path;
	
	path="train-images/"+to_string(count)+".jpg";
	input=imread(path, CV_LOAD_IMAGE_GRAYSCALE);
	c=input.cols;
	r=input.rows;
	for(int i=0; i<10; i++)
		if(i==num[count])
			y[i]=1;
		else
			y[i]=0;
	
	for(int i=0, k=0; i<r; i++)
	{
		for(int j=0; j<c; j++,k++)
		{
			a[k]=input.at<uchar>(i,j);
			a[k]/=255;
		}
	}
	
return 0;
}
int maximum(long double *g)
{
	int label=0;
	long double max=g[0];
	for(int i=1; i<10; i++)
	{
		if(g[i]>max)
		{
			label=i;
			max=g[i];
		}
	}
return label;
}
int random_num(int low, int high)
{
 int ran;
	
	ran=rand()%((high-low)+1)+low;
return floor(ran);
}
void Predict_accuracy(long double w1[Hidden_layer_1][784], long double w2[Hidden_layer_2][Hidden_layer_1], long double w3[10][Hidden_layer_2], long double b1[Hidden_layer_1], long double b2[Hidden_layer_2], long double b3[10], int lab[10000])
{
 Mat input;
 int count=0;
 int c, r,correct=0;
 int y[10], label;
 string path;
 long double x[784], z1[Hidden_layer_1], g1[Hidden_layer_1], z2[Hidden_layer_2], z3[10], g2[Hidden_layer_2], g3[10], sum=0, accuracy=0;

	
	for(int i=0; i<10000; i++)
	{
		path="test-images/"+to_string(i)+".jpg";
		input=imread(path, CV_LOAD_IMAGE_GRAYSCALE);
		c=input.cols;
		r=input.rows;
		sum=0;
		for(int i=0, k=0; i<r; i++)
		{
			for(int j=0; j<c; j++,k++)
			{
				x[k]=input.at<uchar>(i,j);
				x[k]/=255;
			}
		}
		for(int j=0; j<Hidden_layer_1; j++)
		{	
			for(int i=0; i<784; i++)
			{
				z1[j] += w1[j][i]*x[i];
			}
			z1[j]+=b1[j];
			if(z1[j]<0)
				g1[j]=0;
			else
				g1[j]=z1[j];
		}
		for(int i=0; i<Hidden_layer_2; i++)
		{
			for(int j=0; j<Hidden_layer_1; j++)
			{
				z2[i]+=w2[i][j]*g1[j];
			}
			z2[i]+=b2[i];
			if(z2[i]<0)
				g2[i]=0;
			else
				g2[i]=z2[i];
		}
	
		for(int i=0; i<10; i++)
		{	
			for(int j=0; j< Hidden_layer_2; j++)
			{
				z3[i]+= w3[i][j]*g2[j];
			}
			z3[i]+=b3[i];
			g3[i]=exp(z3[i]);
			sum+=g3[i];
		}
		for(int i=0; i<10; i++)
		{
			g3[i] /= sum;
		}
	 	label=maximum(g3);

		if(lab[i]==label)
		{
			correct++;
		}
		for(int i=0; i<Hidden_layer_1; i++)
		{
			z1[i]=0;
			g1[i]=0;
		}
		for(int i=0; i<Hidden_layer_2; i++)
		{
			z2[i]=0;
			g2[i]=0;
		}
		for(int i=0; i<10; i++)
		{
			z3[i]=0;
			g3[i]=0;
		}
		
	}
	accuracy=(long double)correct/10000;
	cout<<"The accuracy obtained is:- "<<accuracy<<endl;
return;
}

int main(int argc, char *argv[])
{
 long double dz1[Hidden_layer_1], dz2[Hidden_layer_2], x[784], w1[Hidden_layer_1][784], b1[Hidden_layer_1], activation_b1[Hidden_layer_1], z1[Hidden_layer_1], g1[Hidden_layer_1], activation_w1[Hidden_layer_1][784];
 long double w2[Hidden_layer_2][Hidden_layer_1], activation_w2[Hidden_layer_2][Hidden_layer_1], b2[Hidden_layer_2], activation_b2[Hidden_layer_2], z2[Hidden_layer_2], g2[Hidden_layer_2];
long double  activation_w3[10][Hidden_layer_2],  w3[10][Hidden_layer_2], b3[10], activation_b3[10],  z3[10], g3[10] , Loss=0, output[10], sum=0, alpha=0.001;
 Mat input;
 int pred[10];
 unsigned short int xi[3];
 FILE *fp, *fq; 
 int count=0;
 int temp=1000 ,r ;
 int num[60000], y[10], k=0,l=0, pos[60000], lab[10000];
 std:: random_device rd;
 std::mt19937 gen(rd());
 xi[0]=1;
 xi[1]=1;
 xi[2]=1;
 std::normal_distribution<long double> d(0.5, 1/sqrt(784));

	for(int i=0; i<Hidden_layer_1; i++)
	{
		for(int j=0; j<784; j++)
		{
			//w1[i][j]=erand48(xi)*0.001;
			w1[i][j]=d(gen)*0.001;
			activation_w1[i][j]=0;
		}
		activation_b1[i]=0;
		b1[i]=0;
		dz1[i]=0;
	}
	for(int i=0; i<Hidden_layer_2; i++)
	{
		for(int j=0; j<Hidden_layer_1; j++)
		{
			w2[i][j]=d(gen)*0.001;
			activation_w2[i][j]=0;
		
		}
		b2[i]=0;
		activation_b2[i]=0;
		dz2[i]=0;
	}
	for(int i=0; i<10; i++)
	{
		for(int j=0; j<Hidden_layer_2; j++)
		{
			w3[i][j]=d(gen)*0.001;
			activation_w3[i][j]=0;
		}
		b3[i]=0;
		activation_b3[i]=0;
	}
		


fp=fopen("label.txt","r");
fq=fopen("test_label.txt","r");
for(int i=0; i<60000; i++)
{
	fscanf(fp, "%d", num+i);
	if(i<10000)
		fscanf(fq, "%d", lab+i);
}
fclose(fp);
fclose(fq);

	for(int o=0; o<60000; o++)
	{
		r=random_num(o, 59999);
		if(!((r>=0)&&(r<60000)))
			cout<<"Something is fishy with random_num"<<endl;
		pos[o]=r;
		
	}

for(int h=0; h<50; h++)
{
	for(int s=0,k=0; s<(60000/batch_size); s++)
	{	
		
		for(int p=0; p<batch_size; p++, k++)
		{
		//	if(p%500==0)
		//		cout<<p<<endl; 
	
			for(int i=0; i<10; i++)
				{
					g3[i]=0;
					z3[i]=0;
				}		
			for(int i=0; i<Hidden_layer_1; i++)
			{
				z1[i]=0;
				g1[i]=0;
				dz1[i]=0;
			}
			for(int i=0; i<Hidden_layer_2; i++)
			{
				z2[i]=0;
				g2[i]=0;
				dz2[i]=0;
			}
	
		
			if(read_image(x, num, pos[k], y)!=0)
			{
				cout<<"Read_image function returned an error:"<<endl;
			}
			sum=0;
		//////////////////////////Forward Prop Starts
			for(int j=0; j<Hidden_layer_1; j++)
			{
				for(int i=0; i<784; i++)
				{
					z1[j] += w1[j][i]*x[i];
				}
				z1[j]+=b1[j];
				if(z1[j]>=0)
					g1[j]=z1[j];
				else
					g1[j]=0;	
			}
			for(int i=0; i<Hidden_layer_2; i++)
			{	
				for(int j=0; j< Hidden_layer_1; j++)
				{
					z2[i]+= w2[i][j]*g1[j];
				}
				z2[i]+=b2[i];
				if(z2[i]>=0)
					g2[i]=z2[i];
				else
					g2[i]=0;
			}
     
			for(int i=0; i<10; i++)
			{
				for(int j=0; j<Hidden_layer_2; j++)
				{
					z3[i]+=w3[i][j]*g2[j];
				}
				z3[i]+=b3[i];
				g3[i]=exp(z3[i]);
				sum+=g3[i];
			}
			for(int i=0; i<10; i++)
			{
				g3[i] /= sum;
			}	
			for(int i=0; i<10; i++)
			{
				if(g3[i]==0)
				{
					cout<<"Log error:"<<endl;
					continue;	
				}
				else
					Loss+= -(y[i]*log(g3[i]));
			}
			for(int i=0; i<10; i++)
			{
				for(int j=0; j<Hidden_layer_2; j++)
				{
					activation_w3[i][j]+= ((g3[i]-y[i])*g2[j]);
				}
				activation_b2[i] += (g3[i]-y[i]);
			}
			
			for(int i=0; i<Hidden_layer_2; i++)
			{
				for(int j=0; j<10; j++)
				{
					dz2[i]+=w3[j][i]*(g3[j]-y[j]);
				}
				if(z2[i]<0)
					dz2[i]=0;
			}
			for(int j=0; j<Hidden_layer_2; j++)
			{
				for(int l=0; l<Hidden_layer_1; l++)
				{
					activation_w2[j][l]+= dz2[j]*g1[l];
				}
				activation_b2[j]+=dz2[j];
			}
		
			for(int i=0; i<Hidden_layer_1; i++)
			{
				for(int j=0; j< Hidden_layer_2; j++)
				{
					dz1[i]+=dz2[j]*w2[j][i];
				}
				if(z1[i]<0)
					dz1[i]=0;
			}
		
			for(int i=0; i<Hidden_layer_1; i++)
			{
				for(int j=0; j<784; j++)
				{
					activation_w1[i][j]+= dz1[i]*x[j];
				}
				activation_b1[i]+=dz1[i];
			}
	
	}
		for(int i=0; i<10; i++)
		{
			for(int j=0; j<Hidden_layer_2; j++)
			{
				w3[i][j]-=alpha*activation_w3[i][j];
				activation_w3[i][j]=0;	
			}
			b3[i] -= alpha*activation_b3[i];
			activation_b3[i]=0;
		}

		for(int i=0; i<Hidden_layer_2; i++)
		{
			for(int j=0; j<Hidden_layer_1; j++)
			{
				w2[i][j]-=alpha*activation_w2[i][j];
				activation_w2[i][j]=0;	
			}
			b2[i] -= alpha*activation_b2[i];
			activation_b2[i]=0;
		}
	
		for(int j=0; j<Hidden_layer_1; j++)
		{
			for(int l=0; l<784; l++)
			{
				w1[j][l]-=alpha*activation_w1[j][l];
				activation_w1[j][l]=0;
			}
			b1[j]-=alpha*activation_b1[j];
			activation_b1[j]=0;
		}
		
		if(s%200==0)
		{
			cout<<"("<<s<<"/"<<60000/batch_size<<")"<<endl;
			Loss/=(batch_size);
			cout<<" Loss:- "<<Loss<<endl;
			Loss=0;
			Predict_accuracy(w1, w2, w3, b1, b2, b3, lab);
		//	getchar();
		}
		else
			Loss=0;
	 

	}

	Predict_accuracy(w1, w2, w3, b1, b2, b3, lab);
}

return 0;
}



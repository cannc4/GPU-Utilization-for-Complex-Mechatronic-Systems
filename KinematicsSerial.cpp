#include<iostream>
#include<fstream>
#include<cmath>
#include"Joint1.h"
#include"Joint2.h"
#include"Joint3.h"
#include"time.h"

using namespace std;


float LA = 40;
float LB = 68;
float RA = 40.50;
float RB = 30;

const float pi = 3.141592;

float Theta1 = 0*pi/180;
float Theta2 = 120*pi/180;
float Theta3 = 240*pi/180;

float nozzle = 8;   

void Delta_Inv(float* x, 
               float* y, 
               float* z, 
               float* alpha1, 
               float* alpha2, 
               float* alpha3)
{

    float R = RA-RB;    
 
    float z_o = *z; 

    float x_2 = pow(*x,2);
    float y_2 = pow(*y,2);
    float z_o_2 = pow(z_o,2);
    float LA_2 = pow(LA,2);
    float LB_2 = pow(LB,2);
    float R_2 = pow(R,2);
    
    float S = (1/LA)*(-x_2-y_2-z_o_2+LB_2-LA_2-R_2);
    
    float Q1 = 2*(*x)*cos(Theta1)+2*(*y)*sin(Theta1);
    float Q2 = 2*(*x)*cos(Theta2)+2*(*y)*sin(Theta2);
    float Q3 = 2*(*x)*cos(Theta3)+2*(*y)*sin(Theta3);
    
    float Q1_2 = pow(Q1,2);
    float Q2_2 = pow(Q2,2);
    float Q3_2 = pow(Q3,2);
    
    float S_2 = pow(S,2);
    
    float T1 = (2*z_o+sqrt(4*z_o_2+4*R_2-S_2+Q1_2*(1-R_2/LA_2)+(Q1)*(-2*R*(S)/LA-4*R)))/(-2*R-(S)-(Q1)*(R/LA-1));
    float T2 = (2*z_o+sqrt(4*z_o_2+4*R_2-S_2+Q2_2*(1-R_2/LA_2)+(Q2)*(-2*R*(S)/LA-4*R)))/(-2*R-(S)-(Q2)*(R/LA-1));
    float T3 = (2*z_o+sqrt(4*z_o_2+4*R_2-S_2+Q3_2*(1-R_2/LA_2)+(Q3)*(-2*R*(S)/LA-4*R)))/(-2*R-(S)-(Q3)*(R/LA-1));
    
    *alpha1 = (-2*atan(T1));
    *alpha2 = (-2*atan(T2));
    *alpha3 = (-2*atan(T3));
    
}

void Delta_Fwd(float alpha1rad, 
               float alpha2rad, 
               float alpha3rad,
               float* x_out,
               float* y_out,
               float* z_out)
{   

    float R = RA-RB;

    float LA_2 = pow(LA,2);
    float LB_2 = pow(LB,2);
    float R_2 = pow(R,2);

    float D1 = -LB_2 + LA_2 + R_2 + 2*R*LA*cos(alpha1rad);
    float E1 = 2*(R+LA*cos(alpha1rad))*cos(Theta1);
    float F1 = 2*(R+LA*cos(alpha1rad))*sin(Theta1);
    float G1 = -2*LA*sin(alpha1rad);

    float D2 = -LB_2 + LA_2 + R_2 + 2*R*LA*cos(alpha2rad);
    float E2 = 2*(R+LA*cos(alpha2rad))*cos(Theta2);
    float F2 = 2*(R+LA*cos(alpha2rad))*sin(Theta2);
    float G2 = -2*LA*sin(alpha2rad);

    float D3 = -LB_2 + LA_2 + R_2 + 2*R*LA*cos(alpha3rad);
    float E3 = 2*(R+LA*cos(alpha3rad))*cos(Theta3);
    float F3 = 2*(R+LA*cos(alpha3rad))*sin(Theta3);
    float G3 = -2*LA*sin(alpha3rad);

    float H1 = E1*G2 - E1*G3 - E2*G1 + E2*G3 + E3*G1 - E3*G2;
    float H2 = -E1*F2 + E1*F3 + E2*F1 - E2*F3 - E3*F1 + E3*F2;
    float H3 = -E1*D2 + E1*D3 + E2*D1 - E2*D3 - E3*D1 + E3*D2;
    float H4 = F1*D2 - F1*D3 - F2*D1 + F2*D3 + F3*D1 - F3*D2;
    float H5 = -F1*G2 + F1*G3 + F2*G1 - F2*G3 - F3*G1 + F3*G2;

    float H1_2 = pow(H1,2);
    float H2_2 = pow(H2,2);  
    float H3_2 = pow(H3,2);
    float H4_2 = pow(H4,2);
    float H5_2 = pow(H5,2);   

    float L = ((H5_2 + H1_2)/H2_2) + 1;
    float M = 2*((H5*H4 + H1*H3)/H2_2) - ((H5*E1 + H1*F1)/H2) - G1;
    float Q = ((H4_2 + H3_2)/H2_2) - ((H4*E1 + H3*F1)/H2) + D1;

    float M_2 = pow(M,2);    

    *z_out = (-M - sqrt(M_2 - 4*L*Q))/(2*L);
    
    *x_out = (*z_out*H5)/H2 + H4/H2;
    *y_out = (*z_out*H1)/H2 + H3/H2;
    *z_out = *z_out; // - nozzle; 
  
}

int main()
{
    //Write results into txt files
	ofstream XYZoutput("XYZoutput.txt");
	XYZoutput << "X\tY\tZ" << endl;
	ofstream AlphaOutput("AlphaOutput.txt");
	AlphaOutput << "Alpha1\tAlpha2\tAlpha3" << endl;
	
	//Start clock
	time_t start,end,diff;
    start = clock();	
	
    float *x = new float[1];// If I don't do like that it gives segmentation fault error
	float *y = new float[1];
	float *z = new float[1];
	*x = 0.0;
	*y = 0.0;
	*z = 0.0;
	
	float *alpha1 = new float[1];// If I don't do like that it gives segmentation fault error
	float *alpha2 = new float[1];
	float *alpha3 = new float[1];
	*alpha1 = 0.0;
	*alpha2 = 0.0;
	*alpha3 = 0.0;
	
	int input;
	
	for(int i=0; i<sizeOfArray; ++i)
	{
        //Compute joint angles from forward kinematics
		Delta_Fwd(Joint1Angle[i],Joint2Angle[i], Joint3Angle[i], x, y, z);
		//Compute task coordinates from invserse kinematics
	    Delta_Inv(x, y, z, alpha1, alpha2, alpha3);
        
        //Write outputs of both functions in columns
        XYZoutput << *x << " " << *y << " " << *z << endl; 
		AlphaOutput << *alpha1 << " " << *alpha2 << " " << *alpha3 << endl;   
	
              
	}
	
	XYZoutput.close();
	AlphaOutput.close();

  //End clock, print computation time
  end = clock();
  diff = (end - start); // /CLOCKS_PER_SEC;
  printf("Time taken by CPU for computation: %d\n",diff);
  
  getchar();
	
	return 0;
}

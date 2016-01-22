#include <iostream>
#include <fstream>
#include <cuda.h>
#include"Joint1.h"
#include"Joint2.h"
#include"Joint3.h"

using namespace std;

cudaEvent_t start, stop;
float time1, sumTime = 0;

const float LA = 40.0;
const float LB = 68.0;
const float LA_2 = LA * LA;
const float LB_2 = LB * LB;

const float RA = 40.50;
const float RB = 30.0;
const float R = RA - RB;
const float R_2 = R * R;

const float PI = 3.141592;
const float THETA = PI * 120 / 180;

template <typename T>
class GpuMemory {
public:
    GpuMemory(int n) : size_(n*sizeof(T)) { cudaMalloc(&ptr_, size_); }
    ~GpuMemory() { cudaFree(ptr_); }
    operator T*() { return ptr_; }
    T* Get() { return ptr_; }
    void ReadFrom(const T* source) { cudaMemcpy(ptr_, source, size_, cudaMemcpyHostToDevice); }
    void WriteTo(T* dest) { cudaMemcpy(dest, ptr_, size_, cudaMemcpyDeviceToHost); }
    void Initialize() { cudaMemset(ptr_, 0, size_); }
private:
    T* ptr_;
    int size_;
};

__global__ void ParallelPowFunction(float *nums, float *pows) {
  int INDEX = blockIdx.x * blockDim.x + threadIdx.x;  ///FOR 1D GRID 1D BLOCK
  pows[INDEX] = nums[INDEX] * nums[INDEX];
}

__global__ void InverseKinematics(float *coords /* x,y,z */,
                                  float *alphas /* alpha1,alpha2,alpha3 */,
                                  float *pows   /* x^2,y^2,z^2 */) {
    int INDEX = blockIdx.x * blockDim.x + threadIdx.x;  ///FOR 1D GRID 1D BLOCK
    /* INDEX =0   ---> x alpha1 theta1
       INDEX =1   ---> y alpha2 theta2
       INDEX =2   ---> z alpha3 theta3
       coords[0] -> x;
       coords[1] -> y;
       coords[2] -> z;
       pows[0]   -> x^2;
       pows[1]   -> y^2;
       pows[2]   -> z^2; */

    float Q = 2 * coords[0] * cos(THETA * INDEX) + 2 * coords[1] * sin(THETA * INDEX);
    float Q_2 = pow(Q,2);

    float S = (1 / LA) * (-pows[0] - pows[1] - pows[2] + LB_2 - LA_2 - R_2);///Same for All
    float S_2 = pow(S,2);///Same for All

    float T_sqrt = sqrt(4 * pows[2] + 4 * R_2 - S_2 + Q_2 * (1 - R_2 / LA_2) + Q * (-2 * R * S / LA - 4*R));
    float T = (2 * coords[2] + T_sqrt) / (-2 * R - S - Q * (R / LA - 1));

    //alphas[INDEX] = (180.0 / PI) * (-2 * atan(T)) - 30.0;
    alphas[INDEX] = -2 * atan(T); //let's calculate it in radians without offset for now
}

#define TERM_D 0
#define TERM_E 1
#define TERM_F 2
#define TERM_G 3

__global__ void ComputeDEFG(const float alphas[3], float defg[4][3]) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const float alpha = alphas[index];

    defg[0][index] = -LB_2 + LA_2 + R_2 + 2 * R * LA * cos(alpha);
    defg[1][index] = 2 * (R + LA * cos(alpha)) * cos(THETA * index);
    defg[2][index] = 2 * (R + LA * cos(alpha)) * sin(THETA * index);
    defg[3][index] = -2 * LA * sin(alpha);
}
/*
__device__ void PrintFloats(const char* name, const float *f, int count) {
    printf("%s: ", name);
    for (int i = 0; i < count; ++i)
        printf("%f, ", f[i]);
    printf("\n");
}
*/
__global__ void ComputeCoords(const float defg[4][3], float coords[3]) {
    __shared__ float products[4][3][4][3];
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    int term1 = index / 36;
    int rest = index % 36;
    int el1 = rest / 12;
    rest = rest % 12;
    int term2 = rest / 3;
    int el2 = rest % 3;
/*
#ifdef DEBUG
    if (!(term1 < 4 && el1 < 3 && term2 < 4 && el2 < 3))
        printf("OOPS. %d %d %d %d\n", term1, el1, term2, el2);
#endif
*/
    products[term1][el1][term2][el2] = defg[term1][el1] * defg[term2][el2];

    __syncthreads();

    if (index > 0)
        return;

    float h[5];
    h[0] = products[TERM_E][0][TERM_G][1] - products[TERM_E][0][TERM_G][2] -
        products[TERM_E][1][TERM_G][0] + products[TERM_E][1][TERM_G][2] +
        products[TERM_E][2][TERM_G][0] - products[TERM_E][2][TERM_G][1];
    h[1] = -(products[TERM_E][0][TERM_F][1] - products[TERM_E][0][TERM_F][2] -
        products[TERM_E][1][TERM_F][0] + products[TERM_E][1][TERM_F][2] +
        products[TERM_E][2][TERM_F][0] - products[TERM_E][2][TERM_F][1]);
    h[2] = -(products[TERM_E][0][TERM_D][1] - products[TERM_E][0][TERM_D][2] -
        products[TERM_E][1][TERM_D][0] + products[TERM_E][1][TERM_D][2] +
        products[TERM_E][2][TERM_D][0] - products[TERM_E][2][TERM_D][1]);
    h[3] = products[TERM_F][0][TERM_D][1] - products[TERM_F][0][TERM_D][2] -
        products[TERM_F][1][TERM_D][0] + products[TERM_F][1][TERM_D][2] +
        products[TERM_F][2][TERM_D][0] - products[TERM_F][2][TERM_D][1];
    h[4] = -(products[TERM_F][0][TERM_G][1] - products[TERM_F][0][TERM_G][2] -
        products[TERM_F][1][TERM_G][0] + products[TERM_F][1][TERM_G][2] +
        products[TERM_F][2][TERM_G][0] - products[TERM_F][2][TERM_G][1]);
/*
#ifdef DEBUG
    PrintFloats("D", defg[0], 3);
    PrintFloats("E", defg[1], 3);
    PrintFloats("F", defg[2], 3);
    PrintFloats("G", defg[3], 3);
    PrintFloats("H", h, 5);
#endif
*/
    float L = ((h[4]  * h[4] + h[0] * h[0]) / (h[1] * h[1])) + 1;
    float M = 2 * ((h[4] * h[3] + h[0] * h[2]) / (h[1] * h[1])) -
        ((h[4] * defg[TERM_E][0] + h[0] * defg[TERM_F][0]) / h[1]) - defg[TERM_G][0];
    float Q = ((h[3] * h[3] + h[2] * h[2]) / (h[1] * h[1])) -
        ((h[3] * defg[TERM_E][0] + h[2] * defg[TERM_F][0]) / h[1]) + defg[TERM_D][0];
/*
#ifdef DEBUG
    float LMQ[3] = {L,M,Q};
    PrintFloats("LMQ", LMQ, 3);
#endif
*/
    coords[2] = (-M - sqrt(M * M - 4 * L * Q)) / (2 * L);

    coords[0] = (coords[2] * h[4]) / h[1] + h[3] / h[1];
    coords[1] = (coords[2] * h[0]) / h[1] + h[2] / h[1];
    coords[2] = coords[2]; // +nozzle will be added
}

void ForwardKinematics(const float alphas[3], float coords[3]) {
    GpuMemory<float> d_alphas(3);
    d_alphas.ReadFrom(alphas);
    GpuMemory<float> d_defg(4 * 3);
    ComputeDEFG <<< 1, 3 >>>(d_alphas, reinterpret_cast<float (*)[3]>(d_defg.Get()));

    GpuMemory<float> d_coords(3);
    ComputeCoords <<< 1, 4 * 3 * 4 * 3 >>>(reinterpret_cast<float (*)[3]>(d_defg.Get()), d_coords);
    d_coords.WriteTo(coords);
}

__global__ void MaptoEncoder(float *alpha) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    float pulse = 360.0f / 4000;
    int temp = (alpha[index]/pulse);
    alpha[index] = temp * pulse;
}

int main() {

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Change these values to compute with different inputs.
    float coords[3] = { 4.4239, 2.60742, -54.1189 }; // x, y, z
    float alphas[3] = { 0.1, 0.2, 0.3 }; // alpha1 , alpha2, alpha3

    GpuMemory<float> coords_D(3);
    GpuMemory<float> alphas_D(3);
    GpuMemory<float> pows_D(3);
    
    ofstream XYZoutput("ParallelXYZoutput.txt");
    XYZoutput << "X\tY\tZ" << endl;
    ofstream AlphaOutput("ParallelAlphaOutput.txt");
    AlphaOutput << "Alpha1\tAlpha2\tAlpha3" << endl;
    
    for(int i=0; i<sizeOfArray; ++i)
    {
    	    alphas[0] = Joint1Angle[i];
    	    alphas[1] = Joint2Angle[i];
    	    alphas[2] = Joint3Angle[i];
    	    	
	    cudaEventRecord(start, 0);
	    
	    alphas_D.ReadFrom(alphas);
	    
	    float coords_out[3];
	    ForwardKinematics(alphas_D, coords_out);
	    
	    coords_D.ReadFrom(coords_out);

	    ParallelPowFunction <<< dim3(1,1,1), dim3(3,1,1) >>> (coords_D, pows_D);
	    
	    float pows[3];   // x^2, y^2, z^2
	    pows_D.WriteTo(pows);

	    InverseKinematics <<< dim3(1,1,1), dim3(3,1,1) >>> (coords_D, alphas_D, pows_D);
	    float alpha_out[3];
	    alphas_D.WriteTo(alpha_out);
	    
	    cudaEventRecord(stop, 0);
	    cudaEventSynchronize(stop);
	    cudaEventElapsedTime(&time1, start, stop);
		
	    sumTime += time1;

	    XYZoutput << coords_out[0] << " " << coords_out[1] << " " << coords_out[2] << endl; 
	    AlphaOutput << alpha_out[0] << " " << alpha_out[1] << " " << alpha_out[2] << endl;

	/*
	    std::cout << "Input coords: " << coords[0] << ", " << coords[1] << ", " << coords[2] << "\n";
	    std::cout << "  inverse kinematics result angles: " << alpha_out[0] << ", " << alpha_out[1] << ", "
		<< alpha_out[2] << "\n";
	    std::cout << "Input alphas: " << alphas[0] << ", " << alphas[1] << ", " << alphas[2] << "\n";
	    std::cout << "  forward kinematics result coords: " << coords_out[0] << ", " << coords_out[1] << ", "
		<< coords_out[2] << "\n";
    	*/
    }
    
    cout << sumTime << endl;

    XYZoutput.close();
    AlphaOutput.close();    
    return 0;
}

/* OUTPUT:
 * Input coords: 4.4239, 2.60742, -54.1189
 *   inverse kinematics result angles: 0.100001, 0.200001, 0.3
 * Input alphas: 0.1, 0.2, 0.3
 *   forward kinematics result coords: 4.4239, 2.60742, -54.1189
 */



#include <iostream>
struct CylindricalVector {
  float rho;

};
struct CartesianVector {
  float x;
  float y;
  float z;
};

__global__ void myKernel(CylindricalVector * cyl, int size) {

  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < size) {
    cyl[idx].rho = 342323;
  }

}




int main() {

  CylindricalVector cyl_array[4];
  for (int i = 0; i < 4; ++i) {
    struct CylindricalVector temp = {0};
    cyl_array[i] = temp;
  }

  CylindricalVector* cyl_array_d;
  CylindricalVector cyl_array_h[4];

  int size = 4 * sizeof(CylindricalVector);
  cudaMalloc(&cyl_array_d, size);

  cudaMemcpy(cyl_array_d, &cyl_array, size, cudaMemcpyHostToDevice);

  myKernel<<<1, 4>>>(cyl_array_d, 4);

  cudaDeviceSynchronize();

  cudaMemcpy(&cyl_array_h, cyl_array_d, size, cudaMemcpyDeviceToHost);

  for (int i = 0; i < 4; ++i) {
    float x = cyl_array_h[i].rho;
    std::cout <<  x << std::endl;
  }



  return 0;
}

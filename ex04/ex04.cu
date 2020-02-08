#include <stdio.h>
// Here you can set the device ID that was assigned to you
#define MYDEVICE 1
__global__
void saxpy(unsigned int n, double a, double *x, double *y)
{
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
}

int main(int argc, char *argv[])
{
  cudaSetDevice(MYDEVICE);

  // 1<<N is the equivalent to 2^N
  unsigned int N = 900 * (1 << 20);

  if (argc > 1) {
    
  }
  double *x, *y, *d_x, *d_y;
  x = (double*)malloc(N*sizeof(double));
  y = (double*)malloc(N*sizeof(double));

  cudaMalloc(&d_x, N*sizeof(double)); 
  cudaMalloc(&d_y, N*sizeof(double));

  for (unsigned int i = 0; i < N; i++) {
    x[i] = 1.0;
    y[i] = 2.0;
  }

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaMemcpy(d_x, x, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N*sizeof(double), cudaMemcpyHostToDevice);

  cudaEventRecord(start);

  saxpy<<<(N+511)/512, 512>>>(N, 2.0, d_x, d_y);

  cudaEventRecord(stop);

  cudaMemcpy(y, d_y, N*sizeof(double), cudaMemcpyDeviceToHost);

  cudaEventSynchronize(stop);


  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  printf("Time elapsed: %f \n", milliseconds);

  double kernelThroughput = (sizeof(unsigned int)+sizeof(double)+N*sizeof(double)+N*sizeof(double))/(milliseconds*std::pow(10, 3));
  kernelThroughput = kernelThroughput/(pow(10, 9));
  printf("Throughput in Gigabyte/s: %f \n", kernelThroughput);

  double deviceThroughput = (1752*std::pow(10.0, 6.0)*512)/(std::pow(10.0, 9.0));
  printf("Device throughput: %f \n", deviceThroughput);

  double maxError = 0.;
  for (unsigned int i = 0; i < N; i++) {
    maxError = max(maxError, abs(y[i]-4.0));
  }
  
  cudaFree(d_x);
  cudaFree(d_y);
  free(x);
  free(y);

}




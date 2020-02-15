#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <numeric>
#include <iostream>

// Here you can set the device ID that was assigned to you
#define MYDEVICE 0

double random_double(void)
{

  return static_cast<double>(rand()) / RAND_MAX;
}


// Part 1 of 6: implement the kernel
__global__ void block_sum(const double *input,
                          double *per_block_results,
                          const size_t n)
{
    int g_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (g_idx < n) {

    const int BLOCKSIZE = 80;
    __shared__ double s[BLOCKSIZE];
    // every thread -> one element to shared memory
    s[threadIdx.x] = input[g_idx];

    __syncthreads();

    for (int i=1; i<blockDim.x; i = 2*i) {
       if (threadIdx.x % (2*i) == 0) {
            s[threadIdx.x] = s[threadIdx.x] + s[threadIdx.x + i];
       }
    }
    __syncthreads();

    if (threadIdx.x == 0) {per_block_results[blockIdx.x] = s[0];}

    }
 





}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(void)
{
  // create array of 256ki elements
  const int num_elements = 1<<18;
  srand(time(NULL));
  // generate random input on the host
  std::vector<double> h_input(num_elements);
  for(int i = 0; i < h_input.size(); ++i)
  {
    h_input[i] = random_double();
  }

  const double host_result = std::accumulate(h_input.begin(), h_input.end(), 0.0f);
  std::cerr << "Host sum: " << host_result << std::endl;

  //Part 1 of 6: move input to device memory
  double *d_input = 0;
  cudaMalloc(&d_input, h_input.size()*sizeof(double));
  int size_mem = h_input.size()*sizeof(double);
  cudaMemcpy(d_input, &h_input[0], size_mem, cudaMemcpyHostToDevice);

  // Part 1 of 6: allocate the partial sums: How much space does it need?
  int block_size = 80;
  int num_blocks = 3277;
  double *d_partial_sums_and_total = 0;
  cudaMalloc(&d_partial_sums_and_total, num_blocks*sizeof(double));

  // Part 1 of 6: launch one kernel to compute, per-block, a partial sum. How much shared memory does it need?
  block_sum<<<num_blocks,block_size>>>(d_input, d_partial_sums_and_total, num_elements);

  // Part 1 of 6: compute the sum of the partial sums
  double *d_temp = 0;
  cudaMalloc(&d_temp, sizeof(double));
  block_sum<<<1, num_blocks>>>(d_partial_sums_and_total, d_temp, num_blocks);

  // Part 1 of 6: copy the result back to the host
  double device_result = 0;
  cudaMemcpy(&device_result, &d_temp[0], sizeof(double), cudaMemcpyDeviceToHost);

  std::cout << "Device sum: " << device_result << std::endl;

  // Part 1 of 6: deallocate device memory
  cudaFree(d_partial_sums_and_total); cudaFree(d_input); cudaFree(d_temp);

  return 0;
}

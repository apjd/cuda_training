// system include files
#include <cmath>

// CUDA include files
#include <cuda_runtime.h>

// CMSSW include files
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "cudavectors.h"

#include <stdio.h>


namespace cudavectors {

  __host__ __device__ inline void convert(CylindricalVector const& cylindrical, CartesianVector & cartesian) {
    // fill here ...
    cartesian.x = cylindrical.rho * cos(cylindrical.phi);
    cartesian.y = cylindrical.rho * sin(cylindrical.phi);
    cartesian.z = cylindrical.rho * sinh(cylindrical.eta);
  }

  __global__ void convertKernel(CylindricalVector const* cylindrical, CartesianVector* cartesian, size_t size) {
    // fill here ...
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < size) {
      convert(cylindrical[idx], cartesian[idx]);
    }

  }

  void convertWrapper(CylindricalVector const* cylindrical, CartesianVector* cartesian, size_t size) {
    // fill here ...
    //convertKernel<<<gridSize, blockSize>>>(cylindrical, cartesian, size);
    int gridSize = 10;
    int blockSize = 1000;
    convertKernel<<<gridSize, blockSize>>>(cylindrical, cartesian, size);
    //cudaCheck(cudaGetLastError());
  }

}  // namespace cudavectors

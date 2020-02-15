// system include files
#include <cmath>
#include <memory>
#include <vector>

#include <iostream>

// CMSSW include files
#include "DataFormats/Math/interface/Vector3D.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

#include "cudavectors.h"

#define MYDEVICE 1

class ConvertToCartesianVectorsCUDA : public edm::stream::EDProducer<> {
public:
  explicit ConvertToCartesianVectorsCUDA(const edm::ParameterSet&);
  ~ConvertToCartesianVectorsCUDA() = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  using CartesianVectors = std::vector<math::XYZVectorF>;
  using CylindricalVectors = std::vector<math::RhoEtaPhiVectorF>;

  virtual void produce(edm::Event&, const edm::EventSetup&) override;

  edm::EDGetTokenT<CylindricalVectors> input_;
  edm::EDPutTokenT<CartesianVectors> output_;
};

ConvertToCartesianVectorsCUDA::ConvertToCartesianVectorsCUDA(const edm::ParameterSet& config)
    : input_(consumes<CylindricalVectors>(config.getParameter<edm::InputTag>("input"))) {
  output_ = produces<CartesianVectors>();
}

void ConvertToCartesianVectorsCUDA::produce(edm::Event& event, const edm::EventSetup& setup) {

  cudaSetDevice(MYDEVICE);
  auto const& input = event.get(input_);
  auto elements = input.size();
  auto product = std::make_unique<CartesianVectors>(elements);

  // allocate memory on the GPU for the cylindrical and cartesian vectors
  // fill here ...

  cudavectors::CartesianVector* cartesian_d;
  cudavectors::CylindricalVector* cylindrical_d;
  cudavectors::CartesianVector cartesian_h[elements];
  cudavectors::CylindricalVector cylindrical_h[elements];

  size_t size = elements * sizeof(cudavectors::CylindricalVector); // every structures defines three float numbers

  cudaCheck(cudaMalloc(&cylindrical_d, size));
  cudaCheck(cudaMalloc(&cartesian_d, size));


  // copy the input data to the GPU
  // fill here ...

  for (long unsigned int i = 0; i < elements; ++i) {
    cylindrical_h[i].rho = input[i].rho();
    cylindrical_h[i].eta = input[i].eta();
    cylindrical_h[i].phi = input[i].phi();

  }

  cudaCheck(cudaMemcpy(cylindrical_d, cylindrical_h, size, cudaMemcpyHostToDevice));

  // convert the vectors from cylindrical to cartesian coordinates, on the GPU
  // fill here ...
  cudavectors::convertWrapper(cylindrical_d, cartesian_d, elements);

  cudaCheck(cudaDeviceSynchronize());


  // copy the result from the GPU
  // fill here ...
  //cudavectors::CartesianVector result;
  cudaCheck(cudaMemcpy(&cartesian_h, cartesian_d, size, cudaMemcpyDeviceToHost));


  // free the GPU memory
  // fill here ...
  cudaCheck(cudaFree(cylindrical_d)); cudaCheck(cudaFree(cartesian_d));

  for (long unsigned int i = 0; i < elements; ++i) {
    ((*product)[i]).SetCoordinates(cartesian_h[i].x, cartesian_h[i].y, cartesian_h[i].z);
  }

  event.put(output_, std::move(product));

}

void ConvertToCartesianVectorsCUDA::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("input", edm::InputTag("cylindricalVectors"));
  descriptions.addWithDefaultLabel(desc);
}

// define this as a plug-in
DEFINE_FWK_MODULE(ConvertToCartesianVectorsCUDA);

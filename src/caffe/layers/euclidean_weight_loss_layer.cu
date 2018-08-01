#include <vector>

#include "caffe/layers/euclidean_weight_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void EuclideanWeightLossForwardGPU(const int nthreads,
	const Dtype* ldata, const float rw, const float rh,
	const float h_2, const float w_2, const int dim, Dtype* wdata) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		const int ni = index*dim;
		float minx = ldata[ni];
		float maxx = minx;
		float miny = ldata[ni + 1];
		float maxy = miny;
		for (int j = 2; j < dim; j += 2){
			if (ldata[ni + j] < minx) minx = ldata[ni + j];
			else if (ldata[ni + j] > maxx) maxx = ldata[ni + j];
			if (ldata[ni + j + 1] < miny) miny = ldata[ni + j + 1];
			else if (ldata[ni + j + 1] > maxy) maxy = ldata[ni + j + 1];
		}
		miny = h_2*(maxy-miny);
	  	minx = w_2*(maxx-minx);
		for (int j = 0; j < dim; j+=2){
			wdata[ni+j] = rw/minx;
		  	wdata[ni+j+1] = rh/miny;
		}		
	}
}

template <typename Dtype>
void EuclideanWeightLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  
	const int dim = bottom[0]->count() / bottom[0]->num();
	const float h_2 = (this->layer_param_.euclidean_weight_loss_param().dh() - 1)*0.5;
	const float w_2 = (this->layer_param_.euclidean_weight_loss_param().dw() - 1)*0.5;
	const float rh = this->layer_param_.euclidean_weight_loss_param().rh()*this->layer_param_.euclidean_weight_loss_param().dh();   
  	const float rw = this->layer_param_.euclidean_weight_loss_param().rw()*this->layer_param_.euclidean_weight_loss_param().dw(); 
	EuclideanWeightLossForwardGPU<Dtype> << <CAFFE_GET_BLOCKS(bottom[0]->num()),
	  CAFFE_CUDA_NUM_THREADS >> >(bottom[0]->num(), bottom[1]->gpu_data(), rw,rh, h_2, w_2, dim, data_.mutable_gpu_diff());

  int count = bottom[0]->count();
  caffe_gpu_sub(
      count,
	  bottom[0]->gpu_data(),
      bottom[1]->gpu_data(),
      diff_.mutable_gpu_data());

  caffe_gpu_mul(count, data_.gpu_diff(), diff_.gpu_data(), data_.mutable_gpu_data());

  Dtype dot;
  caffe_gpu_dot(count, data_.gpu_data(), diff_.gpu_data(), &dot);
  Dtype loss = dot / bottom[0]->num() / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void EuclideanWeightLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      caffe_gpu_axpby(
          bottom[i]->count(),              // count
          alpha,                              // alpha
		  data_.gpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_gpu_diff());  // b
	}	
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(EuclideanWeightLossLayer);

}  // namespace caffe

#include <vector>

#include "caffe/layers/accuracy_distance_layer.hpp"
#include "caffe/util/math_functions.hpp"


namespace caffe {

template <typename Dtype>
__global__ void AccuracyForwardGPU(const int nthreads,
          const Dtype* bottom_data, const Dtype* bottom_label, Dtype* acc,
          const int dim, const float h_2, const float w_2) {
  CUDA_KERNEL_LOOP(i, nthreads) {
    acc[i] = 0;
    for(int j=0;j<dim;j+=2){
	int x1 = bottom_data[i*dim+j]*w_2+w_2+0.5;
	int x2 = bottom_label[i*dim+j]*w_2+w_2+0.5;
	int y1 = bottom_data[i*dim+j+1]*h_2+h_2+0.5;
	int y2 = bottom_label[i*dim+j+1]*h_2+h_2+0.5;
        float v = (x1-x2)*(x1-x2)+(y1-y2)*(y1-y2);
	acc[i] += sqrtf(v);       
    }
  }
}

template <typename Dtype>
void AccuracyDistanceLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* bottom_label = bottom[1]->gpu_data();
  const int nthreads = bottom[1]->num();

  Dtype* acc_data = bottom[1]->mutable_gpu_diff();
  if (top.size() == 1) {
    AccuracyForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, bottom_data, bottom_label,
        acc_data, bottom[1]->channels(), dh_2, dw_2);
    Dtype acc;
    caffe_gpu_asum(nthreads, acc_data, &acc);
    Dtype valid_count = bottom[1]->num();
    if (valid_count > 0) {
      top[0]->mutable_cpu_data()[0] = acc / bottom[1]->num()/bottom[1]->channels()*2.0;
    } else {
      top[0]->mutable_cpu_data()[0] = 0;
    }
  } else {
    top[0]->mutable_cpu_data()[0] = 0;
  }
}


template <typename Dtype>
void AccuracyDistanceLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {  NOT_IMPLEMENTED;  }
}

INSTANTIATE_LAYER_GPU_FUNCS(AccuracyDistanceLayer);
}  // namespace caffe

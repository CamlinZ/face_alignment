#include <vector>

#include "caffe/layers/euclidean_weight_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void EuclideanWeightLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  diff_.ReshapeLike(*bottom[0]);
  data_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void EuclideanWeightLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  
  const int dim = bottom[0]->count()/bottom[0]->num();
  const float h_2 = (this->layer_param_.euclidean_weight_loss_param().dh() - 1)*0.5;
  const float w_2 = (this->layer_param_.euclidean_weight_loss_param().dw() - 1)*0.5;
  const float rh = this->layer_param_.euclidean_weight_loss_param().rh()*this->layer_param_.euclidean_weight_loss_param().dh();   
  const float rw = this->layer_param_.euclidean_weight_loss_param().rw()*this->layer_param_.euclidean_weight_loss_param().dw();  
  for (int i = 0; i < bottom[0]->num(); i++){
	  const Dtype *ldata = bottom[1]->cpu_data() + bottom[1]->offset(i);
	  float minx = ldata[0];
	  float maxx = minx;
	  float miny = ldata[1];
	  float maxy = miny;
	  for (int j = 2; j < dim; j += 2){
		  if (ldata[j] < minx) minx = ldata[j];
		  else if (ldata[j] > maxx) maxx = ldata[j];
		  if (ldata[j + 1] < miny) miny = ldata[j + 1];
		  else if (ldata[j + 1] > maxy) maxy = ldata[j + 1];
	  }
	  miny = h_2*(maxy-miny);
	  minx = w_2*(maxx-minx);
	  Dtype *wdata = data_.mutable_cpu_diff() + data_.offset(i);
	  for (int j = 0; j < dim; j+=2){
		  wdata[j] = rw/minx;
		  wdata[j+1] = rh/miny;
	  }
  }

  int count = bottom[0]->count();
  caffe_sub(
      count,
	  bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff_.mutable_cpu_data());
  //caffe_gpu_mul(count, data_.cpu_diff(), diff_.gpu_data(), data_.mutable_cpu_data());

  Dtype dot = caffe_cpu_dot(count, data_.cpu_data(), diff_.cpu_data());
  Dtype loss = dot / bottom[0]->num() / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void EuclideanWeightLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      caffe_cpu_axpby(
          bottom[i]->count(),              // count
          alpha,                              // alpha
		  data_.cpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_cpu_diff());  // b
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(EuclideanWeightLossLayer);
#endif

INSTANTIATE_CLASS(EuclideanWeightLossLayer);
REGISTER_LAYER_CLASS(EuclideanWeightLoss);

}  // namespace caffe

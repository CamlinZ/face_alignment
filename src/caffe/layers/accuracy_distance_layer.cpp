#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/accuracy_distance_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void AccuracyDistanceLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  dh_2 = (this->layer_param_.accuracy_distance_param().dh()-1)*0.5;
  dw_2 = (this->layer_param_.accuracy_distance_param().dw()-1)*0.5;
  CHECK_GE(this->layer_param_.accuracy_distance_param().dh(),1)<<"dh must greater than 1";
  CHECK_GE(this->layer_param_.accuracy_distance_param().dw(),1)<<"dw must greater than 1";
  has_ignore_label_ =
    this->layer_param_.accuracy_distance_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.accuracy_distance_param().ignore_label();
  }
}

template <typename Dtype>
void AccuracyDistanceLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  label_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.accuracy_distance_param().axis());
  outer_num_ = bottom[0]->count(0, label_axis_);
  inner_num_ = bottom[0]->count(label_axis_ + 1);
  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count()/bottom[1]->channels())
      << "Number of labels must match number of predictions; ";
  vector<int> top_shape(0);  // Accuracy is a scalar; 0 axes.
  top[0]->Reshape(top_shape);
  CHECK_EQ(top.size(),1)<<"only can set 1 top out";
}

template <typename Dtype>
void AccuracyDistanceLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype accuracy = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  
  for(int i=0;i<bottom[1]->num();i++){    
     int offset = bottom[1]->offset(i);
     for(int j=0;j<bottom[1]->channels();j+=2){
	int x1 = bottom_data[offset+j]*dw_2+dw_2+0.5;
	int x2 = bottom_label[offset+j]*dw_2+dw_2+0.5;
	int y1 = bottom_data[offset+j+1]*dh_2+dh_2+0.5;
	int y2 = bottom_label[offset+j+1]*dh_2+dh_2+0.5;
        float v = (x1-x2)*(x1-x2)+(y1-y2)*(y1-y2);
        accuracy += sqrtf(v);
     }
  }
  top[0]->mutable_cpu_data()[0] = accuracy / bottom[1]->num()/bottom[1]->channels()*2.0;

  // Accuracy layer should not be used as a loss function.
}

#ifdef CPU_ONLY
STUB_GPU(AccuracyDistanceLayer);
#endif

INSTANTIATE_CLASS(AccuracyDistanceLayer);
REGISTER_LAYER_CLASS(AccuracyDistance);

}  // namespace caffe

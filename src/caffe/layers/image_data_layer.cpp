#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/image_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
ImageDataLayer<Dtype>::~ImageDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
int ImageDataLayer<Dtype>::Rand(int n) {
  if (n < 1) return 1;
  caffe::rng_t* rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  return ((*rng)() % n);
}


template <typename Dtype>
void ImageDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int new_height = this->layer_param_.image_data_param().new_height();
  const int new_width  = this->layer_param_.image_data_param().new_width();
  const bool is_color  = this->layer_param_.image_data_param().is_color();
  const bool shuffleflag = this->layer_param_.image_data_param().shuffle();
  string root_folder = this->layer_param_.image_data_param().root_folder();
  
  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
  // Read the file with filenames and labels
  const string& source = this->layer_param_.image_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string line;
  int pos;
  int label_dim = 0;
  bool gfirst = true;
  int rd = shuffleflag?4:0;
  while (std::getline(infile, line)) {
	if(line.find_last_of(' ')==line.size()-2) line.erase(line.find_last_not_of(' ')-1);	
        pos = line.find_first_of(' ');	
	string str = line.substr(0, pos);
	int p0 = pos + 1;
	vector<float> vl;
	while (pos != -1){
		pos = line.find_first_of(' ', p0);
		vl.push_back(atof(line.substr(p0, pos).c_str()));
		p0 = pos + 1;
	}	
	if (shuffleflag) {
		float minx = vl[0];
		float maxx = minx;
		float miny = vl[1];
		float maxy = miny;
		for (int i = 2; i < vl.size(); i += 2){
			if (vl[i] < minx) minx = vl[i];
			else if (vl[i] > maxx) maxx = vl[i];
			if (vl[i + 1] < miny) miny = vl[i + 1];
			else if (vl[i + 1] > maxy) maxy = vl[i + 1];
		}
		vl.push_back(minx);
		vl.push_back(maxx + 1);
		vl.push_back(miny);
		vl.push_back(maxy + 1);
	}
	if (gfirst){
		label_dim = vl.size();
		gfirst = false;
		LOG(INFO) << "label dim: " << label_dim - rd;
		//LOG(INFO) << line;		
	}
	CHECK_EQ(vl.size(), label_dim)  << "label dim not match in: " << lines_.size()<<", "<<lines_[lines_.size()-1].first;
	lines_.push_back(std::make_pair(str, vl));
  }

  CHECK(!lines_.empty()) << "File is empty";

  if (shuffleflag) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data & randomly crop image";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  } else {
    if (this->phase_ == TRAIN && Caffe::solver_rank() > 0 &&
        this->layer_param_.image_data_param().rand_skip() == 0) {
      LOG(WARNING) << "Shuffling or skipping recommended for multi-GPU";
    }
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";

  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.image_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.image_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }
  // Read an image, and use it to initialize the top blob.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
                                    0, 0, is_color);
  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
  // Use data_transformer to infer the expected blob shape from a cv_image.
  vector<int> top_shape(4);
  top_shape[0] = 1;
  top_shape[1] = cv_img.channels();
  top_shape[2] = shuffleflag ? new_height : cv_img.rows;
  top_shape[3] = shuffleflag ? new_width : cv_img.cols;
  this->transformed_data_.Reshape(top_shape);
  // Reshape prefetch_data and top[0] according to the batch_size.
  const int batch_size = this->layer_param_.image_data_param().batch_size();
  CHECK_GT(batch_size, 0) << "Positive batch size required";
  top_shape[0] = batch_size;
  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->data_.Reshape(top_shape);
  }
  top[0]->Reshape(top_shape);

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  vector<int> label_shape(2, batch_size);
  label_shape[1] = label_dim-rd;
  top[1]->Reshape(label_shape);
  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->label_.Reshape(label_shape);
  }
}

template <typename Dtype>
void ImageDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is called on prefetch thread
template <typename Dtype>
void ImageDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  ImageDataParameter image_data_param = this->layer_param_.image_data_param();
  const int batch_size = image_data_param.batch_size();
  const float rate_height = this->layer_param_.image_data_param().rate_height();
  const float rate_width = this->layer_param_.image_data_param().rate_width();
  const bool is_color = image_data_param.is_color();
  const bool shuffleflag = this->layer_param_.image_data_param().shuffle();
  string root_folder = image_data_param.root_folder();

  // Reshape according to the first image of each batch
  // on single input batches allows for inputs of varying dimension.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
      0, 0, is_color);
  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
  const int new_height = shuffleflag ? image_data_param.new_height() : cv_img.rows;
  const int new_width = shuffleflag ? image_data_param.new_width() : cv_img.cols;
  // Use data_transformer to infer the expected blob shape from a cv_img.
  vector<int> top_shape(4);
  top_shape[0] = 1;
  top_shape[1] = cv_img.channels();
  top_shape[2] = new_height;
  top_shape[3] = new_width;
  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);
  vector<int> top_shape1(4);
  top_shape1[0] = batch_size;
  top_shape1[1] = shuffleflag ? lines_[0].second.size() - 4 : lines_[0].second.size();
  top_shape1[2] = 1;
  top_shape1[3] = 1; 
  batch->label_.Reshape(top_shape1);

  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();

  // datum scales
  const int lines_size = lines_.size();
  const float dh_2 = (new_height - 1)*0.5;
  const float dw_2 = (new_width - 1)*0.5;
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    timer.Start();
    CHECK_GT(lines_size, lines_id_);
    cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
        0, 0, is_color);
    CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply transformations (mirror, crop...) to the image
	int x1 = 0;
	int y1 = 0;
	int x2 = cv_img.cols;
	int y2 = cv_img.rows;
	if (shuffleflag){
		CHECK_GE(cv_img.rows, new_height) << lines_[lines_id_].first;
		CHECK_GE(cv_img.cols, new_width) << lines_[lines_id_].first;
		int minx = lines_[lines_id_].second[top_shape1[1]];
		int maxx = lines_[lines_id_].second[top_shape1[1] + 1];
		int miny = lines_[lines_id_].second[top_shape1[1] + 2];
		int maxy = lines_[lines_id_].second[top_shape1[1] + 3];
		x1 = Rand(2 * round(rate_width*cv_img.cols));
		y1 = Rand(2 * round(rate_height*cv_img.rows));
		x2 = x1 + new_width;
		y2 = y1 + new_height;
		if (x1 > minx){
			x2 -= x1 - minx;
			x1 = minx;
		}
		if (x2 < maxx){
			x1 += maxx - x2;
			x2 = maxx;
		}
		if (x1<0){
			x2 += -x1;
			x1 = 0;
		}
		if (x2 > cv_img.cols){
			x1 -= x2 - cv_img.cols;
			x2 = cv_img.cols;
		}
		if (y1 > miny){
			y2 -= y1 - miny;
			y1 = miny;
		}
		if (y2 < maxy){
			y1 += maxy - y2;
			y2 = maxy;
		}
		if (y1<0){
			y2 += -y1;
			y1 = 0;
		}
		if (y2>cv_img.rows){
			y1 -= y2 - cv_img.rows;
			y2 = cv_img.rows;
		}
	}
	if (y2 - y1 != new_height || x2 - x1 != new_width){
		printf("%s y1:%d, y2:%d, x1:%d, x2:%d\n", lines_[lines_id_].first.c_str(),y1,y2,x1,x2);
	}
    //
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(prefetch_data + offset);
    this->data_transformer_->Transform(cv_img(cv::Range(y1, y2), cv::Range(x1, x2)), &(this->transformed_data_));
    trans_time += timer.MicroSeconds();

    for (int i = 0; i < top_shape1[1]; i++){
      if (i % 2 == 0) prefetch_label[item_id*top_shape1[1] + i] = (lines_[lines_id_].second[i] - x1 - dw_2) / dw_2;
      else prefetch_label[item_id*top_shape1[1] + i] = (lines_[lines_id_].second[i] - y1 - dh_2) / dh_2;
    }
    // go to the next iter
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (shuffleflag) {
        ShuffleImages();
      }
    }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(ImageDataLayer);
REGISTER_LAYER_CLASS(ImageData);

}  // namespace caffe
#endif  // USE_OPENCV

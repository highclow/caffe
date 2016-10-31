#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/image_pair_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

//Originally by Ashish Shrivastava, refactored for use in caffe-sl by Isay Katsman

namespace caffe {

template <typename Dtype>
ImagePairDataLayer<Dtype>::~ImagePairDataLayer<Dtype>() {
  //not in use anymore: this->JoinPrefetchThread();
  this->StopInternalThread();
}

template <typename Dtype>
void ImagePairDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int new_height = this->layer_param_.image_pair_data_param().new_height();
  const int new_width  = this->layer_param_.image_pair_data_param().new_width();
  const bool is_color  = this->layer_param_.image_pair_data_param().is_color();
  string root_folder = this->layer_param_.image_pair_data_param().root_folder();

  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
  // Read the file with filenames and labels
  const string& source = this->layer_param_.image_pair_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string filename1, filename2;
  int label;
  while (infile >> filename1 >> filename2 >> label) {
    lines_.push_back(std::make_pair(std::make_pair(filename1, filename2), label));
  }

  if (this->layer_param_.image_pair_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  }
  LOG(INFO) << "A total of " << lines_.size() << " image pairs.";

  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.image_pair_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.image_pair_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }
  // Read an image, and use it to initialize the top blob.
  cv::Mat cv_img1 = ReadImageToCVMat(root_folder + lines_[lines_id_].first.first,
                                    new_height, new_width, is_color);
  const int channels = 2*cv_img1.channels(); // 2 time number of channels for pair of image
  const int height = cv_img1.rows;
  const int width = cv_img1.cols;
  // image
  const int crop_size = this->layer_param_.transform_param().crop_size();
  const int crop_h = crop_size ? crop_size: this->layer_param_.transform_param().crop_h();
  const int crop_w = crop_size ? crop_size: this->layer_param_.transform_param().crop_w();
  const int batch_size = this->layer_param_.image_pair_data_param().batch_size();
  if (crop_size > 0 || crop_h || crop_w) {
    top[0]->Reshape(batch_size, channels, crop_h, crop_w);
    for (int i = 0; i < this->PREFETCH_COUNT; i++) {
      this->prefetch_[i].data_.Reshape(batch_size, channels, crop_h, crop_w);
    }
    this->transformed_data_.Reshape(1, channels, crop_h, crop_w);
  } else {
    top[0]->Reshape(batch_size, channels, height, width);
    for (int i = 0; i < this->PREFETCH_COUNT; i++) {
      this->prefetch_[i].data_.Reshape(batch_size, channels, height, width);
    }
    this->transformed_data_.Reshape(1, channels, height, width);
  }
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  vector<int> label_shape(1, batch_size);
  top[1]->Reshape(label_shape);
  for (int i = 0; i < this->PREFETCH_COUNT; i++) {
    this->prefetch_[i].label_.Reshape(label_shape);
  }
}

template <typename Dtype>
void ImagePairDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void ImagePairDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  ImagePairDataParameter image_pair_data_param = this->layer_param_.image_pair_data_param();
  const int batch_size = image_pair_data_param.batch_size();
  const int new_height = image_pair_data_param.new_height();
  const int new_width = image_pair_data_param.new_width();
  //unused variable: const int crop_size = this->layer_param_.transform_param().crop_size();
  const bool is_color = image_pair_data_param.is_color();
  string root_folder = image_pair_data_param.root_folder();

  CHECK_GT(new_height, 0) << "New height should be greater than zero." <<
		  	  	  	  	  "This ensures that the pair of images are of same size." <<
		  	  	  	  	  "otherwise they cannot be concatenated";
  CHECK_GT(new_width, 0)  << "New width should be greater than zero." <<
						  "This ensures that the pair of images are of same size." <<
						  "otherwise they cannot be concatenated";


  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();

  // datum scales
  const int lines_size = lines_.size();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    timer.Start();
    CHECK_GT(lines_size, lines_id_);
    cv::Mat cv_img1 = ReadImageToCVMat(root_folder + lines_[lines_id_].first.first,
        new_height, new_width, is_color);
    CHECK(cv_img1.data) << "Could not load " << lines_[lines_id_].first.first;

    cv::Mat cv_img2 = ReadImageToCVMat(root_folder + lines_[lines_id_].first.second,
        new_height, new_width, is_color);
    CHECK(cv_img2.data) << "Could not load " << lines_[lines_id_].first.second;

    // merge two images into on Mat array of 6 channel
    // this will be split by Slice layer for siamese network training

    cv::Mat cv_img_arr[] = {cv_img1, cv_img2};
    cv::Mat cv_img;

    cv::merge(cv_img_arr, 2, cv_img); // cv_img is 6 channel matrix now

    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply transformations (mirror, crop...) to the image
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(prefetch_data + offset);
    this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
    trans_time += timer.MicroSeconds();

    prefetch_label[item_id] = lines_[lines_id_].second;
    // go to the next iter
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.image_pair_data_param().shuffle()) {
        ShuffleImages();
      }
    }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(ImagePairDataLayer);
REGISTER_LAYER_CLASS(ImagePairData);

}  // namespace caffe

#endif //USE_OPENCV

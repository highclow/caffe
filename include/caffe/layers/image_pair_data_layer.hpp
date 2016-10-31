#ifndef CAFFE_IMAGE_PAIR_DATA_LAYER
#define CAFFE_IMAGE_PAIR_DATA_LAYER

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Provides data to the Net from pair of image files.
 *         Usefule for training siamese network
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class ImagePairDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit ImagePairDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~ImagePairDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ImagePairData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

  //Ashish-- This function helps get the name of image file by returning line_id
  // which is the next line number in source file. Starts from 0.  
  int getLinesId() {  return lines_id_;  }
  //Ashish-- This function helps get the name of image file by returning all image names
  // which are stored in the protected variables lines_id_
  vector<std::pair<std::pair<std::string, std::string>, int> > getLines() {
    // make deep copy of the protected variable
   vector<std::pair<std::pair<std::string, std::string>, int> > all_lines = lines_; 
    // return it
    return all_lines;
  }

 protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;
  virtual void ShuffleImages();
  virtual void load_batch(Batch<Dtype>* batch);

  vector<std::pair<std::pair<std::string, std::string>, int> >  lines_;
  int lines_id_;
};


} //namespace caffe

#endif //CAFFE_IMAGE_PAIR_DATA_LAYER

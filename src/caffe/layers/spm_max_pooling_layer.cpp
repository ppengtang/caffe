#include "caffe/layers/spm_layer.hpp"

namespace caffe {

template <typename Dtype>
void SPMMaxPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  SPMParameter spm_param = this->layer_param().spm_param();
  const int SPM_type_ = spm_param.spm_type();
  if (SPM_type_ == 0) {
    num_grid = 1;
  }
  else {
    if (SPM_type_ == 1) {
      num_grid = 8;
    }
    else {
      num_grid = 21;
    }
  } 
}

template <typename Dtype>
void SPMMaxPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_GE(bottom[0]->num_axes(), 2)
      << "Number of axes of bottom blob must be >=2.";
  int channels = bottom[0]->channels();
  int batch_size = bottom[1]->num();
  
  vector<int> top_shape(2);
  top_shape[0] = bottom[1]->num();
  top_shape[1] = channels * num_grid;
  top[0]->Reshape(top_shape);
  
  vector<int> idx_shape(1);
  idx_shape[0] = channels * num_grid * batch_size;
  max_idx_.Reshape(idx_shape);
}

template <typename Dtype>
void SPMMaxPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  int* real_max_idx = max_idx_.mutable_cpu_data();
  
  int num_rois = bottom[2]->num();
  const Dtype* bottom_rois = bottom[2]->cpu_data();
  const Dtype* bottom_shape = bottom[1]->cpu_data();
  int batch_size = bottom[1]->num();
  const int channels = bottom[0]->channels();

  float spm0[32] = {0, 1, 0, 1, 0, 0.5, 0, 0.5, 0, 0.5, 0.5, 1, 0.5, 1, 0, 0.5, 
      0.5, 1, 0.5, 1, 0, 1, 0, 0.33, 0, 1, 0.33, 0.67, 0, 1, 0.67, 1};
  float spm1[84] = {0, 1, 0, 1, 0, 0.5, 0, 0.5, 0, 0.5, 0.5, 1, 0.5, 1, 0, 0.5,
      0.5, 1, 0.5, 1, 0, 0.25, 0, 0.25, 0, 0.25, 0.25, 0.5, 0, 0.25, 0.5, 0.75, 
      0, 0.25, 0.75, 1, 0.25, 0.5, 0, 0.25, 0.25, 0.5, 0.25, 0.5, 0.25, 0.5, 0.5, 0.75, 
      0.25, 0.5, 0.75, 1, 0.5, 0.75, 0, 0.25, 0.5, 0.75, 0.25, 0.5, 0.5, 0.75, 0.5, 0.75, 
      0.5, 0.75, 0.75, 1, 0.75, 1, 0, 0.25, 0.75, 1, 0.25, 0.5, 0.75, 1, 0.5, 0.75, 
      0.75, 1, 0.75, 1};
  float * spm;
  if (num_grid > 16) {
    spm = spm1;
  }
  else {
    spm = spm0;
  }
  
  int batch_id = 0;

  float center_x = float(bottom_rois[1] + bottom_rois[3]) / (2.0 * float(bottom_shape[0]));
  float center_y = float(bottom_rois[2] + bottom_rois[4]) / (2.0 * float(bottom_shape[1]));
  for (int i = 0; i < num_grid; i++) {
    if (center_x >= spm[i * 4] && center_x < spm[i * 4 + 1] 
        && center_y >= spm[i * 4 + 2] && center_y < spm[i * 4 + 3]) {
      for (int c = 0; c < channels; c++) {
        top_data[i * channels + c] = bottom_data[c];
        real_max_idx[i * channels + c] = 0;
      }
    }
    else {
      for (int c = 0; c < channels; c++) {
        top_data[i * channels + c] = -1000000;
        real_max_idx[i * channels + c] = -1;
      }
    } 
  }
  bottom_rois += bottom[2]->offset(1);
  for (int i = 1; i < num_rois; i++) {
    const Dtype* tmp_rois = bottom_rois - bottom[2]->offset(1);
    if (int(tmp_rois[0]) != int(bottom_rois[0])) {
      batch_id++;
      center_x = float(bottom_rois[1] + bottom_rois[3]) / (2.0 * float(bottom_shape[batch_id * 2]));
      center_y = float(bottom_rois[2] + bottom_rois[4]) / (2.0 * float(bottom_shape[batch_id * 2 + 1]));
      for (int j = 0; j < num_grid; j++) {
        if (center_x >= spm[j * 4] && center_x < spm[j * 4 + 1]
            && center_y >= spm[j * 4 + 2] && center_y < spm[j * 4 + 3]) {
          for (int c = 0; c < channels; c++) {
            top_data[batch_id * num_grid * channels + j * channels + c] = 
              bottom_data[i * channels + c];
            real_max_idx[batch_id * num_grid * channels + j * channels + c] = i;
          }
        }
        else {
          for (int c = 0; c < channels; c++) {
            top_data[batch_id * num_grid * channels + j * channels + c] = -1000000;
            real_max_idx[batch_id * num_grid * channels + j * channels + c] = -1;
          }
        }
      }
    }
    else {
      center_x = float(bottom_rois[1] + bottom_rois[3]) / (2.0 * float(bottom_shape[batch_id * 2]));
      center_y = float(bottom_rois[2] + bottom_rois[4]) / (2.0 * float(bottom_shape[batch_id * 2 + 1]));
      for (int j = 0; j < num_grid; j++) {
        if (center_x >= spm[j * 4] && center_x < spm[j * 4 + 1]
            && center_y >= spm[j * 4 + 2] && center_y < spm[j * 4 + 3]) {
          for (int c = 0; c < channels; c++) {
            if (bottom_data[i * channels + c] > 
                top_data[batch_id * num_grid * channels + j * channels + c]) {
              top_data[batch_id * num_grid * channels + j * channels + c] = 
                bottom_data[i * channels + c];
              real_max_idx[batch_id * num_grid * channels + j * channels + c] = i;
            }
          }
        }
      }
    }
    bottom_rois += bottom[2]->offset(1);
  }
  
  for (int i = 0; i < batch_size * num_grid * channels; i++) {
    if (-1 == real_max_idx[i]) {
      top_data[i] = 0;
    }
  }
}

template <typename Dtype>
void SPMMaxPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  // Propagate to bottom
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    int batch_size = bottom[1]->num();
    int num_rois = bottom[2]->num();
    const int channels = bottom[0]->channels();
    int* real_max_idx = max_idx_.mutable_cpu_data();
    
    for (int i = 0; i < num_rois; i++) {
      for (int c = 0; c < channels; c++) {
        bottom_diff[i * channels + c] = 0;
      }
    }

    for (int i = 0; i < batch_size; i++) {
      for (int j = 0; j < num_grid; j++) {
        for (int c = 0; c < channels; c++) {
          if (real_max_idx[i * num_grid * channels + j * channels + c] != -1) {
            bottom_diff[real_max_idx[i * num_grid * channels + j * channels + c] * channels + c] += 
              top_diff[i * num_grid * channels + j * channels + c];
          }
        }
      }
    }
    // delete [] real_max_idx;
  }
}


#ifdef CPU_ONLY
STUB_GPU(SPMMaxPoolingLayer);
#endif

INSTANTIATE_CLASS(SPMMaxPoolingLayer);
REGISTER_LAYER_CLASS(SPMMaxPooling);

}  // namespace caffe
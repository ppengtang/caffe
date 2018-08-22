#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/refine_loss_layer.hpp"

using std::max;

namespace caffe {

template <typename Dtype>
__global__ void RefineLossForwardGPU(const int nthreads,
          const Dtype* prob_data, const Dtype* label, 
          const Dtype* weights, const Dtype* pc_label, 
          const Dtype* pc_prob, const Dtype* img_cls_loss_weights, 
          const Dtype* im_label, Dtype* loss, const int batch_size, 
          const int channels, const int num_positive) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    loss[index] = 0;
    const Dtype epsilon = 1e-12;
    if (im_label[index] != 0) {
      if (index == 0) {
        for (int i = 0; i < batch_size; i++) {
          if (label[i] == 0) {
            loss[index] -= weights[i] * log(max(prob_data[i * channels + index], epsilon));
          }
        }
      }
      else {
        for (int i = 0; i < num_positive; i++) {
          if (pc_label[i] == index) {
            loss[index] -= img_cls_loss_weights[i] * log(max(pc_prob[i], epsilon));
          }
        }
      }
    }
  }
}

template <typename Dtype>
void RefineLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* prob_data = bottom[0]->gpu_data();
  const Dtype* label = bottom[1]->gpu_data();
  const Dtype* cls_loss_weights = bottom[2]->gpu_data();
  const Dtype* gt_assigment = bottom[3]->gpu_data();
  const Dtype* pc_label = bottom[4]->gpu_data();
  const Dtype* pc_prob = bottom[5]->gpu_data();
  const Dtype* img_cls_loss_weights = bottom[7]->gpu_data();
  const Dtype* im_label = bottom[8]->gpu_data();
  // Since this memory is not used for anything until it is overwritten
  // on the backward pass, we use it here to avoid having to allocate new GPU
  // memory to accumulate intermediate results in the kernel.
  Dtype* loss_data = bottom[0]->mutable_gpu_diff();

  const int nthreads = channels;
  // NOLINT_NEXT_LINE(whitespace/operators)
  RefineLossForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, prob_data, label, cls_loss_weights, 
      pc_label, pc_prob, img_cls_loss_weights, im_label, loss_data, batch_size, 
      channels, num_positive);
  Dtype loss;
  caffe_gpu_asum(nthreads, loss_data, &loss);
  top[0]->mutable_cpu_data()[0] = loss / batch_size;
}

template <typename Dtype>
__global__ void RefineLossBackwardGPU(const int nthreads, const Dtype* prob_data, 
          const Dtype* label, const Dtype* weights, const Dtype* gt_assigment,
          const Dtype* pc_label, const Dtype* pc_prob, const Dtype* pc_count, 
          const Dtype* img_cls_loss_weights, const Dtype* im_label, 
          Dtype* bottom_diff, const int channels) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int i = index / channels;
    const int c = index % channels;
    bottom_diff[index] = 0;
    const Dtype epsilon = 1e-12;
    if (im_label[c] != 0) {
      if (c == 0) {
        if (label[i] == 0) {
          bottom_diff[index] = -weights[i] / max(prob_data[index], epsilon);
        }
      }
      else {
        if (label[i] == c) {
          const int pc_index = gt_assigment[i];
          if (c != pc_label[pc_index]) {
            printf("labels mismatch.\n");
          }
          bottom_diff[index] = -img_cls_loss_weights[pc_index]  
            / max(pc_count[pc_index] * pc_prob[pc_index], epsilon);
        }
      }
    }
  }
}

template <typename Dtype>
void RefineLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[2]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to weights inputs.";
  }
  if (propagate_down[3]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to gt_assigment inputs.";
  }
  if (propagate_down[4]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to pc_label inputs.";
  }
  if (propagate_down[5]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to pc_prob inputs.";
  }
  if (propagate_down[6]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to pc_count inputs.";
  }
  if (propagate_down[7]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to img_cls_loss_weights inputs.";
  }
  if (propagate_down[8]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to image label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* prob_data = bottom[0]->gpu_data();
    const Dtype* label = bottom[1]->gpu_data();
    const Dtype* cls_loss_weights = bottom[2]->gpu_data();
    const Dtype* gt_assigment = bottom[3]->gpu_data();
    const Dtype* pc_label = bottom[4]->gpu_data();
    const Dtype* pc_prob = bottom[5]->gpu_data();
    const Dtype* pc_count = bottom[6]->gpu_data();
    const Dtype* img_cls_loss_weights = bottom[7]->gpu_data();
    const Dtype* im_label = bottom[8]->gpu_data();
    // NOLINT_NEXT_LINE(whitespace/operators)

    const int nthreads = batch_size * channels;
    RefineLossBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, prob_data, label, cls_loss_weights, 
        gt_assigment, pc_label, pc_prob, pc_count, img_cls_loss_weights, im_label, 
        bottom_diff, channels);

    const Dtype loss_weight = top[0]->cpu_diff()[0] / batch_size;
    caffe_gpu_scal(bottom[0]->count(), loss_weight , bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(RefineLossLayer);

}  // namespace caffe

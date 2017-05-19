# Updates
Several layers are added for compatibility to deeplab-v2, etc.

- layer/seg_accuracy_layer (.cpp, .hpp)
- layer/interp layer (.cpp, .hpp)
- util/interp (.cpp, .cu)
- util/confusion_matrix (.cpp, .hpp)
- common.cuh: Add follows (for CUDA 8.0)
	
		#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
		#else
		// CUDA: atomicAdd is not defined for doubles
		static __inline__ __device__ double atomicAdd(double *address, double val) {
		  unsigned long long int* address_as_ull = (unsigned long long int*)address;
		  unsigned long long int old = *address_as_ull, assumed;
		  if (val==0.0)
		    return __longlong_as_double(old);
		  do {
		    assumed = old;
		    old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val +__longlong_as_double(assumed)));
		  } while (assumed != old);
		  return __longlong_as_double(old);
		}
		#endif
- layer/BN_layer (.cpp, .cu, .hpp)
- layer/image_seg_data_layer (.cpp, .hpp)
- Updates: utils/io, data transformer, layer/base data layer, Makefile (Add ````CXXFLAGS += --std=c++11```` in Line 12)

- Common things: PREFETCH_COUNT -> prefetch_.size() 

- Enhanced Image Seg Data Layer with data augmentation, which is based on:
	[@twtygqyy](https://github.com/twtygqyy)'s [caffe-augmentation](https://github.com/twtygqyy/caffe-augmentation).

	- Usage:

		In training phase,

			layer {
				name: "data"
				type: "ImageSegData"
				top: "data"
				top: "mask"
				include {
				  phase: TRAIN
				}
				transform_param {
				    mirror: true
					crop_size: 224

					mean_value: 104.008
					mean_value: 116.669
					mean_value: 122.675
					scale_factors: 0.5
					scale_factors: 1.0
					scale_factors: 1.5
					scale_factors: 2.0

				    contrast_brightness_adjustment: true
				    smooth_filtering: true
				    min_side_min: 256
				    min_side_max: 480				    				    
				    min_contrast: 0.8
				    max_contrast: 1.2
				    max_smooth: 6
				    apply_probability: 0.5
				    max_color_shift: 20
				    debug_params: false
				}
				image_data_param {
				  source: "train_list.txt"
				  batch_size: 64
				}
			}

 		In testing phase,

			layer {
				name: "data"
				type: "ImageData"
				top: "data"
				top: "label"
				include {
				  phase: TEST
				}
				transform_param {
				    mirror: false
				    min_side: 256
				    crop_size: 224
				    mean_value: 104.008
					mean_value: 116.669
					mean_value: 122.675
				}
				image_data_param {
				  source: "test_list.txt"
				  batch_size: 32
				}
			}


---
# Caffe

[![Build Status](https://travis-ci.org/BVLC/caffe.svg?branch=master)](https://travis-ci.org/BVLC/caffe)
[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

Caffe is a deep learning framework made with expression, speed, and modularity in mind.
It is developed by Berkeley AI Research ([BAIR](http://bair.berkeley.edu))/The Berkeley Vision and Learning Center (BVLC) and community contributors.

Check out the [project site](http://caffe.berkeleyvision.org) for all the details like

- [DIY Deep Learning for Vision with Caffe](https://docs.google.com/presentation/d/1UeKXVgRvvxg9OUdh_UiC5G71UMscNPlvArsWER41PsU/edit#slide=id.p)
- [Tutorial Documentation](http://caffe.berkeleyvision.org/tutorial/)
- [BAIR reference models](http://caffe.berkeleyvision.org/model_zoo.html) and the [community model zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)
- [Installation instructions](http://caffe.berkeleyvision.org/installation.html)

and step-by-step examples.

[![Join the chat at https://gitter.im/BVLC/caffe](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/BVLC/caffe?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Please join the [caffe-users group](https://groups.google.com/forum/#!forum/caffe-users) or [gitter chat](https://gitter.im/BVLC/caffe) to ask questions and talk about methods and models.
Framework development discussions and thorough bug reports are collected on [Issues](https://github.com/BVLC/caffe/issues).

Happy brewing!

## License and Citation

Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
The BAIR/BVLC reference models are released for unrestricted use.

Please cite Caffe in your publications if it helps your research:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }

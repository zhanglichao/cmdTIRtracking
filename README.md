## Unsupervised Cross-Modal Distillation for Thermal Infrared Tracking [[paper]](https://arxiv.org/pdf/2108.00187.pdf)

Jingxian Sun<sup>\*</sup>, Lichao Zhang<sup>\*</sup>, Yufei Zha, Abel Gonzalez-Garcia, Peng Zhang, Wei Huang and Yanning Zhang

*ACM International Conference on Multimedia (ACM MM), 2021*

## Citation
Please cite our paper if you are inspired by this idea (come soon...).

## Instructions
We propose to distill the representation of the TIR modality from the RGB modality with Cross-Modal Distillation (CMD) on a large amount of unlabeled paired RGB-T data. We take advantage of the two-branch architecture of the baseline tracker, i.e. DiMP, for cross-modal distillation working on two components of the tracker. Specifically, we use one branch as a teacher module to distill the representation learned by the model into the other branch. Benefiting from the powerful model in the RGB modality, the cross-modal distillation can learn the TIR-specific representation for promoting TIR tracking. The proposed approach can be incorporated into different baseline trackers conveniently as a generic and independent component. Furthermore, the semantic coherence of paired RGB and TIR images is utilized as a supervised signal in the distillation loss for model knowledge transfer. In practice, three different approaches are explored to generate paired RGB-T patches with the same semantics for training in an unsupervised way. It is easy to extend to an even larger scale of unlabeled training data. Extensive experiments on the LSOTB-TIR dataset and PTB-TIR dataset demonstrate that our proposed cross-modal distillation method effectively learns TIR-specific target representations transferred from the RGB modality. Our tracker is trained in an end-to-end manner. Our tracker outperforms the baseline tracker by achieving an absolute gain of 2.3% Success Rate, 2.7% Precision, and 2.5% Norm Precision respectively.

## Pre-trained models and the annotated Data
The pre-trained CMD model and the annotated bounding boxes for the 'detector' can be downloaded in:

[baiduyun](https://pan.baidu.com/s/1xqzuFHuk532rjenQohpL7g) with the password 7cl7.

The two detectors are [FairMOT](https://github.com/ifzhang/FairMOT) and [yoloV5](https://github.com/ultralytics/yoloV5).
## Contact
Please contact zhanglichao@outlook.com for the questions in the repository.

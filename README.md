# NAR-Former: Neural Architecture Representation Learning towards Holistic Attributes Prediction

This is the source code for paper:<br> 
[NAR-Former: Neural Architecture Representation Learning towards Holistic Attributes Prediction](https://arxiv.org/pdf/2211.08024.pdf)
![image](https://github.com/yuny220/NAR-Former/blob/master/figures/overall.png)


## Experiments on NAS-Bench-201
We introduce the experimental process using the NAS-Bench-201 dataset (5% training setting) as an example. The experiments on NAS-Bench-101 are similar to this.

### Data preprocessing with proposed tokenizer
You can directly download [datas/nasbench201](https://pan.baidu.com/s/1FupUMFX7hf5nvOBMacMwZA?pwd=qr1v) and put it in `./data/` or generate by yourself following the steps below:
1. Download [NAS-Bench-201-v1_0-e61699.pth](https://drive.google.com/file/d/1SKW0Cu0u8-gb18zDpaAGi0f74UdXeGKs/view?pli=1) and put it in `./data/nasbench201/`.
```
python preprocessing/gen_json_201.py
```
The generated file `cifar10_valid_converged.json` will be saved in `./data/nasbench201/`.


2. Encode each architecture with our proposed tokenizer.
```
python data_and_encoding_generate.py --dataset nasbench201 --data_path data/nasbench201/cifar10_valid_converged.json --save_dir data/nasbench201/
```
The generated file `all_nasbench201.pt` will be saved in `./data/nasbench201/`.

3. (Results in the paper.) If you want to use information flow consistency augmentation, run the following code to generate the augmented data file.
```
python ac_aug_generate.py --dataset nasbench201 --data_path data/nasbench201/all_nasbench201.pt 
```
The file of augmented data will be saved in `./data/nasbench201/`.

### Train NAR-Former
You can directly download [pretrained_modes/nasbench201/checkpoints_5%_aug](https://pan.baidu.com/s/1FupUMFX7hf5nvOBMacMwZA?pwd=qr1v) and put it in `./experiment/Accuracy_Predict_NASBench201/` or train from scratch following the steps below:
1. Change the  `BASE_DIR` in script files in folder `experiment/Accuracy_Predict_NASBench201/` to the current absolute path.

2. For model training, you can choose to use augmented data or not.
- Without augmented data:
```
cd experiment/Accuracy_Predict_NASBench201/
bash train_5%.sh
```
The pretrained models will be saved in `./checkpoints_5%/`

- (Results in the paper.) With augmented data:
```
cd experiment/Accuracy_Predict_NASBench201/
bash train_5%_aug.sh
```
The pretrained models will be saved in `./checkpoints_5%_aug/`

### Evaluate the pretrained model
- For models trained without augmented data:
```
bash test_5%.sh
```

- (Results in the paper.) For models trained with augmented data:
```
bash test_5%_aug.sh
```

## Citation
If you find our codes or trained models useful in your research, please consider to star our repo and cite our paper:
```
@article{yi2022nar,
  title={NAR-Former: Neural Architecture Representation Learning towards Holistic Attributes Prediction},
  author={Yi, Yun and Zhang, Haokui and Hu, Wenze and Wang, Nannan and Wang, Xiaoyu},
  journal={arXiv preprint arXiv:2211.08024},
  year={2022}
}
```

## Acknowledge
1. [NAS-Bench-101](https://github.com/google-research/nasbench)
2. [NAS-Bench-201](https://github.com/D-X-Y/NAS-Bench-201)
3. [NNLQP](https://github.com/auroua/NPENASv1)

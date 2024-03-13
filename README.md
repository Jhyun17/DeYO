# Entropy is not Enough for Test-Time Adaptation: From the Perspective of Disentangled Factors
![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)  

This is the official implementation of [Entropy is not Enough for Test-Time Adaptation: From the Perspective of Disentangled Factors 🔗](https://openreview.net/forum?id=9w3iw8wDuE) 
by Jonghyun Lee, Dahuin Jung, Saehyung Lee, Junsung Park, Juhyeon Shin, Uiwon Hwang and Sungroh Yoon (**ICLR 2024 Spotlight, Top-5% of the submissions**).  
This implementation is based on [SAR implementation 🔗](https://github.com/mr-eggplant/SAR).

## Environments  

You should modify [username] and [env_name] in environment.yaml, then  
> $ conda env create --file environment.yaml  

## Baselines  
[TENT 🔗](https://arxiv.org/abs/2006.10726) (ICLR 2021)  
[EATA 🔗](https://arxiv.org/abs/2204.02610) (ICML 2022)  
[SAR 🔗](https://arxiv.org/abs/2302.12400) (ICLR 2023)  

## Dataset
You can download ImageNet-C from a link [ImageNet-C 🔗](https://zenodo.org/record/2235448).  

After downloading the dataset, move to the root directory ([data_root]) of datasets.  

If you run on [ColoredMNIST 🔗](https://arxiv.org/abs/1907.02893) or [Waterbirds 🔗](https://arxiv.org/abs/1911.08731), run  
> $ python pretrain_[dataset_name].py --root_dir [data_root] --dset [dataset_name]

Then datasets are automatically downloaded in your [data_root] directory.  
(ColoredMNIST from [torchvision 🔗](https://pytorch.org/vision/stable/index.html) and ./dataset/ColoredMNIST_dataset.py, Waterbirds from [wilds 🔗](https://pypi.org/project/wilds/) package)

Your [data_root] will be as follows:
```bash
data_root
├── ImageNet-C
│   ├── brightness
│   ├── contrast
│   └── ...
├── ColoredMNIST
│   ├── ColoredMNIST_model.pickle
│   ├── MNIST
│   ├── train1.pt
│   ├── train2.pt
│   └── test.pt
├── Waterbirds
│   ├── metadata.csv
│   ├── waterbirds_dataset.h5py
│   ├── waterbirds_pretrained_model.pickle
│   ├── 001. Black_footed_Albatross
│   ├── 002. Laysan_Albatross
└── └── ...
```
If you don't want to pre-train, you can just copy and paste the [dataset_name]_model.pickle from './pretrained/' directory.

## Experiment

You can run most of the experiments in our paper by  
> $ chmod +x exp_deyo.sh  
> $ ./exp_deyo.sh  

If you want to run on the ImageNet-R or VISDA-2021, you should use main_da.py

You should modify ROOT variable as [data_root] in exp_deyo.sh.  

## Citation
If our DeYO method or biased test-time adaptation settings are helpful in your research, please consider citing our paper:
```
@inproceedings{
    lee2024entropy,
    title={Entropy is not Enough for Test-Time Adaptation: From the Perspective of Disentangled Factors},
    author={Jonghyun Lee and Dahuin Jung and Saehyung Lee and Junsung Park and Juhyeon Shin and Uiwon Hwang and Sungroh Yoon},
    booktitle={The Twelfth International Conference on Learning Representations},
    year={2024},
    url={https://openreview.net/forum?id=9w3iw8wDuE}
}
```

## Acknowledgment
The code is inspired by the [Tent 🔗](https://github.com/DequanWang/tent), [EATA 🔗](https://github.com/mr-eggplant/EATA), and [SAR 🔗](https://github.com/mr-eggplant/SAR).

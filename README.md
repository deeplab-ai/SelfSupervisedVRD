# Self-Supervised Learning for Visual Relationship Detection through Masked Bounding Box Reconstruction

**[deeplab.ai](https://deeplab.ai/)**

**Zacharias Anastasakis, Dimitrios Mallis, Markos Diomataris, George Alexandridis, Stefanos Kollias, Vassilis Pitsikalis**

We propose Masked Bounding Box Reconstruction, a variation of Masked Image Modeling where a percentage 
of the entities/objects within a scene are masked and subsequently reconstructed based on the unmasked objects. 
Through object-level masked modeling, our proposed network learns context-aware representations 
that capture the interaction of objects within a scene and are highly predictive of visual object relationships.

This repository contains the code for reproducing our IEEE/CVF Winter Conference on Applications of Computer Vision 2024 paper and is based on the [grounding-consistent-vrd](https://github.com/deeplab-ai/grounding-consistent-vrd). You can find our paper [here](https://openaccess.thecvf.com/content/WACV2024/papers/Anastasakis_Self-Supervised_Learning_for_Visual_Relationship_Detection_Through_Masked_Bounding_Box_WACV_2024_paper.pdf).

![](https://github.com/deeplab-ai/SelfSupervisedVRD/blob/main/images/pre-training.jpg?raw=true)

## Environment Setup
After cloning this repository, you can set up a conda environment using the *mbbr.yml* config file:
```bash
conda env create -f mbbr.yml
conda activate mbbr
```

## Dataset Setup
You can download the VRD and/or VG200 dataset by running the main_prerequisites python file. You can define 
the dataset as an argument:
```python
python3 main_prerequisites.py VG200
```

## Train
Training involves 2 steps:
1. Pre-train a transformer network in a self-supervised manner through Masked Bounding Box Reconstruction (MBBR)
```python
python3 main_research.py --model=MBBR --net_name=MBBRNetwork --projection_head --dataset=VG200 --pretrain_arch=encoder
```

2. Train an MLP network in a few-shot setting on random samples, using the pre-trained network from the previous step:
```python
python3 main_research.py --model=SSL_finetune --net_name=FinetunedNetwork --dataset=VG200 --pretrain_arch=encoder --random_few_shot=10 --random_seed=4 --pretrained_model=MBBRNetwork --projection_head --normal --pretrain_task=reconstruction
```
The above command trains a 2-layer MLP network on 5 random samples from the VRD dataset. However, in our work we also manually selected {1,2,5} accurate relationships per Predicate Category and used them to train our classifier.
These relationships are given in the **prerequisites/{VG200/VRD}_few_shot_dict.json** files. You can train a classifier on these
manually-selected samples by running the following command:

```python
python3 main_research.py --model=SSL_finetune --net_name=FinetunedNetwork --dataset=VG200 --pretrain_arch=encoder --few_shot=5 --pretrained_model=MBBRNetwork --projection_head --normal --pretrain_task=reconstruction
```

## Test
After training, testing is automatically performed and micro/macro Recal@[20, 50, 100] is printed for both constrained and unconstrained scenarios while also calculating zero-shot results.

Checkpointing is performed so re-running step 2 for an already trained model will simply perform testing.

## Bibtex
```bash
@InProceedings{Anastasakis_2024_WACV,
    author    = {Anastasakis, Zacharias and Mallis, Dimitrios and Diomataris, Markos and Alexandridis, George and Kollias, Stefanos and Pitsikalis, Vassilis},
    title     = {Self-Supervised Learning for Visual Relationship Detection Through Masked Bounding Box Reconstruction},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2024},
    pages     = {1206-1215}
}
```

#### Feel free to contact us for any issues!!

# Rebalanced Vision-Language Retrieval Considering Structure-Aware Distillation

This is the official PyTorch implementation of the paper "**Rebalanced Vision-Language Retrieval Considering Structure-Aware Distillation**". 

💥 We observe that existing vision-language retrieval methods struggle with **modality imbalance**, where precisely learning cross-modal consistency can disrupt instance structures in the shared space, adversely affecting single-modal retrieval. To address this, we propose a **multi-granularity distillation module that incorporates representation-level distillation and structure-aware distillation for single modalities** on top of the cross-modal matching loss, ensuring semantic and structural consistency among instances.

[![pAR55Xd.png](https://s21.ax1x.com/2024/11/18/pAR55Xd.png)](https://imgse.com/i/pAR55Xd)

### Requirements

- torch1.8.1+cu111

- transformers 4.28.1

  For more detailed environment information, please refer to **environment.yml**.

### Dataset Preparation

- Download [MS-COCO 2014](https://cocodataset.org/#download) / [FLICKR30K](https://shannon.cs.illinois.edu/DenotationGraph/data/index.html) /  [Vizwiz](https://vizwiz.org/tasks-and-datasets/vqa/) dataset and set your `image_root` in "**configs/Retrieval_{dataname}.yaml**".
- Due to file size limitations for uploads, you can download the `coco_train.json` file [here](https://drive.google.com/file/d/1Bq2dkwxavPFxl9ZYpV0oWZefaTS1Nl2j/view?usp=sharing) and place it in the "**data/**".

### Download

- Download the pre-trained models for [SWIN](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth) and [BERT](https://hf-mirror.com/google-bert/bert-base-uncased/tree/main), and place them in the "**data/**".
- Use the [ROUGE-L similarity](https://drive.google.com/drive/folders/1Vev2wCJPuMvLTs0spKAeviV2ngGiXytw?usp=sharing) between data within each dataset as ground truth to evaluate retrieval performance.
- We pre-trained the model using vision-language contrastive learning (ITC) and vision-language matching (ITM) for downstream retrieval tasks. The checkpoint can be downloaded [here](https://drive.google.com/drive/folders/1HSawwl7-wk7IbqcVyUC8vMQtpR-XRoPQ?usp=sharing). For example, "**output/flickr/pretrain/checkpoint_best.pth**".
- We have saved the [features](https://drive.google.com/drive/folders/1OR-ywvySlK6HUKcUwFviqWwsqG2KewMG?usp=sharing) output by the best teacher model as `train_image_embed_regu.npy` and `train_text_embed_regu.npy`. After downloading them, modify the file paths in `dataset/re_dataset.py` accordingly.
- The checkpoint for our method can be found [here](https://drive.google.com/drive/folders/1HSawwl7-wk7IbqcVyUC8vMQtpR-XRoPQ?usp=sharing). For example, "**output/flickr/best/checkpoint_best.pth**".

### Train

```bash
nohup sh train_coco.sh > logs/coco.txt 2>&1 &
nohup sh train_flickr.sh > logs/flickr.txt 2>&1 &
nohup sh train_vizwiz.sh > logs/vizwiz.txt 2>&1 &
```

If you only want to use ITC and ITM for pre-training, set `load_pretrain` to **`False`** in `Retrieval.py`.

### Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 Retrieval.py --output_dir=$output_dir --dataname=$dataname --distance=$distance --auto=$auto --lamda=$lamda --bs=$train_bs --dist_url=$dist_url --temp=$temp --temp2=$temp2 --evaluate
```

### Results

- Cross-modal retrieval

  [![pAR5onA.png](https://s21.ax1x.com/2024/11/18/pAR5onA.png)](https://imgse.com/i/pAR5onA)

- Single-modal retrieval

  [![pAR577t.png](https://s21.ax1x.com/2024/11/18/pAR577t.png)](https://imgse.com/i/pAR577t)

- Mixed retrieval

  [![pAR5bAP.png](https://s21.ax1x.com/2024/11/18/pAR5bAP.png)](https://imgse.com/i/pAR5bAP)

### Citation

If you find this code to be useful for your research, please consider citing.

```
@article{RVLR,
  title={Rebalanced Vision-Language Retrieval Considering Structure-Aware Distillation},
  author={Yang Yang, Wenjuan Xi, Luping Zhou, and Jinhui Tang},
  journal={IEEE Transactions on Image Processing},
  year={2024}
}
```
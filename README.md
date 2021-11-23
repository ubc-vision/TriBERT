# TriBERT

This repository contains the code for the NeurIPS 2021 paper titled ["TriBERT: Full-body Human-centric Audio-visual Representation Learning for Visual Sound Separation"](https://arxiv.org/pdf/2110.13412.pdf).

# Data pre-processing:

Please download [MUSIC21](https://github.com/roudimit/MUSIC_dataset). we found 314 videos are missing. Moreover, the train/val/test split was unavailable. Therefore, we used a random 80/20 train/test split which is given in [data](https://github.com/ubc-vision/TriBERT/tree/master/data). 

After downloading the dataset, please consider following steps as data pre-processing.

1. Following [Sound-of-Pixels](https://github.com/hangzhaomit/Sound-of-Pixels) we extracted video frames at 8fps and waveforms at 11025Hz from videos. We considered these frames and waveforms as our visual and audio input for TriBERT model.
2. Setup [AlphaPose toolbox](https://github.com/MVIG-SJTU/AlphaPose) to detect 26 keypoints for body joints and 21 keypoints for each hand.
3. Re-train [ST-GCN network](https://github.com/yysijie/st-gcn) with the keypoints detected using AlphaPose and extract body joint features of size 256 × 68. These features will be considered as pose embedding to pose stream of TriBERT model. 

# Pre-trained model

Please download our pre-trained model from [Google Drive](https://drive.google.com/file/d/1cOIEUzcp7tKO1C6OyXwso2Rrm0wZHuu2/view?usp=sharing).

# Acknowledgment

This repository is developed on top of [ViLBERT](https://github.com/jiasenlu/vilbert_beta) and [Sound-of-Pixels](https://github.com/hangzhaomit/Sound-of-Pixels). Please also refer to the original License of these projects.

# Bibtext

If you find this code is useful for your research, please cite our paper


```
@inproceedings{rahman2021tribert,
  title={TriBERT: Human-centric Audio-visual Representation Learning},
  author={Rahman, Tanzila and Yang, Mengyu and Sigal, Leonid},
  booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
  year={2021}
}
```

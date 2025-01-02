# FaceShot: Bring Any Character into Life
[**FaceShot: Bring Any Character into Life**](https://arxiv.org/abs/)

[Junyao Gao](https://jeoyal.github.io/home/), [Yanan Sun](https://scholar.google.com/citations?hl=zh-CN&user=6TA1oPkAAAAJ)<sup>&Dagger; *</sup>, [Fei Shen](https://muzishen.github.io/), [Xin Jiang](https://whitejiang.github.io/), [Zhening Xing](https://scholar.google.com/citations?user=sVYO0GYAAAAJ&hl=en), [Kai Chen*](https://chenkai.site/), [Cairong Zhao*](https://vill-lab.github.io/)

(* corresponding authors, <sup>&Dagger;</sup> project leader)

 <a href='https://arxiv.org/abs/2407.01414'><img src='https://img.shields.io/badge/arXiv-2407.01414-b31b1b.svg'></a> 
 <a href='https://styleshot.github.io/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> 

Bringing characters like Teddy Bear into life requires a bit of *magic*. **FaceShot** makes this *magic* a reality by introducing a training-free portrait animation framework which can animate
any character from any driven video, especially for non-human characters, such as emojis and toys.

**Your star is our fuel!  We're revving up the engines with it!**

<img src="__assets__/teaser.gif">

## News
- [2025/1/23] 🔥 We release the code, [project page](https://styleshot.github.io/) and [paper](https://arxiv.org/abs/2407.01414).

## TODO List
- [ ] Preprocessing script for pre-store target images and appearance gallery.
- [ ] Appearance gallery.
- [ ] Gradio demo.

## Gallery
<div align="center">
  <h3>
    Bring Any Character into Life!!!
  </h3>
</div>

<table align="center">
  <tr>
    <td align="center">
      <img src="__assets__/toy.gif"/>
      <br />
    </td>
  </tr>
  <tr>
    <td colspan="3" align="center" style="border: none;">
      Toy Character
    </td>
  </tr>
</table>

<table align="center">
  <tr>
    <td align="center">
      <img src="__assets__/2danime.gif"/>
      <br />
    </td>
  </tr>
  <tr>
    <td colspan="3" align="center" style="border: none;">
      2D Anime Character
    </td>
  </tr>
</table>

<table align="center">
  <tr>
    <td align="center">
      <img src="__assets__/3danime.gif"/>
      <br />
    </td>
  </tr>
  <tr>
    <td colspan="3" align="center" style="border: none;">
      3D Anime Character
    </td>
  </tr>
</table>

<table align="center">
  <tr>
    <td align="center">
      <img src="__assets__/animal.gif"/>
      <br />
    </td>
  </tr>
  <tr>
    <td colspan="3" align="center" style="border: none;">
      Animal Character
    </td>
  </tr>
</table>

<div align="center">
Check the gallery of our <a href='https://styleshot.github.io/' target='_blank'>project page</a> for more visual results!
</div>

## Get Started
### Clone the Repository

```
git clone https://github.com/Jeoyal/FaceShot.git
cd ./FaceShot
```

### Environment Setup

This script has been tested on CUDA version of 12.4.

```
conda create -n faceshot python==3.10
conda activate faceshot
pip install -r requirements.txt
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
pip install "git+https://github.com/XPixelGroup/BasicSR.git"

```

#### Downloading Checkpoints

1. Download the checkpoint of CMP from [here](https://huggingface.co/MyNiuuu/MOFA-Video-Hybrid/resolve/main/models/cmp/experiments/semiauto_annot/resnet50_vip%2Bmpii_liteflow/checkpoints/ckpt_iter_42000.pth.tar) and put it into `./models/cmp/experiments/semiauto_annot/resnet50_vip+mpii_liteflow/checkpoints`.

2. Download the `ckpts` [folder](https://huggingface.co/MyNiuuu/MOFA-Video-Hybrid/tree/main/ckpts) from the huggingface repo which contains necessary pretrained checkpoints and put it under `./ckpts`. You may use `git lfs` to download the **entire** `ckpts` folder.

    

#### Running Inference Scripts

```
chmod 777 inference.sh
./inference.sh
```

## License and Citation
All assets and code are under the [license](./LICENSE) unless specified otherwise.

If this work is helpful for your research, please consider citing the following BibTeX entry.
```
@article{gao2024faceshot,
  title={FaceShot: Bring Any Character into Life},
  author={Gao, Junyao and Sun, Yanan and Shen, Fei and Xin, Jiang and Xing, Zhening and Chen, Kai and Zhao, Cairong},
  journal={arXiv preprint},
  year={2025}
}
```

## Acknowledgements
The code is built upon [MOFA-Video](https://github.com/MyNiuuu/MOFA-Video) and [DIFT](https://github.com/Tsingularity/dift).

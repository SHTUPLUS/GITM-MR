# Grounded Image Text Matching with Mismatched Relation Reasoning

This repository contains the official Python implementation for the ICCV 2023 paper 8957 **Grounded Image Text Matching with Mismatched Relation Reasoning**.

\[[__project page__](https://weiyana.github.io/pages/dataset.html)\] \[[__paper__](https://openaccess.thecvf.com/content/ICCV2023/papers/Wu_Grounded_Image_Text_Matching_with_Mismatched_Relation_Reasoning_ICCV_2023_paper.pdf)\] \[[__supp__](https://openaccess.thecvf.com/content/ICCV2023/supplemental/Wu_Grounded_Image_Text_ICCV_2023_supplemental.pdf)\] \[[__preprint__](https://arxiv.org/abs/2308.01236)\] \[[__video__](https://youtu.be/eHXm2LrSSqE)\] 

## Abstract

> This paper introduces Grounded Image Text Matching with Mismatched Relation (GITM-MR), a novel visual-linguistic joint task that evaluates the relation understanding capabilities of transformer-based pre-trained models. GITM-MR requires a model to first determine if an expression describes an image, then localize referred objects or ground the mismatched parts of the text. We provide a benchmark for evaluating vision-language (VL) models on this task, with a focus on the challenging settings of limited training data and out-of-distribution sentence lengths. Our evaluation demonstrates that pre-trained VL models often lack data efficiency and length generalization ability. To address this, we propose the Relation-sensitive Correspondence Reasoning Network (RCRN), which incorporates relation-aware reasoning via bi-directional message propagation guided by language structure. Our RCRN can be interpreted as a modular program and delivers strong performance in terms of both length generalization and data efficiency. 

## GITM-MR Benchmark

We appreciate the contribution [Ref-Reasoning](https://sibeiyang.github.io/dataset/ref-reasoning/) [1] dataset, which our benchmark is constructed on.  Explore our benchmark from [the link](https://shanghaitecheducn-my.sharepoint.com/:f:/g/personal/wuyu1_shanghaitech_edu_cn/EjLw8i-lNQtEjEvyh8C6224B_op5kryv2ifVIlztQ7WYNw?e=nKfnMO). The structure and detail of the data directory is shown as follows:

```shell
└─data
    ├─counter        # The correspondence from the original expressions to mismatch ones.
    ├─expression     # Referring expression annotation files.
    ├─parse          # Parsed language scene graphs.
    ├─small          # Training subset annotations.
    ├─uniter         # UNITER checkpoints and BERT tokenizer.
    ├─vinvl_objects  # Detected boxes and features in h5 format.
    ├─word2token     # Word to UNITER token indices used in representation extraction.
```

The annotated images are GQA [2] images and can be downloaded from [the official website](https://cs.stanford.edu/people/dorarad/gqa/download.html), but our model doesn't necessitate the original images as input. Feel free to explore them based on your requirements. 

## Prerequisites and Installation

Our implementation is based on [*Detectron2*](https://github.com/facebookresearch/detectron2) framework. You need to prepare the required packages and build the local Detectron2 from the repository. Refer to the [*Common Installation Issues*](https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md#common-installation-issues) section in the installation manual in Detectron2 might be helpful to debug the process.

1. Prerequisites

   ```shell
   conda create -n gitm python=3.7
   pip install -r requirements.txt
   ```

2. Installation

   ```shell
   python setup.py build develop
   ```

## Reproduce the RCRN Result

1. Download the model checkpoint from [the link](https://shanghaitecheducn-my.sharepoint.com/:f:/g/personal/wuyu1_shanghaitech_edu_cn/EqN7UKU80rJLjPunKSBwRhoBUrF2cTsqgBfjtIPiCZj74w?e=ICBMKm) and put them into `ckpt` directory.

2. Download all the dataset into `data` directory. The expected directory structure should be similar to: 

   ```shell
   └─GITM-MR
       ├─data
       ├─ckpt
       ├─configs
       ├─detectron2
       ├─scripts
       ├─tools
   ```

3. Run the evaluation process by:

   ```shell
   python tools/train_refdet.py --num-gpus $num_gpu --config-file configs/{RCRN_len16.yaml, RCRN_len11.yaml} --config configs/train-ng-base-1gpu.json --eval-only --resume OUTPUT_DIR $output_dir
   ```

   Specify your paralleled GPU number in `$num_gpu` and the output directory in  `$output_dir`. Refer to `scripts` directory for the example.

4. If necessary, refer to `detectron2/modeling/refdet_heads/RCRN.py` file to explore our model implementation.

## References

[1] Sibei Yang, Guanbin Li, and Yizhou Yu. Graph-structured referring expression reasoning in the wild. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9952–9961, 2020. 

[2] Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6700–6709, 2019.  

## Citing GITM-MR

If you find our work useful for your research, please consider citing us:

```bibtex
@InProceedings{Wu_2023_ICCV,
    author    = {Wu, Yu and Wei, Yana and Wang, Haozhe and Liu, Yongfei and Yang, Sibei and He, Xuming},
    title     = {Grounded Image Text Matching with Mismatched Relation Reasoning},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {2976-2987}
}
```

## Contact

Please feel free to contact us at wuyu1@shanghaitech.edu.cn or weiyn1@shanghaitech.edu.cn if you have further questions or comments.

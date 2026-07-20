---
license: apache-2.0
tags:
- image
- video
inference: false
---
# LVM

This is the model implementation of the CVPR 2024 'Sequential Modeling Enables Scalable Learning for Large Vision Models'. (https://arxiv.org/abs/2312.00785)

LVM is a vision pretraining model that converts various kinds of visual data into visual sentences and performs next-token prediction autoregressively. It is compatible with both GPU and TPU.

You can try out the demo [here](https://huggingface.co/spaces/Emma02/LVM).

LVM is built on top of [OpenLLaMA](https://github.com/openlm-research/open_llama) (an autoregressive model) and [OpenMuse](https://github.com/huggingface/open-muse) (a VQGAN that converts images into visual tokens).

This was trained in collaboration with HuggingFace. Thanks [Victor Sanh](https://huggingface.co/VictorSanh) for the support in this project.

## Key Differences from the Original Paper Version
1. We are currently releasing the 7B model (previously 3B). Additional model size variants will be available soon.
2. Deep filtering (including quality filters, deduplication, and known CSAM content removal) has been applied to the LAION dataset, reducing the dataset size from 1.5B to 1.2B images.

3. The tokenizer has been improved for better performance.

## License
LVM is licensed under the Apache 2.0 License.


## Citation
If you found LVM useful in your research or applications, please cite our work using the following BibTeX:
```bibtex
@article{bai2023sequential,
  title={Sequential modeling enables scalable learning for large vision models},
  author={Bai, Yutong and Geng, Xinyang and Mangalam, Karttikeya and Bar, Amir and Yuille, Alan and Darrell, Trevor and Malik, Jitendra and Efros, Alexei A},
  journal={arXiv preprint arXiv:2312.00785},
  year={2023}
}
```
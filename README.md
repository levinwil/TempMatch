# TempMatch
[![License](https://img.shields.io/badge/License-MIT-blue)](https://opensource.org/licenses/MIT)
`TempMatch` (**Temp**eratue **Match**) is a package for exploring and using the Semi-Supervised learning algorithm developed by [Clarifai](https://clarifai.com)

Some system/package requirements:
* **Python**: 3.6+
* **OS**: All major platforms (Linux, MacOS, Windows)
* **Dependencies**: numpy, torch, pillow, tqdm

## Results

### CIFAR10
| #Labels | 40 | 250 | 4000 | Full Dataset |
|:---:|:---:|:---:|:---:|:---:|
| FixMatch | 86.19 | 94.93 | 95.74 | 96.21 |
| This code | 91.45 | 95.46 | 95.98 | 96.21 |

## Citations
This algorithm is heavily inspired by FixMatch:
```
@article{sohn2020fixmatch,
    title={FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence},
    author={Kihyuk Sohn and David Berthelot and Chun-Liang Li and Zizhao Zhang and Nicholas Carlini and Ekin D. Cubuk and Alex Kurakin and Han Zhang and Colin Raffel},
    journal={arXiv preprint arXiv:2001.07685},
    year={2020},
}
```

The source code is heavily based on [https://github.com/kekmodel/FixMatch-pytorch](https://github.com/kekmodel/FixMatch-pytorch)

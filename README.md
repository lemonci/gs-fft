# Frequency-based View Selection in Gaussian Splatting Reconstruction
Monica M.Q. Li, Pierre-Yves Lajoie, and Giovanni Beltrame

This repository is code for the associated with the paper "Frequency-based View Selection in Gaussian Splatting Reconstruction", which can be found [here](https://arxiv.org/abs/2409.16470). Please install according to the following steps. We only tested the code on Ubuntu.

## Installation

Clone this repo with the flag `--recursive`.

Follow the steps in [3D-GS](https://github.com/graphdeco-inria/gaussian-splatting) to create a conda environment and install the packages in `environment.yml`, as well as the submodules `diff-gaussian-rasterization` and `simple-knn`. Please note that if you are using Ubuntu 22+, you need to downgrade gcc and g++ to 9 to compile the submodules.

Install [colmap](https://colmap.github.io/install.html#build-from-source) and [glomap](https://github.com/colmap/glomap).

Download the datasets and extract them in `./tandt_db/`. The structure of the dataset directory should be like:

```
|---tandt_db
|   |---tandt
|       |---train
|           |---sparse
|               |---0
|                   |---images.bin
|                   |---points3D.ply
|                   |---points3D.bin
|                   |---cameras.bin
|                   |---project.ini
|           |---images
|               |---000081.jpg
|               |---000103.jpg
|               |---000240.jpg
|               |---000157.jpg
 ...
|       |---truck
...
|   |---db
|       |---drjohnson
...
|       |---playroom
...
```

## Run
```shell
python3 gs-fft.py
```

### Parameters
Lines 322-330 contain the parameters for input and output paths, iterations to train the Gaussian models and the iterations to select the next-best-views.
Lines 95-100 contain the parameters the same as the ones in "Command Line Arguments for train.py" in [3D-GS](https://github.com/graphdeco-inria/gaussian-splatting).

## Citation
<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">BibTeX</h2>
    <pre><code>@article{li2024frequency,
  title={Frequency-based View Selection in Gaussian Splatting Reconstruction},
  author={Li, Monica MQ and Lajoie, Pierre-Yves and Beltrame, Giovanni},
  journal={arXiv preprint arXiv:2409.16470},
  year={2024}
}
</code></pre>
  </div>
</section>

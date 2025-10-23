
## Overview

This repository includes the captured data and spectral brdf implementations of the SIGGRAPH Asia 2025 paper **Spectral Reconstruction with Uncertainty Quantification via Differentiable Rendering and Null-Space Sampling**, by Mengqi (Mandy) Xia, Bai Xue, Rachel Liang, and Holly Rushmeier.

Spectral information is vital across many fields but often costly to capture. We present a method to recover spectra from multispectral images using differentiable rendering and null-space sampling. This approach quantifies uncertainty, incorporates interreflections, accelerates reconstruction, and enables spectral material authoring, offering a practical alternative to hyperspectral imaging.

More information about this project can be found at the [project website](https://mandyxmq.github.io/research/hir.html).


## Data

The `data` directory contains the ground truth spectra of the physical Cornell box used in this paper.

`readdata.ipynb` provides the code for loading these data.

The `images` directory contains the captured photos.

The `scenes` directory contains the Mitsuba 3 scene files for rendering these physical experiments.


## Code

The `brdf` directory contains spectral BRDF implementations, using either a discrete set of values (`spectralDiffuse.py`) and a B-spline representation (`SmoothBsplineDiffuse.py`). 

The `mitsuba` folder contains the modifed `spectrum.h` and `spectrum.cpp` files for the physical experiments. In this work, we set the wavelength range to 400–700 nm and fitted sensitivity functions using the Macbeth chart. Please update these two files in Mitsuba 3, recompile, and set your Mitsuba path in `bspline.py` and `SpectralDiffuse.py`.

## Contact

If you have any questions regarding this work, please contact Mandy Xia (mengqi.xia AT yale.edu).








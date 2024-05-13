# KoVAE
[Koopman VAE (KoVAE)][https://openreview.net/pdf?id=eY7sLb0dVF], a new generative framework that is based on a novel design for the model prior.

<div align=center><img src="fig/arch_fig.png" width="100%"></div>

## Training

First, install the environment from the yaml file given here: environment.yml

In the repository, you can find a training code on the sine and stocks datasets.
To run the training process on the sine dataset or stock dataset, run the script of sine_regular.sh or stock_regular.sh respectively.


## Paper
```bibtex
@inproceedings{
naiman2024generative,
title={Generative Modeling of Regular and Irregular Time Series Data via Koopman {VAE}s},
author={Ilan Naiman and N. Benjamin Erichson and Pu Ren and Michael W. Mahoney and Omri Azencot},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=eY7sLb0dVF}
}
```

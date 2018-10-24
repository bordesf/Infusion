# Infusion Training

<p align="left">
  <img src="infusion.jpg" width="800"/>
</p>

This repository contains the code for the paper: <br />
**"Learning to Generate Samples from Noise through Infusion Training."**, <br />
Florian Bordes, Sina Honari, Pascal Vincent, ICLR 2017. <br />
https://arxiv.org/abs/1703.06975

In order to use it, you have to install Theano, Lasagne, Fuel and theirs dependencies. To run an
experiment on a GPU, you have to use:
```
THEANO_FLAGS=floatX=float32,device=cuda python run.py
```

If you use this code please cite:

Bibtex: 

    @inproceedings{bordes2017learning,
    title={Learning to generate samples from noise through infusion training},
    author={Bordes, Florian and Honari, Sina and Vincent, Pascal},
    booktitle={International Conference on Learning Representations (ICLR)},
    year={2017},
    }

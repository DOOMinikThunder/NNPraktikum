#!/bin/bash

# python RunAE.py dae_nr  dae_lr dae_epochs mlp_lr mlp_epochs hiddenLayerNeurons
#
# dae_nr = 0.3
# dae_lr = 0.05
# dae_epochs = 30
# mlp_lr = 0.05
# mlp_epochs = 30
# hiddenLayerNeurons = 100

python RunAE.py 0.1 0.05 10 0.05 30 10
python RunAE.py 0.1 0.09 10 0.05 30 30
python RunAE.py 0.1 0.05 20 0.05 30 50
python RunAE.py 0.1 0.09 20 0.05 30 70
python RunAE.py 0.1 0.05 30 0.05 30 90
python RunAE.py 0.1 0.09 30 0.05 30 100

python RunAE.py 0.3 0.05 10 0.05 30 10
python RunAE.py 0.3 0.09 10 0.05 30 30
python RunAE.py 0.3 0.05 20 0.05 30 50
python RunAE.py 0.3 0.09 20 0.05 30 70
python RunAE.py 0.3 0.05 30 0.05 30 90
python RunAE.py 0.3 0.09 30 0.05 30 100

python RunAE.py 0.5 0.05 10 0.05 30 10
python RunAE.py 0.5 0.09 10 0.05 30 30
python RunAE.py 0.5 0.05 20 0.05 30 50
python RunAE.py 0.5 0.09 20 0.05 30 70
python RunAE.py 0.5 0.05 30 0.05 30 90
python RunAE.py 0.5 0.09 30 0.05 30 100

python RunAE.py 0.6 0.05 10 0.05 30 10
python RunAE.py 0.6 0.09 10 0.05 30 30
python RunAE.py 0.6 0.05 20 0.05 30 50
python RunAE.py 0.6 0.09 20 0.05 30 70
python RunAE.py 0.6 0.05 30 0.05 30 90
python RunAE.py 0.6 0.09 30 0.05 30 100

python RunAE.py 0.7 0.05 10 0.05 30 10
python RunAE.py 0.7 0.09 10 0.05 30 30
python RunAE.py 0.7 0.05 20 0.05 30 50
python RunAE.py 0.7 0.09 20 0.05 30 70
python RunAE.py 0.7 0.05 30 0.05 30 90
python RunAE.py 0.7 0.09 30 0.05 30 100

python RunAE.py 0.8 0.05 10 0.05 30 10
python RunAE.py 0.8 0.09 10 0.05 30 30
python RunAE.py 0.8 0.05 20 0.05 30 50
python RunAE.py 0.8 0.09 20 0.05 30 70
python RunAE.py 0.8 0.05 30 0.05 30 90
python RunAE.py 0.8 0.09 30 0.05 30 100
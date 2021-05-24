# IRL-Depth_for_walking
This is the repo for the paper "An Environment-aware Predictive Modeling Framework for Human-Robot Symbiotic Walking"

About the network structure:

The depth prediction network is built upon an dense autoencoder with skip connection from encoder to decoder.
The bottlenect has size 64
The building blocks include regular conv layers (with kernel 3x3 and 1x1), conv_blocks and identity_blocks (from Resnet).
Loss function: MSE + gradient + Total variance (TV)

Input: 180x320 RGB images
Output: 180x320 depth images

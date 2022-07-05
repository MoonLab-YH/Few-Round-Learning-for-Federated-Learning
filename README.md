# Few-Round Learning for Federated Learning
Official Pytorch implementation for the paper titled "Few-Round Learning for Federated Learning"  presented on NeurIPS 2021.
![FRL_Overview](https://user-images.githubusercontent.com/54431060/177270454-9e95a2fb-b00b-4a48-acf8-6fb10e1a6b40.png)

# Abstract
In federated learning (FL), a number of distributed clients targeting the same task
collaborate to train a single global model without sharing their data. The learning
process typically starts from a randomly initialized or some pretrained model. In
this paper, we aim at designing an initial model based on which an arbitrary group
of clients can obtain a global model for its own purpose, within only a few rounds
of FL. The key challenge here is that the downstream tasks for which the pretrained
model will be used are generally unknown when the initial model is prepared.
Our idea is to take a meta-learning approach to construct the initial model so that
any group with a possibly unseen task can obtain a high-accuracy global model
within only R rounds of FL. Our meta-learning itself could be done via federated
learning among willing participants and is based on an episodic arrangement to
mimic the R rounds of FL followed by inference in each episode. Extensive
experimental results show that our method generalizes well for arbitrary groups
of clients and provides large performance improvements given the same overall
communication/computation resources, compared to other baselines relying on
known pretraining methods.

# Prerequsites
To load anaconda virtual environment,
```setup
conda env create -f FRL_env.yaml
```


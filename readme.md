# Benchmarking binary neural networks in computer vision by HyperbeeAI

Copyrights © 2023 Hyperbee.AI Inc. All rights reserved. hello@hyperbee.ai

This repository contains our experiments for quantized and binary neural networks for computer vision tasks, evaluated over the CIFAR100 benchmark dataset.

See checkpoints/ and associated evaluation scripts. See documentation/ for more information on results:

![results](./documentation/edited-results-graph.png)

## Installation / Setup (Linux)

Run the installer:

	bash -i setup.sh

## Evaluate a checkpoint

Activate the virtual environment installed by setup.sh, copy files "evaluation.py" and "training_checkpoint.pth.tar" from the checkpoint folder you want to evaluate to the top level, and run evaluation.py. Note that you need to modify the following line:

	checkpoint = torch.load('training_checkpoint.pth.tar');

to 

	checkpoint = torch.load('training_checkpoint.pth.tar', map_location=torch.device('cpu'));

if you are evaluating on a CPU-only instance since the weight files are originally saved for CUDA-capable GPUs.

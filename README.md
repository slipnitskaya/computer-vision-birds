# Bird by Bird using Deep Learning
## Tutorial on Advanced Classification of Bird Species

This repository contains materials accompanying a serie of articles “Bird by Bird using Deep Learning” coupled with interactive Python notebooks and published by Sofya Lipnitskaya on [Medium](https://medium.com/@slipnitskaya). 

### Motivation
Here, we are going to tackle such an established problem in computer vision as fine-grained classification of bird species. The tutorial demonstrates how to classify bird images from the Caltech-UCSD Birds-200-2011 ([CUB-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)) dataset using [PyTorch](https://github.com/pytorch/pytorch), one of the most popular open-source frameworks for deep learning experiments.

### Learning goals 
By the end of this tutorial, you will be able to:

* Understand basics of image classification problem of bird species.
* Determine the data-driven image pre-processing strategy.
* Create your own deep learning pipeline for image classification.
* Build, train and evaluate ResNet-50 model to predict bird species.
* Enhance CNN's performance by using different techniques.

### Structure
 Here you can get familiarized with the content more properly:

* Part 1: “Advancing CNN model for fine-grained classification of birds” ([notebook](https://github.com/slipnitskaya/caltech-birds-advanced-classification/blob/master/notebook.ipynb), [article](https://towardsdatascience.com/bird-by-bird-using-deep-learning-4c0fa81365d7))
* Part 2: “Interpretable deep learning for computer vision“ [TBA]
* Part 3: “Multimodal data fusion approach for bird classification“ [TBA]

### Summary
Part 1 demonstrates how to perform the data-driven image pre-processing, to build a baseline ResNet-based classifier, and to further improve it's performance for bird classification using different apporaches. Results indicate that the final variant of the ResNet-50 model advanced with transfer and multi-task learning, as well as with the attention module greatly contributes to the more accurate bird predictions.

### Libraries
Before running the tutorial, please install `numpy`, `torch`, `torchvision`, `scikit-learn`, and `matplotlib` packages.

### License
Except as otherwise noted, the content of this repository is licensed under the [Creative Commons Attribution Non Commercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/legalcode), and code samples are licensed under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0).
All materials can be freely used, distributed and adapted for non-commercial purposes only, given appropriate attribution to the licensor and/or the reference to this repository.

SPDX-License-Identifier: CC-BY-NC-4.0 AND Apache-2.0

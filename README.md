# Bird by Bird AI Tutorials
## Python tutorials on computer vision for classification of bird species

This repository contains materials accompanying a series of articles “Bird by Bird Tech” published on [Medium](https://medium.com/@slipnitskaya). 

### Motivation
Here, we are going to tackle such an established problem in computer vision as fine-grained classification of bird species.
The first part of the tutorials demonstrates how to use CNN models to classify bird images based on the Caltech-UCSD Birds-200-2011 
([CUB-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)) dataset using [PyTorch](https://github.com/pytorch/pytorch).
By the end of these tutorials, you will be able to:
* Understand basics of image classification problem of bird species.
* Determine the data-driven image pre-processing strategy.
* Create your own deep learning pipeline for image classification.
* Build, train and evaluate ResNet-50 model to predict bird species.
* Enhance CNN's performance by using different techniques.

### Structure
 Here you can get familiarized with the content more properly:

* Part 1: “Advancing CNN model for fine-grained classification of birds” ([notebook](https://github.com/slipnitskaya/caltech-birds-advanced-classification/blob/main/notebooks/cnn_image_classification.ipynb), [article](https://towardsdatascience.com/bird-by-bird-using-deep-learning-4c0fa81365d7)).
* Part 2: “Finite automata simulation for leveraging AI-assisted systems“ ([notebook](https://github.com/slipnitskaya/caltech-birds-advanced-classification/blob/main/notebooks/fsm_simulation_modelling.ipynb), [article](https://towardsdatascience.com/bird-by-bird-using-finite-automata-9d50b36bcbd3), [tutorial](TBA)).
* Part 3: “Optimizing AI-based systems on object detection using Monte-Carlo“ [TBA].
* Part 4: “Interpretable deep learning for computer vision“ [TBA].
* Part 5: “Multimodal data fusion approach for bird classification“ [TBA].

### Summary
Part 1 demonstrates how to perform the data-driven image pre-processing, to build a baseline ResNet-based classifier, and to further improve it's performance for bird classification using different approaches.
Results indicate that the final variant of the ResNet-50 model advanced with transfer and multi-task learning, as well as with the attention module greatly contributes to the more accurate bird predictions.
Part 2 focuses on simulation modelling using finite state machines for AI-assisted computer vision systems towards improved efficiency on bird detection.
More information on experimental design and results can be found in notebooks and articles.

### Libraries
Before running the code, make sure to install project dependencies indicated in the [requirements](https://github.com/slipnitskaya/Bird-by-Bird-AI-Tutorials/blob/main/requirements.txt) file.

### License
Except as otherwise noted, the content of this repository is licensed under the [Creative Commons Attribution Non Commercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/legalcode), and code samples are licensed under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0).
All materials can be freely used, distributed and adapted for non-commercial purposes only, given appropriate attribution to the licensor and/or the reference to this repository.

SPDX-License-Identifier: CC-BY-NC-4.0 AND Apache-2.0

# HFT Teaching Repository of Michael Mommert

This repository contains a collection of Jupyter Notebooks that is used by [Michael Mommert](https://mommermi.github.io/) in his teaching activities at the [Stuttgart University of Applied Science (HFT)](https://www.hft-stuttgart.com/).

The notebooks are optimized for running them in cloud computing
environments such as Binder or Google Colab (launchers are provided
for your convenience). Feel free to use and share these notebooks for
your own learning or teaching activities.

## Content

Notebooks are loosely ordered by the following topics. 

### Data Processing

Noteboks related to the processing of different data modalities.

* **Feature Extraction**: An interactive Notebook introducing different feature extraction techniques for different data modalities. *Interactivity*: high, *Prerequisites*: some experience with Python <br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Hochschule-fuer-Technik-Stuttgart/teaching-mommert/main?labpath=data_processing%2Ffeature_extraction%2Ffeature_extraction.ipynb)  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/Hochschule-fuer-Technik-Stuttgart/teaching-mommert/blob/main/data_processing/feature_extraction/feature_extraction.ipynb)

### Python

Introductory Notebooks for learning the Python programming language.

* **Basics**: A general introduction into the basics of the the Python programming language. *Interactivity*: low. *Prequisites*: None <br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Hochschule-fuer-Technik-Stuttgart/teaching-mommert/main?labpath=python%2Fbasics%2Fbasics.ipynb)  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/Hochschule-fuer-Technik-Stuttgart/teaching-mommert/blob/main/python/basics/basics.ipynb)

* **Numpy and Matplotlib**: A more detailed introduction into the Numpy and Matplotlib modules. *Interactivity*: low. *Prequisites*: basic Python <br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Hochschule-fuer-Technik-Stuttgart/teaching-mommert/main?labpath=python%2Fnumpy_matplotlib%2Fnumpy_matplotlib.ipynb)  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/Hochschule-fuer-Technik-Stuttgart/teaching-mommert/blob/main/python/numpy_matplotlib/numpy_matplotlib.ipynb) 

* **Pandas**: An introduction into Pandas module for data processing and analysis. *Interactivity*: low. *Prequisites*: basic Python <br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Hochschule-fuer-Technik-Stuttgart/teaching-mommert/main?labpath=python%2Fpandas%2Fpandas.ipynb)  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/Hochschule-fuer-Technik-Stuttgart/teaching-mommert/blob/main/python/pandas/pandas.ipynb) 


### Remote Sensing

Notebooks related to remote sensing activities and tasks.

* **LULC Classification with Machine Learning**: This example showcases the use of traditional machine learning methods
for pixel-wise land-use/land-cover classification. Simple data annotation by hand, as well as Maximum likelihood estimation and k-nearest neighbors methods are introduced for this purpose. *Interactivity*: medium. *Prerequisites*: some experience with Python. <br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Hochschule-fuer-Technik-Stuttgart/teaching-mommert/main?labpath=remote_sensing%2Fclassification%2Flulc_ml%2Flulc_ml.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/Hochschule-fuer-Technik-Stuttgart/teaching-mommert/blob/main/remote_sensing/classification/lulc_ml/lulc_ml.ipynb)

* **LULC Classification with Convolutional Neural Networks**: This example introduces the use of Convolutional Neural Networks for the task of image classification. *Interactivity*: low. *Prerequisites*: some experience with Python, familiarity with LULC machine learning Notebook useful. <br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Hochschule-fuer-Technik-Stuttgart/teaching-mommert/main?labpath=remote_sensing%2Fclassification%2Flulc_cnn%2Flulc_cnn.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/Hochschule-fuer-Technik-Stuttgart/teaching-mommert/blob/main/remote_sensing/classification/lulc_cnn/lulc_cnn.ipynb)



* **LULC Classification with Multilayer Perceptrons**: This tutorial introduces the use of Multilayer Perceptrons for pixel-wise classification for land-use/land-cover classification. *Interactivity*: low. *Prerequisites*: some experience with Python. <br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Hochschule-fuer-Technik-Stuttgart/teaching-mommert/main?labpath=remote_sensing%2Fclassification%2Flulc_dl%2Flulc_dl.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/Hochschule-fuer-Technik-Stuttgart/teaching-mommert/blob/main/remote_sensing/classification/lulc_dl/lulc_dl.ipynb)

* **LULC Data Labeling and Classification**: This tutorial introduces the data labeling process for multiband imaging data. The data are used for a pixel-wise classification of Sentinel-2 data. *Interactivity*: medium. *Prerequisites*: some experience with Python. <br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Hochschule-fuer-Technik-Stuttgart/teaching-mommert/main?labpath=remote_sensing%2Fclassification%2Flulc_labeling%2Flulc_labeling.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/Hochschule-fuer-Technik-Stuttgart/teaching-mommert/blob/main/remote_sensing/classification/lulc_labeling/lulc_labeling.ipynb)

* **Regression on the California Housing Dataset**: This is an introductory notebook, featuring different traditional machine learning methods applied to a tabular dataset. *Interactivity*: medium. *Prerequisites*: some experience with Python. <br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Hochschule-fuer-Technik-Stuttgart/teaching-mommert/main?labpath=remote_sensing%2Fregression%2Fcalifornia_housing%2Fcalifornia_housing.ipynb)  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/Hochschule-fuer-Technik-Stuttgart/teaching-mommert/blob/main/remote_sensing/regression/california_housing/california_housing.ipynb)

* **Unsupervised Segmentation of Sentinel-2 Images**: This example contains segmentation examples based on k-Means clustering and SLIC for a small Sentinel-2 multispectral dataset. *Interactivity*: medium. *Prerequisites*: some experience with Python. <br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Hochschule-fuer-Technik-Stuttgart/teaching-mommert/main?labpath=remote_sensing%2Fsegmentation%2Fkmeans_slic%2Fkmeans_slic.ipynb)  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/Hochschule-fuer-Technik-Stuttgart/teaching-mommert/blob/main/remote_sensing/segmentation/kmeans_slic/kmeans_slic.ipynb)

* **LULC Segmentation of Sentinel-2 Images with a UNet**: This Notebook introduces the workflow for supervised learning with a UNet architecture available in Pytorch. We will us dense labels available in the *ben-ge-800* dataset for this task. *Interactivity*: low. *Prerequisites*: some experience with Python. <br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Hochschule-fuer-Technik-Stuttgart/teaching-mommert/main?labpath=remote_sensing%2Fsegmentation%2Flulc_unet%2Flulc_unet.ipynb)  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/Hochschule-fuer-Technik-Stuttgart/teaching-mommert/blob/main/remote_sensing/segmentation/lulc_unet/lulc_unet.ipynb)

* **Data Augmentations for LULC Segmentation**: Building on the "LULC Segmentation of Sentinel-2 Images with a UNet"-Notebook, this Notebook introduces data augmentations and experiments with a few simple transformations. *Interactivity*: medium. *Prerequisites*: experience with Python and semantic segmentation using a UNet. <br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Hochschule-fuer-Technik-Stuttgart/teaching-mommert/main?labpath=remote_sensing%2Fsegmentation%2Flulc_unet_dataaugmentations%2Flulc_unet_dataaugmentations.ipynb)  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/Hochschule-fuer-Technik-Stuttgart/teaching-mommert/blob/main/remote_sensing/segmentation/lulc_unet_dataaugmentations/lulc_unet_dataaugmentations.ipynb)

* **Object Detection with YOLO for Aerial Imagery**: This Notebook introduces object detection using YOLO for aerial imagery. We will detect cars from aerial imagery of the city of Stuttgart. *Interactivity*: low. *Prerequisites*: some experience with Python. <br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Hochschule-fuer-Technik-Stuttgart/teaching-mommert/main?labpath=remote_sensing%2Fobject_detection%2Fyolo%2Fyolo.ipynb)  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/Hochschule-fuer-Technik-Stuttgart/teaching-mommert/blob/main/remote_sensing/object_detection/yolo/yolo.ipynb)

* **Thermal Imaging with flyr**: [Flyr](https://pypi.org/project/flyr/) is a library for extracting thermal data from FLIR images written fully in Python. We use this library to read in, modify and analyse thermograms. *Interactivity*: medium. *Prerequisites*: some experience with Python. <br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Hochschule-fuer-Technik-Stuttgart/teaching-mommert/main?labpath=remote_sensing%2fthermal%2Fthermal_imaging_flyr%2Fthermal_imaging_flyr.ipynb)  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/Hochschule-fuer-Technik-Stuttgart/teaching-mommert/blob/main/remote_sensing/thermal/thermal_imaging_flyr/thermal_imaging_flyr.ipynb)



### Supervised Learning

Introduction to supervised learning concepts and methods.


* **Preparing Data for Supervised Learning**: This Notebook serves as an interactive worksheet for preparing a dataset for use in a machine learning model. It will guide you through the steps required to use a dataset in a supervised learning setup. *Interactivity*: high. *Prerequisites*: Numpy experience. <br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Hochschule-fuer-Technik-Stuttgart/teaching-mommert/main?labpath=supervised_learning%2Fpreparation%2Fsl_preparation.ipynb)  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/hft-teaching/blob/main/supervised_learning/preparation/sl_preparation.ipynb)

* **Full Supervised Learning Pipeline**: This Notebook combines all the pieces together to perform a classification task with scikit-learn. *Interactivity*: low. *Prerequisites*: some experience with Python. <br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Hochschule-fuer-Technik-Stuttgart/teaching-mommert/main?labpath=supervised_learning%2Fpipeline%2Fsl_pipeline.ipynb)  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/Hochschule-fuer-Technik-Stuttgart/teaching-mommert/blob/main/supervised_learning/pipeline/sl_pipeline.ipynb)

* **Image Classification with a Multilayer Perceptron**: In this Notebook we build a Multilayer Perceptron from scratch using Pytorch and train it on a simple image classification task.  *Interactivity*: low. *Prerequisites*: some experience with Python. <br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Hochschule-fuer-Technik-Stuttgart/teaching-mommert/main?labpath=supervised_learning%2Fmlp%2Fmlp.ipynb)  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/Hochschule-fuer-Technik-Stuttgart/teaching-mommert/blob/main/supervised_learning/mlp/mlp.ipynb)

* **Image Classification with a Convolutional Neural Network**: In this Notebook we build a convolutional neural network from scratch using Pytorch and train it on a simple image classification task.  *Interactivity*: low. *Prerequisites*: some experience with Python. <br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Hochschule-fuer-Technik-Stuttgart/teaching-mommert/main?labpath=supervised_learning%2Fcnn%2Fcnn.ipynb)  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/Hochschule-fuer-Technik-Stuttgart/teaching-mommert/blob/main/supervised_learning/cnn/cnn.ipynb)


### Unsupervised Learning

Introduction to supervised learning concepts and methods.

* **Clustering and PCA**: A brief introduction into unsupervised learning methods such as clustering and Principal Component Analysis.
*Interactivity*: medium. *Prerequisites*: Numpy experience. <br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Hochschule-fuer-Technik-Stuttgart/teaching-mommert/main?labpath=unsupervised_learning%2Fclustering_pca%2Fclustering_pca.ipynb)  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/Hochschule-fuer-Technik-Stuttgart/teaching-mommert/blob/main/unsupervised_learning/clustering_pca/clustering_pca.ipynb)



## License

All notebooks contained in this repository are provided under MIT license. 
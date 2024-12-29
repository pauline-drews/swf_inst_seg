This is the project for my Master Thesis with the title:

"Instance-Aware Semantic Segmentation of Small Woody Landscape Features 
 Using High-Resolution Aerial Images in Biodiversity Exploratories"

 The goal of the project is to create an instance segmentation deep learning 
 model that segments and instantiates small woody features of three classes 
 (deciduous trees and tree groups, coniferuous trees and tree groups, and 
 hedges < 5 m height) in rgb nir 32 bit aerial images of 0.02 m resolution. The 
 approach is to combine a deep learning segmentation model for pixel-wise 
 classification of the images and a subsequent watershed algorithm on the 
 segmentation predictions for instantiation. Images and mask polygons are 
 available, but must be pre-processed and converted to a suitable training 
 dataset.

 ##############################################################################
 1. Features

 The main features of this project are saved individual scripts with the run-
 ning numbers 01-06 and the segmentation model architectures:

 Script 01_extract_images.R serves for retrieving the images from the RSDB 
 database and perform first pre-processing related to crs compatibility and 
 removing damaged data. The images are used both, for generating a training
 dataset and for generating large maps on which the final models were used to 
 predict. Parallel to this, the polygons from atkis and modified atkis were 
 merged and filtered in QGis to produce one .shp file containing all SWF 
 polygons.

 Script 02_dataset_generation.py serves to create a training dataset from the
 retrieved images and filtered polygons. This is done by cutting the large 
 images into patches of 256x256 pixels and generating the corresponding mask 
 patches by determining the intersection between the images and polygons. Only
 image-mask pairs with sufficient informative value are kept and saved (at 
 least 5% of pixels is a polygon intersection).

 Script 03_data_preparation.py serves to prepare the dataset for training. This
 consists of sanity checks, pre-processing and spliting it into training, 
 validation, and test subsets for different mask variants (binary, each 
 individual class, trees vs. hedges, multi-class).

 Script 04_train_models.py serves to define the deep learning segmentation 
 models, train them on different mask variant datasets, and extract the 
 training statistics, and final model weights.

 Script 05_model_evaluation.py serves to test the final segmentation models on 
 unseen test data and determine their performance.

 Script 06_model_prediction_and_watershed.py serves to use the segmentation 
 models to predict on the large prediction images and instantiate the result 
 using watershed algorithm.

 Scripts res_unet_attention.py, simple_multi_unet_model.py, and 
 simple_unet_model.py are the model architectures for the models trained in 
 04_train_models.py.

 Script z_some_plots.py is used to create some visualizations for the report.

 ##############################################################################
 2. Setup

 The R script was run using R version 4.4.0 (2024-04-24 ucrt).

 All python scripts were run in the "segmod" conda environment, which is saved
 to the environment.yml file. For model training, some scripts were exported to
 Google Colab to use faster A100 GPU.

 Paths are mostly absolute and refer to local directories and must be adaped if 
 working from other devices. 

 ##############################################################################
 3. Credits and Acknowledgements

 The idea for this project was developed by Dr. Nils Nölke who also provided me
 with the prepared SWF polygons as well as connected me with the responsibles
 of the RSDB database to make the image data available to me, too. He also 
 provided guidance and support throughout my thesis journey.

 The choice for the used segmentation models was strongly influenced by the 
 great work of Dr. Sreenivas Bhattiprolu. His YouTube channel DigitalSreeni and 
 his GitHub repository https://github.com/bnsreenu/python_for_microscopists was
 the base of my workflow and an indispensible source of example code. The code 
 for training the segmentation models and applying the watershed algorithm were 
 strongly oriented on his tutorials and example scripts, as mentioned in more 
 detail in each of the scripts.

 
 









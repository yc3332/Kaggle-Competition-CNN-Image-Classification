# Kaggle-Competition-CNN-Image-Classification

### Background
here are 2 types of bottles: coke bottle and water bottle, and the students were asked to take pictures with their cellphones. A post-processing was done to make sure each picture has the same size

### Dataset
Training: 15000 images (3000 for each class). The labels are indicated by their sub-folders in the data zip file.
Testing: 3500 images (700 for each class). The labels are held by the Kaggle.com.

### Overview of the files
task5-kaggle.ipynb:Main code, all functions run in this notebook file, and all results are printed here
kaggle.py: This script defines the method to train our model and evaluate the trained model and the prediction of test data

### Layers Construction
input >> [conv2d-maxpooling-norm] >> [conv2d-maxpooling-norm] >> flatten >> DenseLayer >> AffineLayer >> softmax loss >> output

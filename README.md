# Cloud Image Segmentation #

Climate change has been a very debated topic and
has taken importance in a lot of decisions in the last years.
The purpose of this research is to better understand the clouds and their classification.
This research has the origin as machine learning contest by the Max Planck Institute for Meteorology.
To understand the clouds, cloud images where taken from space, the insititute has been determined to classify the
following four types of clouds.
![Alt text](images/cloud_types.png?raw=true "Cloud types")

### Segmentation Process ###
The segmentation process of the cloud dataset can be represented with the following image. Where from an image we get
four different segmentations, one per class. The left column represents the segmentation of the fish cloud type and the right of the
sugar cloud type, where flower and gravel where omitted because their presence is not in the original image.
![Alt text](images/cloud_segmentation_process.png?raw=true "Segementation process")
First we need to decode the ground truth labels from the dataset, then we use a fully convolutional
neural network to perform the segmentation, next we do a mask processing to ensure it only has values
of 0 or 1, by discarding values lower than a threshold and finally we join connected groups in the mask

### Network architecture ###
The network architecture used was an encoder-decoder where the encoder network was the pretained VGG19
network, the decoder was a symetrical network with the same number of pooling layers as the encoder
but with less convolutional layers. The loss function that was used is the Dice coefficient(2∗|X∩Y||X|+|Y|).
![Alt text](images/convnet.png)

### Model tracking ###
The development of the model training is tracked using W&B in [link](https://app.wandb.ai/polmonroig/cloud_segmentation)

### Where can I get the dataset? ###
The original dataset was downloaded from NASA Worldview altough for the purpose of this
contest the dataset was at Kaggle(https://www.kaggle.com/c/understanding_cloud_organization/data)

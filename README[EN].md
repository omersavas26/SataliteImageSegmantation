[EN]
# **BTK - Huawei Coding Marathon (2022 - AI)**
## **Project for segmentation of satellite imagery using artificial intelligence**

# **Important Disclosure**
I shared the problems that I encountered while trying to develop on Kaggle from the discord group. In the screen sharing I made with Mr. Alp, I showed him the situation. The biggest problem was not being able to pass the runtime to the GPU. While working on the CPU, I found inconsistencies such as the same code not working on different notebooks. Finally, I started to work in the Google Colab environment with the knowledge of  Mr. Alp (even though it cost me 1 day). This project was developed using the Google Colab environment.

### **Summary**

This project was developed by Ömer SAVAŞ. 
Due to the Hardware/Time constraint,
a limited training was carried out by creating a
network of optimum depth deliberately. 
If training is performed for a longer time with a deeper mesh,
the accuracy will increase even more.

In addition, 
the training weights could not be recorded periodically 
with the callback mechanism while the
 model was being trained due to disk constraints, 
 instead a manual recording mechanism was set
 up and the weights were always updated on the same file.

"train.ipbynb" is the file required to 
perform the training operations. It completes the training
 by performing the steps in the flow. The "predict.ipynb"
 file should be used to export the desired "output.csv"
 and "scores.txt" files using the weights obtained as a 
 result of the training.

In addition, the questions and stages in the documents sent 
to you are explained in this document. 
Content titles and questions requested from your side are 
highlighted in bold.

### **Flow**

1. Loading and importing libraries
2. Defining helper functions and variables
3. Reading the dataset and caching it for later
4. Normalization of the data set and transform to appropriate formats
5. Designing the model
6. Performing the training and recording the weights
7. Evaluation of loss and accuracy data
8. Estimation and preview of test set

### **Project Description**

#### **General**

deep learning models autoencode 
and mask rcnn were evaluated and autoencoder was chosen to use in this project.
Selection criteria described in the section of "Which machine learning model did you choose?".

Firstly, a few epoch trainings were carried out with a very limited model,
 and the following pictures were obtained and it was seen that
 the autoencoder model could be successful in solving this problem. 
 Model source code and initial estimation images are below

``` python
input_img = Input(shape = (size, size, 3))
x = Conv2D(16, (3,3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2,2), padding='same')(x)
x = Conv2D(16, (3,3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2,2), padding='same')(x)

x = Conv2D(16, (3,3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2,2))(x)
x = Conv2D(16, (3,3), activation='relu', padding='same')(x)
x = UpSampling2D((2,2))(x)
decoded = Conv2D(1, (3,3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
```

![autoencoderminimodelpredict.jpg](https://omersavas.com/dosya/autoencoderminimodelpredict.jpg)

Then the model specified in source 2 was examined and a 
depth was chosen between our mini autoencoder and the 
model in the source. Some layers were simplified and the 
model whose summary and diagram was drawn below, was decided upon. 
It is understood from the picture below that this model will be 
much more successful than the previous model after a few epochs 
of training.

![derinmodel](https://omersavas.com/dosya/derinmodel.jpg)

As a result, the training was carried out and 
continuous improvement was observed in the graphics.

![grafik](https://omersavas.com/dosya/grafik.jpg)

#### **Which data preprocessing steps did you perform on the dataset? **
Operations such as data reading, formatting, normalization were
 performed in the read_images() function. 
 Firstly, satellite images with jp2 extension were converted to RGB. Then it was converted to a numpy array and split by /255. Segment files with .tif extension were also converted to one hot vectors after being read as a single channel.

#### **How ​​did you set the attributes**
Since the training was carried out with the Autoencoder (deep learning) model,
 no extra attribute operation was performed. 
 "Undefined" has been added as the first element of the 
 class list only for "0", that is, undefined data in files with .tif extension.
 After performing the training, the undefined class was ignored while 
 exporting the output.csv and socres.txt files.

#### **How ​​did you use the dataset to prepare the model**
When the dataset was examined, it was seen that the fields of the classes should be prepared as polygons from the .tif files for the Mask RCNN model before the training. 
For this reason, it was decided to use the autoencoder model, which is thought to be more suitable for the data set and the problem instead of Mask RCNN model.

#### **Which machine learning model did you choose?**
Deep learning, which has proven itself in image processing, was chosen as the model. 
Among the deep learning approaches, Mask RCNN and Autoencoder were evaluated.
In fact, although the mask RCNN model was thought to give higher accuracy at first glance, 
the autoencoder approach was preferred since the attribute positions in polygons should be created as a separate file from the
 segment data with the .tif extension[1]. Autoencoder models are very successful in producing the desired pictures by looking at
 a source picture in the computer vision field without the need for extra attribute data due to their structure.

#### **How ​​did you determine the parameters of the model you chose**
I have used the autoencoder model for various tasks such as noise removal before. Basically, a few epoch tests were performed with 
a mini auto encoder model and it was decided that the model was suitable for solving the problem.
Then, it was decided to simplified the model in Source 2 and to use the version that would work with more performance.
 Afterwards, it was tried to obtain better results by changing hyperparameters such as learning rate and optimization 
 algorithm by making tests specific to this problem with a small data set.

#### **Explanations about the project steps requested in the sent document**
##### **1- Data Pre-processing**
Under the heading "**Defining helper functions and variables**", the "**read_images**" function performs this operation.
 A more detailed explanation is also in the header of this document "**Which data preprocessing steps did you perform on the dataset? **".

##### **2- Selection of Machine Learning model**
In this document, "**Which machine learning model did you choose?**" contains theoretical information and comparisons about model selection. Moreover
In the "**Designing the model**" section, the model and its layers are coded in detail. 
In the following blocks, there is a summary of the model and diagram drawings.

##### **3- Training the model**
The model training designed in the section "**Performing the training and recording the weights**" was carried out.
 As mentioned above, due to the disk constraint problem, a manual weight and learning information recording mechanism is coded in the same section.

##### **4- Measuring model performance**
The accuracy performance is measured step by step during the training with the validation set.
 In addition, since the learning information was recorded for each step, the loss and accuracy information were followed graphically after the training part. 
 For this, you may see the section "**Assessment of loss and accuracy data**". 
 In addition, the "predict.ipynb" file also measures the prediction time information for the image set it predicts.

##### **5- Obtaining validation set results**
This process is carried out in the "predict.ipynb" file so that the "train.ipynb" file does not to be more complicated and has 
the appropriate content for its name. The validation set and test set are estimated with the final weights of the model, 
and the relevant output.csv and scores.txt files are exported.

### **Conclusion**
Even as a result of the limited training on Google colab, a good result of 75% was obtained. 
Train and Predict files were created and documented in 2 languages.

### **Personal Evaluation**
Clean code principles have been tried to be applied throughout the entire coding, and function/variable names have been chosen as appropriate and descriptive. However, do not hesitate to call if there are any unclear points (Ömer SAVAŞ: 0554 377 54 43)

As mentioned above, I was torn between two deep learning models. The Mask RCCN model will definitely run faster than the auto encoder model. But does it produce more accurate results? It's unclear without trying. If I have time, I will write a script that automatically extracts polygon attributes from .tif images and compare the results with the Mask RCNN model.

During model detection, I tried to use transfer learning using the VGG16 (imagenet) model before the autoencoder, but it did not show the accuracy increase I expected, and it brought a serious performance loss. Again, if I can find time, I would like to produce a more successful model by making various experiments. I'm considering options such as trying to put some of the models like VGG16 before the Auto encoder or between the encoder and decoder.

In addition, there is a situation that I overlooked on the first day because I lost motivation and time while dealing with the problem caused by kaggle . Or I'm deceiving myself with this excuse :) Training and estimating with a big 1000*1000 picture is very costly. Instead, it would be more appropriate to split the image into smaller sizes and add it to the dataset as thumbnails.

### **Annotation**
[1]: It is known that Tensorflow's object detection API and RCNN models also benefit from transfer learning and give very successful results with small data sets. For this reason, in the first place, I determined the boundaries of the areas by passing the tif image through the Canny edge detection algorithm with the help of open cv. Then automatically generate the polygons using these boundaries; I planned to export the polygons file that the API will use in training from there, but I gave up because time was limited.

### **Resources**
1. https://medium.com/@omersavas26/derin-%C3%B6%C4%9Frenme-hakk%C4%B1nda-neredeyse-her-%C5%9Fey-1-91bb8ddfde0
2. https://colab.research.google.com/github/dhassault/tf-semantic-example/blob/master/01b_semantic_segmentation_basic_colab.ipynb#scrollTo=TlAIZzR600uK
3. https://colab.research.google.com/drive/1ICnxAcVKOLaDcgrHh2SvI5Rvbdmrpxsd#scrollTo=qKM9ZgMB7umJ
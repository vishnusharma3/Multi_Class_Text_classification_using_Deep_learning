# Multiclass-Text-classification-with-Deeplearning

Repository containing Deep Learning model for multiclass classification for large data sets and huge number of classes 



## Technologies used:

 * Python 3.8
 * Tensorflow 2.3
 * keras 2.4.3


## Setup:

#### Python distribution
* Anaconda
* or any other python distribution 

#### Packages to install

* keras 
* tensorflow
* nltk.words
* nltk.stem
* nltk.corpus
* nltk.tokenize
* pathlib

#### Installing packages 

* open anaconda prompt and then in the console just type pip install package_name to install the package required

if you want to install tensorflow just type
`pip install tensorflow`


## input data requirement:

## Launch:
 * you need a working python environment with all the packages installed .
 * place the data file(file which contains training data) in the input data folder. 
 * Then you can run the python file(multi_class_text_classification_deeplearning) to Train the model.
 * you can only run the file by giving the full path of the python file along with its name while running it using the python command.
 ![Start Training](/images/start_training.PNG)
 
## Model architecture:
* The model we have created is a fully connected neural network of three layers 
* we have followed the 1/3rd approach by reducing the number of nodes by 1/3 in the next nodes
* we have used the TF-IDF vectorizer as the vectorization technique



 ![Model](/images/model.PNG)
 ## solution design:
 ![Solution design](/images/solution_design.PNG)
## performance comparision:

#### the comparision between conventional machine learning models and our deep learning model
* conventional machine learning models are designed only for less number of classes and less data points
* we have  increased accuracy upto of about 20 percent more than the conventional machine learning models

![Performance](/images/performance.PNG)

## business value:
* Model can handle large number of data points
* The model can handle large number of classes
* Increased accuracy
* Automated training 
* We have incorporated transfer learning also

![Business value](/images/business.PNG)


## scope of functionalities:

 can be used in
 * social media analytics
 * chat bots
 * healthcare

![Applications](/images/applications.PNG)







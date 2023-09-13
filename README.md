# Final Year Project Installation


This project includes the implementation of the Convolutional Neural Network (CNN), with this a better understanding can be constructed to look at how deep learning can be implemented within the medical field. Not only does the project aim to provide an application for the identification of Alzheimer’s with the use of CNN but also to gain a better insight into the different deep learning layers. Therefore, you will also be able to find other well-known deep learning models like the ResNet model, and VGG model.



 


## Installing packages required using virtual enviroment

1. This project was built using the version of Python 3.9.13 with anaconda3 for setting up the virtual enviroment.
2. Once you've downloaded Python 3.9.13 and anacdonda3 go into the root directory of the project using the anaconda prompt.
3. Create the virtual enviroment using "conda create --name myenv python=3.9.13"
4. Continue as follows once its finished setting the virtual enviroment use "conda activate myenv" to activate the virtual enviroment
5. Once you've activated the enviroment use "pip install -r requirements.txt" to install all packages from the requirement file.
6. You should see all the packages inside the requirement.txt file being downloaded in your terminal.


To successfully run any of these scripts please run the individual python file and not "Run Code".

If your using virtual enviroment from anaconda you can do so by doing: & C:/Users/user_name/anaconda3/envs/alzvenv/python.exe "c:/Users/user_name/OneDrive/Desktop/Actual ProjectsFinal/Final Project/Main.py"


## Content

NetworkLayer.py - Consists of source code for layers in a CNN coded from scratch with the trainnetwork function. Running it wont do anything. (Learning_Rate can be adjusted here)

NetworkTrainer.py - Includes the methods for training and predicting a network consists of the CNN architecture. (Running it will initialise a new folder in training logs and begin training, and save it there after training)

Main.py - Script that handles the GUI interaction and placement. Run this to open the GUI and test the model. (The script is set to use the latest model so the latest cnn model number is used)

ResNetModel.py - Pre-trained model using resnet, was used for experiments, running it will create a new cnn_model in training log and save weights after training is done

VGG-16.py - Pre trained module using VGG-16, was used for experiments, running it will create a new cnn_model in training log and save weights after training is done

TrainingLog - This folder consists of all the models were built with the training of CNN models. They are stored so they can be loaded and used for in model selection when the user is ready for prediction.

UITemplate - Consists of all the UI templates made with Qt Designer.

Alzheimer_sDataset - This is the dataset used and was later found out to be unreliable with the data source missing, loss of spatial information due to the extraction from a volumetric image, as well as the different distribution between the test and the training sets. The dataset was extracted from "https://www.kaggle.com/datasets/tourist55/alzheimers-dataset-4-class-of-images"




## Usage

Once you've run Main.py a small GUI interface will should appear shortly where you can select the pre-trained model you want to use to identify
your brain scan with. The current models that are available there are my own built-from-scratch model as well as other trained models
from known packages like tensorflow/keras.

Once you've selected your model, you can then import your image and click "predict image". Following this you should see a results page 
based on the image you put in along with confident measures based on the model you picked, and what the model believes the image is classed as.
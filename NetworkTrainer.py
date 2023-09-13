from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from keras.models import load_model
import os
import cv2
import numpy as np
from NetworkLayers import *
from PIL import Image, ImageEnhance
from scipy import ndimage
import time
import random
from sklearn.utils import class_weight


STEP_BATCH = 100
EPOCHS = 2
classes = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]
current_images = np.zeros(len(classes))
data_direc = "Alzheimer_s Dataset"
batch_size = 32
label_to_int = {label: i for i, label in enumerate(classes)}

print(label_to_int)
dataset = []


CNN_Layers = [
       
       

        ConvolutionalLayer(3, 32, 1),
        ConvolutionalLayer(3, 32, 1),
        MaxPoolingLayer(2, 2),

        ConvolutionalLayer(3, 128, 1),
        ConvolutionalLayer(3, 128, 1),
        MaxPoolingLayer(2, 2),

       # ConvolutionalLayer(3, 256, 1),
       # ConvolutionalLayer(3, 128, 1),
      #  MaxPoolingLayer(2, 2),

        FlattenLayer(),
        DropoutLayer(.2),
       
        SoftmaxDenseLayer(127008, len(classes)),
        
    ]


def PredictResult(file_path, model_no):
    if model_no == None:
        model_no = len(os.listdir("./TrainingLogs"))
    load_dict = None
    if os.path.exists("./TrainingLogs/cnn_model_info_"+str(model_no)) and os.path.exists("./TrainingLogs/cnn_model_info_"+str(model_no)+"/model_data.npy"):
        load_dict = np.load("./TrainingLogs/cnn_model_info_"+str(model_no)+"/model_data.npy", allow_pickle=True).item()
        output = Predict(file_path, load_dict)
        dict = {}
        for i in range(len(output)):
            if classes[i]:
                dict[classes[i]] = output[i]   
        return dict
    elif os.path.exists("./TrainingLogs/cnn_model_info_"+str(model_no)) and os.path.exists("./TrainingLogs/cnn_model_info_"+str(model_no)+"/model_data.h5"):
        model = load_model("./TrainingLogs/cnn_model_info_"+str(model_no)+"/model_data.h5") 
        img = cv2.imread(file_path)
        img = cv2.resize(img, (128, 128))
        img = img/255.  
        input_image = np.expand_dims(img, axis=0)
        output = model.predict(input_image)
        dict = {}
        dict = {classes[i]: index for i, index in enumerate(output[0])}

        #predicted_classes = np.argmax(predictions, axis=1)
        return dict
        
    else:
        print("Model to load from does not exist")
        
            
    

def pad_image(img, new_size):
    height, width, _ = img.shape
    height, width, channels = img.shape
   # new_height = int(height * new_width / width)
    img = cv2.resize(img, (new_size, new_size))
    padding_height = new_size - height
    padding_width = new_size - width
    padding = [(0, padding_height), (0, padding_width), (0, 0)]
    return img
    #return np.pad(img, padding, mode="constant")
                
def generate_batches(arr, batch_size):
    batches = []
    for x in range(0, len(arr), batch_size):
        batch = arr[x:x + batch_size]
        batches.append(batch)
    return batches
      
def return_augmented_versions(image, i_class):
    #Previous data augmentation code
   # flip = np.fliplr(image)
    #rotate = ndimage.rotate(image, 90, reshape=True)
    #reflect = np.flipud(image)
   # noise_img = np.random.normal(0, 20, image.shape) + image
    #blur_img = cv2.GaussianBlur(image, (5, 5), 0)
    # return list of images
    
    datagen = ImageDataGenerator (
    rotation_range=20,  # randomly rotate images in the range 0-20 degrees
    zoom_range=0.2,  # randomly zoom image
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images horizontally
    vertical_flip=False,
    brightness_range=[0.5, 1.5],
    )
   
    image = np.expand_dims(image, axis=0)
    augmented_images = []
    for batch in datagen.flow(image, batch_size=1):
        # Reshape the augmented image to remove the batch dimension
        augmented_image = np.reshape(batch, batch.shape[1:])
        # Add the augmented image to the list
        augmented_images.append(augmented_image)
        dataset.append((augmented_image, i_class))
        current_images[label_to_int[i_class]] += 1
        # If we've generated the desired number of images, break the loop
        if len(augmented_images) == 10:
            break
    """
    def return_class_weight(label):
        class_frequency = np.bincount(y_train)
        t_samples = len(y_train)
        num_classes = len(classes)
        weight = t_samples/(num_classes*class_frequency[label])
        class_weight = np.sum(class_frequency) / (num_classes*class_frequency)
        return class_weight
    """
    
    
    
def EvaluateTest(test_folder_dir, model_no):
    acc = 0
    itr = 0
    loss = 0
    for image_classes in os.listdir(test_folder_dir):
            #Checking whether class is authentic
            if image_classes in classes:
                print("Image Class: "+image_classes+" is being loaded")
                for images in os.listdir(os.path.join(test_folder_dir, image_classes)):
                    path_to_image = os.path.join(test_folder_dir, image_classes, images)
                    output = PredictResult(path_to_image, model_no)
                    itr += 1
                    print("Correct class: "+image_classes)
                    print(output)
                    correct_key = max(output, key=output.get)
                    if correct_key == image_classes:
                        print("Correct: ")
                        acc += 1
                            
    acc /= itr
    print("Test Accuracy: "+str(acc)+" |")
    return acc
    # convert data to numpy arrays



    
        
    
    
def main():
    for data_types in os.listdir(data_direc):
    #Only training and validation need to be put into the model for training
        if data_types == "train":
            for image_classes in os.listdir(os.path.join(data_direc, data_types)):
                #Checking whether class is authentic
                if image_classes in classes:
                    print("Image Class: "+image_classes+" is being loaded")
                    for images in os.listdir(os.path.join(data_direc, data_types, image_classes)):
                        path_to_image = os.path.join(data_direc, data_types, image_classes, images)
                        try: 
                            img = cv2.imread(path_to_image)
                            img = pad_image(img, 128)                         
                            if current_images[label_to_int[image_classes]] <= 500:
                                dataset.append((img, image_classes))
                                current_images[label_to_int[image_classes]] += 1
                                if image_classes == "ModerateDemented":
                                    imge, aug_arr, num_of_aug = return_augmented_versions(img, image_classes)
                        except Exception as e:
                            continue
                            #print("Unable to read image "+path_to_image)
    
    print(current_images)
    
    
    
    np.random.shuffle(dataset)
    #Begining of dataset to 70% of the dataset will be our training set
    train_data = dataset[:int(0.7*len(dataset))]  # 70% of the data will be for training
    val_data = dataset[int(0.7*len(dataset)):]  # 30% of the data will be for validation

    # convert data to numpy arrays
    x_train = np.array([x[0] for x in train_data])
    y_train = np.array([label_to_int[x[1]] for x in train_data])
    x_val = np.array([x[0] for x in val_data])
    y_val = np.array([label_to_int[x[1]] for x in val_data])

 
    print(x_train[0].shape)
    num_files = len(os.listdir("./TrainingLogs"))
    model_saving_path_name = "./TrainingLogs/cnn_model_info_"+str(num_files+1)
    if not os.path.exists(model_saving_path_name):
        os.makedirs(model_saving_path_name)
    start_time = time.time()
    
    val_loss_array = []
    val_acc_array = []
    test_loss_array = []
    test_acc_array = []
    print(model_saving_path_name+" folder created.")
    #Save the image shape for pre processing when predicting
    
    for epoch in range(EPOCHS):
        print("---Epoch: "+str(epoch)+"---")
        loss = 0
        accuracy = 0
        print(len(list(zip(x_train, y_train))))
        for i, (img, label) in  enumerate(list(zip(x_train, y_train))):
            if i > 0 and i % STEP_BATCH == STEP_BATCH-1:
                print("[Step "+str(i+1)+"]"+" Reached "+str(STEP_BATCH)+" steps: "+ " Average Loss: "+str(loss/100)+" | Accuracy: "+str(accuracy))
                #Reset loss and accuracy for next step
                loss = 0
                accuracy = 0
            
            computed_loss, computed_acc = TrainNetwork(img, label, CNN_Layers, y_train)
            loss += computed_loss
            accuracy += computed_acc
        
        if len(x_val) != 0:
            test_loss_array= np.append(test_loss_array, loss/len(x_val))
            test_acc_array = np.append(test_acc_array, accuracy/len(x_val))
        
        val_loss, val_acc = 0, 0
        for i, (img, label) in enumerate(list(zip(x_val, y_val))):
            computed_loss, computed_acc = Validate(img, label, CNN_Layers)
            val_loss += computed_loss
            val_acc += computed_acc
        
        # calculate and print the average validation loss and accuracy
        val_loss /= len(x_val)
        val_acc /= len(x_val)
        val_loss_array = np.append(val_loss_array, val_loss)
        val_acc_array =  np.append(val_acc_array, val_acc) 
        print("------------------")
        print("Validation Loss: "+str(val_loss)+" | Validation Accuracy: "+str(val_acc))
        print("------------------")
            
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("Training Completed")
        print("Total Training Time: {:.2f} seconds".format(elapsed_time))

        parameters = {
            "kernels": [],  # List to store the kernel parameters of ConvolutionalLayers
            "weights": [],  # List to store the weight parameters of SoftmaxDenseLayers
            "bias": [],  # List to store the bias parameters of SoftmaxDenseLayers
            "test_loss": test_loss_array,  # Array to store test loss values
            "test_acc": test_acc_array,  # Array to store test accuracy values
            "val_loss": val_loss_array,  # Array to store validation loss values
            "val_acc": val_acc_array,  # Array to store validation accuracy values
            "epochs": [EPOCHS],  # List to store the number of epochs
            "model": CNN_Layers,  # The list of layers in the CNN model
            "input_size": x_train[0].shape  # The shape of the input data
        }

        for layer in CNN_Layers:
            if isinstance(layer, ConvolutionalLayer):
                # If the layer is an instance of ConvolutionalLayer
                parameters["kernels"] = np.append(parameters["kernels"], layer.kernels)
            elif isinstance(layer, SoftmaxDenseLayer):
                # If the layer is an instance of SoftmaxDenseLayer
                parameters["weights"] = layer.weights
                parameters["bias"] = layer.bias

    
    
    
    np.save(model_saving_path_name+"/model_data.npy", parameters)

    
    
        
#EvaluateTest("Alzheimer_s Dataset/test", "1")   
if __name__ == '__main__':
  main()

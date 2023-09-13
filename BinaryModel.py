import tensorflow as tf
from keras.applications import ResNet50
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras.layers import Dense, Flatten, Dropout, BatchNormalization
from keras.utils import to_categorical
from keras.models import Model
import numpy as np
import keras
from keras import regularizers
from keras.optimizers import Adam
import os
import cv2
import matplotlib.pyplot as plt

# Define image size and number of classes
img_size = 128
num_classes = 4
dataset = []
test_data = []
data_direc = "Alzheimer_s Dataset"
classes = ["NonDemented", "Demented"]
current_images = np.zeros(len(classes))
test_images = np.zeros(len(classes))
label_to_int = {
    "NonDemented": 0,
    "MildDemented": 1,
    "ModerateDemented": 1,
    "VeryMildDemented": 1
}

class_split = {
    "MildDemented": 0,
    "ModerateDemented": 0,
    "VeryMildDemented": 0
}
class_test_split= {
    "MildDemented": 0,
    "ModerateDemented": 0,
    "VeryMildDemented": 0
}



def return_augmented_versions(image, i_class, testing):
   # flip = np.fliplr(image)
    #rotate = ndimage.rotate(image, 90, reshape=True)
    #reflect = np.flipud(image)
   # noise_img = np.random.normal(0, 20, image.shape) + image
    #blur_img = cv2.GaussianBlur(image, (5, 5), 0)
    # return list of images
    
    datagen = ImageDataGenerator (
    rotation_range=5,  # randomly rotate images in the range 0-20 degrees
    zoom_range=0.1,  # randomly zoom image
    width_shift_range=0.05,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.05,  # randomly shift images vertically (fraction of total height)
    #horizontal_flip=True,  # randomly flip images horizontally
    #vertical_flip=False,
    brightness_range=[0.8, 1.2],

    )
   
    image = np.expand_dims(image, axis=0)
    
    augmented_images = []
    for batch in datagen.flow(image, batch_size=1):
        # Reshape the augmented image to remove the batch dimension
        augmented_image = np.reshape(batch, batch.shape[1:])
        # Add the augmented image to the list
        augmented_images.append(augmented_image)
        if testing == True:
            test_data.append((augmented_image/255., i_class))
            test_images[label_to_int[i_class]] += 1
            class_test_split[image_classes] += 1
        else:
            dataset.append((augmented_image/255., i_class))
            current_images[label_to_int[i_class]] += 1
            class_split[image_classes] += 1
        # If we've generated the desired number of images, break the loop
        if len(augmented_images) == 20:
            break
        
        
def pad_image(img, new_size):
    height, width, _ = img.shape
    padding_height = new_size - height
    padding_width = new_size - width
    padding = [(0, padding_height), (0, padding_width), (0, 0)]
    return np.pad(img, padding, mode="constant")


for data_types in os.listdir(data_direc):
    #Only training and validation need to be put into the model for training
        if data_types == "train" or data_types == "test":
            for image_classes in os.listdir(os.path.join(data_direc, data_types)):
                #Checking whether class is authentic
                if image_classes in label_to_int:
                    
                    print("Image Class: "+image_classes+" is being loaded")
                    for images in os.listdir(os.path.join(data_direc, data_types, image_classes)):
                        path_to_image = os.path.join(data_direc, data_types, image_classes, images)
                        try:
                            img = cv2.imread(path_to_image)
                            img = cv2.resize(img, (128, 128))
                            if data_types == "train":
                                
                                if current_images[label_to_int[image_classes]] <= 1000:
                                    if image_classes != "NonDemented" and class_split[image_classes] <=330:
                                        class_split[image_classes] += 1
                                        
                                    dataset.append((img/255., image_classes))
                                    current_images[label_to_int[image_classes]] += 1
                                    
                                    if image_classes == "ModerateDemented" or image_classes == "MildDemented": #Classes with least samples
                                        imge, aug_arr, num_of_aug = return_augmented_versions(img, image_classes, False)
                            else:
                                 if test_images[label_to_int[image_classes]] <= 500:
                                    if image_classes != "NonDemented" and class_split[image_classes] <=330: 
                                        class_test_split[image_classes] += 1
                                    test_data.append((img/255., image_classes))
                                    test_images[label_to_int[image_classes]] += 1
                                    #if image_classes == "ModerateDemented" or image_classes == "MildDemented": #Classes with least samples
                                    imge, aug_arr, num_of_aug = return_augmented_versions(img, image_classes, True)
                                        
                                    
                        except Exception as e:
                            continue
                            #print("Unable to read image "+path_to_image)


np.random.shuffle(dataset)


                            
train_data = dataset[:int(0.7*len(dataset))]  # 70% of the data will be for training
val_data = dataset[int(0.7*len(dataset)):]  # 30% of the data will be for validation
x_train = np.array([x[0] for x in train_data])
x_val = np.array([x[0] for x in val_data])
x_test = np.array([x[0] for x in test_data])
y_train = np.array([label_to_int[x[1]] for x in train_data])
y_val = np.array([label_to_int[x[1]] for x in val_data])
y_test = np.array([label_to_int[x[1]] for x in test_data])

# Convert the integer labels to binary labels
y_train = np.array([1 if y == 1 else 0 for y in y_train])
y_val = np.array([1 if y == 1 else 0 for y in y_val])
y_test = np.array([1 if y == 1 else 0 for y in y_test])



print(current_images)
print(test_images)
plt.imshow(x_test[0])
plt.show()
print(x_test[0].shape)


# Create the model
vgg_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
vgg_model.trainable = False
    
    
x = Flatten()(vgg_model.output)
x = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(x)
x = Dropout(0.8)(x)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=vgg_model.input, outputs=predictions)

# Compile the model
opt = Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy',tf.keras.metrics.AUC(name='auc')])

def exponential_decay(lr0, s):
    def exponential_decay_fn(epoch):
        return lr0 * 0.1 **(epoch / s)
    return exponential_decay_fn

exponential_decay_fn = exponential_decay(0.01, 5)

early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=10,
                                                     restore_best_weights=True)
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(exponential_decay_fn)

# Train the model with your dataset
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=32, epochs=5, shuffle=True, callbacks=[lr_scheduler, early_stopping_cb])

# Evaluate the model on the test set
test_loss, test_acc, test_auc = model.evaluate(x_test, y_test)


# Plot the training and validation accuracy over epochs
plt.plot(history.history['auc'])
plt.plot(history.history['val_auc'])
plt.title('Model AUC')
plt.xlabel('Epoch')
plt.ylabel('Auc')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

num_files = len(os.listdir("./TrainingLogs"))
model_saving_path_name = "./TrainingLogs/cnn_model_info_"+str(num_files+1)
if not os.path.exists(model_saving_path_name):
    os.makedirs(model_saving_path_name)
    model.save("./TrainingLogs/cnn_model_info_"+str(num_files+1)+"/model_data.h5")
        
print('Test AUC:', test_acc)    
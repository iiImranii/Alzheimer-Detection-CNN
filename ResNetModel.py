import tensorflow as tf
from keras.applications import ResNet50, ResNet101
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras.layers import Dense, Flatten, Dropout, BatchNormalization
from keras.utils import to_categorical
from keras.models import Model
import numpy as np
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
classes = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]
current_images = np.zeros(len(classes))
test_images = np.zeros(len(classes))
label_to_int = {label: i for i, label in enumerate(classes)}

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
        else:
            dataset.append((augmented_image/255., i_class))
            current_images[label_to_int[i_class]] += 1
        
        # If we've generated the desired number of images, break the loop
        if len(augmented_images) == 20:
            break
        
        
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


for data_types in os.listdir(data_direc):
    #Only training and validation need to be put into the model for training
        if data_types == "train" or data_types == "test":
            for image_classes in os.listdir(os.path.join(data_direc, data_types)):
                #Checking whether class is authentic
                if image_classes in classes:
                    
                    print("Image Class: "+image_classes+" is being loaded")
                    for images in os.listdir(os.path.join(data_direc, data_types, image_classes)):
                        path_to_image = os.path.join(data_direc, data_types, image_classes, images)
                        try:
                            img = cv2.imread(path_to_image)
                            img = pad_image(img, 128)
                            if data_types == "train":
                                if current_images[label_to_int[image_classes]] <= 1000:
                                    dataset.append((img/255., image_classes))
                                    current_images[label_to_int[image_classes]] += 1
                                    
                                    if image_classes == "ModerateDemented" or image_classes == "MildDemented": #Classes with least samples
                                        imge, aug_arr, num_of_aug = return_augmented_versions(img, image_classes, False)
                            else:
                                 if test_images[label_to_int[image_classes]] <= 500:
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

y_train = to_categorical(y_train, num_classes=4)
y_val = to_categorical(y_val, num_classes=4)
y_test= to_categorical(y_test, num_classes=4)

#x_train = tf.keras.utils.normalize(x_train, axis=1)
#x_val = tf.keras.utils.normalize(x_val, axis=1)
#x_test = tf.keras.utils.normalize(x_test, axis=1)

plt.imshow(x_test[0])
plt.show()
print(x_test[0].shape)
print(current_images)
print(test_images)
max_val = np.max(x_train)

print("Maximum value in x_train:", max_val)

# Load the pre-trained VGG16 model without the top layer
vgg_model = ResNet101(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))

print(vgg_model.layers)
# Freeze the layers of the pre-trained model
for layer in vgg_model.layers[:-4]:
    layer.trainable = False

# Add a new top layer for classification
x = Flatten()(vgg_model.output)
x = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(x)
x = Dropout(0.8)(x)
predictions = Dense(num_classes, activation='softmax')(x)

# Create the new model with the pre-trained VGG16 and new top layers
model = Model(inputs=vgg_model.input, outputs=predictions)

# Compile the model with an optimizer, loss function, and metrics
opt = Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy',tf.keras.metrics.AUC(name='auc')])

# Train the model with your dataset
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=32, epochs=10, shuffle=True)

# Evaluate the model on your test set
test_loss, test_auc, test_acc = model.evaluate(x_test, y_test)


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
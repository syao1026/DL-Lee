import keras  
#from keras.applications.vgg16 import VGG16  
from keras.preprocessing import image  
from keras.applications.vgg16 import preprocess_input  
from keras.preprocessing.image import ImageDataGenerator  
import numpy as np  
import vgg16_train 
import my_utils
import visualization 



path_to_train = "F:\\CV\\DL-Lee\\HW3\\data\\train.csv"
train_rate = 0.8
img_rows, img_cols = 48, 48
#num_classes = 7
train_data, train_result, test_data, test_result, num_classes = my_utils.load_data(path_to_train, train_rate)
#train_data, test_data = normalize(train_data, test_data)
train_data = train_data.reshape(train_data.shape[0], img_rows, img_cols, 1)
test_data = test_data.reshape(test_data.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)
batch_size = 50
epochs = 1

VGG16 = vgg16_train.Vgg16(train_data,train_result, input_shape, num_classes, batch_size, epochs)
model = VGG16.net()
score = model.evaluate(test_data, test_result, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
model.save('../models/model2.h5')

layer_name = 'conv2d_2'
layer_dict = dict([(layer.name, layer) for layer in model.layers])
visualization.visualize_weight(model, layer_name, layer_dict)

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 11:56:19 2018

@author: Shiyao Han
"""
import numpy as np
from keras import backend as K 
import matplotlib.pyplot as plt


## filter visualization
# util function to convert a tensor into a valid image
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    #x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def vis_filter(model, layer_name, layer_dict):
    weights = layer_dict[layer_name].get_weights()
    filters = []
    for filter_index in range(weights[0].shape[3]):
        w = weights[:, :, :, filter_index]
        filters.append(w)
    plot_x, plot_y = 5,5
    fig, ax = plt.subplots(plot_x, plot_y, figsize = (12, 12))
    fig.suptitle('Input image and %s filters' % (layer_name,))
    fig.tight_layout(pad = 0.3, rect = [0, 0, 0.9, 0.9])
    for (x, y) in [(i, j) for i in range(plot_x) for j in range(plot_y)]:
        ax[x, y].imshow(filters[x * plot_y + y - 1], interpolation="nearest", cmap = 'gray')
        ax[x, y].set_title('filter %d' % (x * plot_y + y - 1))
        
        
def visualize_weight(model, layer_name, layer_dict):
    weights = []
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    plot_x, plot_y = 5,5
    fig, ax = plt.subplots(plot_x, plot_y, figsize = (8, 8))
    fig.suptitle('%s filters' % (layer_name,))
#    fig.tight_layout(pad = 0, rect = [0, 0, 1, 1])
    layer = layer_dict[layer_name]
    weights = layer.get_weights()[0] # list of numpy arrays
    #    biases = layer.get_weights()[1]
    for (x, y) in [(i, j) for i in range(plot_x) for j in range(plot_y)]:
        ax[x, y].imshow(weights[:,:,1, y + x * 5].reshape(3, 3), interpolation="nearest", cmap = 'gray')
#        ax[x, y].set_title('filter %d' % (x * plot_y + y - 1))
    



#def vis_img_in_filter(img = np.array(X_train[0]).reshape((1, 28, 28, 1)).astype(np.float64), 
#                      layer_name = 'conv2d_2'):
def vis_img_in_filter(img, model, layer_name, layer_dict):
#    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer_output = layer_dict[layer_name].output
    img_ascs = list()
    for filter_index in range(layer_output.shape[3]):
        # build a loss function that maximizes the activation
        # of the nth filter of the layer considered
        loss = K.mean(layer_output[:, :, :, filter_index])

        # compute the gradient of the input picture wrt this loss
        grads = K.gradients(loss, model.input)[0]

        # normalization trick: we normalize the gradient
        grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

        # this function returns the loss and grads given the input picture
        iterate = K.function([model.input], [loss, grads])

        # step size for gradient ascent
        step = 5.

        img_asc = np.array(img)
        # run gradient ascent for 20 steps
        for i in range(20):
            loss_value, grads_value = iterate([img_asc])
            img_asc += grads_value * step

        img_asc = img_asc[0]
        img_ascs.append(deprocess_image(img_asc).reshape((48, 48)))
        
    if layer_output.shape[3] >= 35:
        plot_x, plot_y = 6, 6
    elif layer_output.shape[3] >= 23:
        plot_x, plot_y = 4, 6
    elif layer_output.shape[3] >= 11:
        plot_x, plot_y = 2, 6
    else:
        plot_x, plot_y = 1, 2
    fig, ax = plt.subplots(plot_x, plot_y, figsize = (12, 12))
    ax[0, 0].imshow(img.reshape((48, 48)), cmap = 'gray')
    ax[0, 0].set_title('Input image')
    fig.suptitle('Input image and %s filters' % (layer_name,))
    fig.tight_layout(pad = 0.3, rect = [0, 0, 0.9, 0.9])
    for (x, y) in [(i, j) for i in range(plot_x) for j in range(plot_y)]:
        if x == 0 and y == 0:
            continue
        ax[x, y].imshow(img_ascs[x * plot_y + y - 1], cmap = 'gray')
        ax[x, y].set_title('filter %d' % (x * plot_y + y - 1))


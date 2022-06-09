import numpy as np
#import tensorflow as tf
#import tensorflow.contrib.eager as tfe
from sklearn.metrics import confusion_matrix
import math
import os
#print("Test")
import cv2
#print("Hello")
import torchvision.transforms as tvt
#import torchvision
import torch
#import torch.nn as nn
import torch.nn.functional as F
#from torchvision import datasets, models, transforms
#from torch.utils.data import DataLoader, Dataset
#import pdb

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
device = get_default_device()

# Prints the number of parameters of a model
#def get_params(model):
    # Init models (variables and input shape)
#    total_parameters = 0
#    for variable in model.variables:
#        # shape is an array of tf.Dimension
#        shape = variable.get_shape()
#        variable_parameters = 1
#        for dim in shape:
#            variable_parameters *= dim.value
#        total_parameters += variable_parameters
#    print("Total parameters of the net: " + str(total_parameters) + " == " + str(total_parameters / 1000000.0) + "M")


# preprocess a batch of images
#def preprocess(x, mode='imagenet'):
#   return x

# applies to a lerarning rate tensor (lr) a decay schedule, the polynomial decay
def lr_decay(lr, init_learning_rate, end_learning_rate, epoch, total_epochs, power=0.9):
    #lr.assign((init_learning_rate - end_learning_rate) * math.pow(1 - epoch / 1. / total_epochs, power) + end_learning_rate)
    lr = (init_learning_rate - end_learning_rate) * math.pow(1 - epoch / 1. / total_epochs, power) + end_learning_rate
    return lr


# converts a list of arrays into a list of tensors
def convert_to_tensors(list_to_convert):
    if list_to_convert != []:
        #return ([torch.from_numpy(item).float() for item in list_to_convert[0]] + convert_to_tensors(list_to_convert[1:]))
        return ([torch.from_numpy(list_to_convert[0])] + convert_to_tensors(list_to_convert[1:]))
    else:
        return []
            

# restores a checkpoint model
#def restore_state(saver, checkpoint):
#    try:
#        saver.restore(checkpoint)
#        model2.load_state_dict(torch.load('cifar10-cnn.pth'))
#        print('Model loaded')
#    except Exception as e:
#        print('Model not loaded: ' + str(e))

# inits a models (set input)
#def init_model(model, input_shape):
#    model._set_inputs(np.zeros(input_shape))


# Erase the elements if they are from ignore class. returns the labesl and predictions with no ignore labels
def erase_ignore_pixels(labels, predictions, mask):
    #indices = tf.squeeze(tf.where(tf.greater(mask, 0)))
    indices=[]
    for idx in range(len(mask)):
        if mask[idx]>0:
            indices.append(idx)

    #indices = torch.squeeze(torch.where(torch.gt(mask, 0)))  # not ignore labels
    #labels = tf.cast(tf.gather(labels, indices), torch.int64)
    labels = labels[indices]
    labels = labels.to(torch.int64)
    predictions = predictions[indices]

    return labels, predictions

# generate and write an image into the disk
def generate_image(image_scores, output_dir, dataset, loader, train=False):
    # Get image name
    if train:
        list = loader.image_train_list
        index = loader.index_train
    else:
        list = loader.image_test_list
        index = loader.index_test

    dataset_name = dataset.split('/')
    if dataset_name[-1] != '':
        dataset_name = dataset_name[-1]
    else:
        dataset_name = dataset_name[-2]

    # Get output dir name
    out_dir = os.path.join(output_dir, dataset_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # write it
    image = np.argmax(image_scores, 2)
    name_split = list[index - 1].split('/')
    name = name_split[-1].replace('.jpg', '.png').replace('.jpeg', '.png')
    cv2.imwrite(os.path.join(out_dir, name), image)

def inference(model, batch_images, n_classes, flip_inference=True, scales=[1]):

    model.eval()


    #x = preprocess(batch_images, mode=preprocess_mode)
    x = batch_images
    #x = x.permute(0, 3, 2, 1)


    # creates the variable to store the scores
    y_ = convert_to_tensors([np.zeros((x.shape[0], n_classes, x.shape[2], x.shape[3]), dtype=np.float32)])[0]
    y_ = y_.to(device)
    #print("1")

    for scale in scales:
        # scale the image
        
        #x_scaled = F.interpolate(x, size=(x.shape[2] * scale, x.shape[3] * scale), mode='bilinear', align_corners=True).to(device) 
        #print('a:', x.shape)
        #y_scaled = model(x_scaled) 
        y_=model(x) # no
        #  rescale the output
        #print("3")
        #y_scaled = F.interpolate(y_scaled, size=(x.shape[2], x.shape[3]),mode='bilinear', align_corners=True).to(device) 
        #print("4")
        y_= F.softmax(y_, dim=1) #no
        #x_scaled = x #no
        #y_scaled = y_ #no
        # get scores
        #y_scaled = F.softmax(y_scaled, dim=1) 

        if flip_inference:
            # calculates flipped scores
            hflipper = tvt.RandomHorizontalFlip(p=1.0)
            x_scaled_flipped = hflipper(x_scaled)
            print("5")
            #y_flipped_ = tf.image.flip_left_right(model(tf.image.flip_left_right(x_scaled), training=False))
            y_flipped_ = hflipper(model(x_scaled_flipped#, training=False
            ))
            # resize to rela scale
            y_flipped_ = F.interpolate(y_flipped_, size = (x.shape[2], x.shape[3]),
                                                mode='bilinear', align_corners=True)
            print("7")
            # get scores
            y_flipped_score = F.softmax(y_flipped_, dim=1)

            y_scaled += y_flipped_score
            print("6")

        #y_ += y_scaled
        y_ += y_

    return y_

# get accuracy and miou from a model
def get_metrics(loader, model, n_classes, train=True, flip_inference=False, scales=[1], write_images=False#,preprocess_mode=None
):
    if train:
        loader.index_train = 0
    else:
        loader.index_test = 0

    

    #accuracy = tfe.metrics.Accuracy()
    conf_matrix = np.zeros((n_classes, n_classes))
    if train:
        samples = len(loader.image_train_list)
        print("Samples:", samples)
    else:
        samples = len(loader.image_test_list)
        print("Samples:", samples)
        bs=8
        steps_per_epoch = int((samples / bs) + 1)
    for step in range(steps_per_epoch):  # for every batch
        x, y, mask = loader.get_batch(size=bs, train=train, augmenter=False)

        [y, mask] = convert_to_tensors([y, mask])
        mask = np.transpose(mask, (0, 2, 1))
        y= np.transpose(y, (0,3,2,1))
        [x] = convert_to_tensors([x])     
        x = np.transpose(x, (0, 3, 2, 1))
        
        

        x= x.to(device)
        y = y.to(device)
        mask =mask.to(device)

        #print("hey")

        y_ = inference(model, x, n_classes, flip_inference, scales#, preprocess_mode=preprocess_mode
        )
        #print("inference done")
        # generate images
        if write_images:
            generate_image(y_[0,:,:,:], 'images_out', loader.dataFolderPath, loader, train)

        # Rephape
        y = torch.reshape(y, [y.shape[3] * y.shape[2] * y.shape[0], y.shape[1]])
        y_ = torch.reshape(y_, [y_.shape[3] * y_.shape[2] * y_.shape[0], y_.shape[1]])
        mask = torch.reshape(mask, [mask.shape[1] * mask.shape[2] * mask.shape[0]])

        #pdb.set_trace()

        max_probs1, labels = torch.max(y, 1)
        max_probs2, predictions = torch.max(y_, 1)

        #labels, predictions = erase_ignore_pixels(labels=tf.argmax(y, 1), predictions=tf.argmax(y_, 1), mask=mask)
        labels, predictions = erase_ignore_pixels(labels, predictions, mask=mask)
        
        #accuracy = (predictions == labels).sum().item() / predictions.size(0)
        accuracy = torch.tensor(torch.sum(predictions == labels).item() / len(predictions))

        #accuracy(labels, predictions)
        conf_matrix += confusion_matrix(labels.to('cpu').numpy(), predictions.to('cpu').numpy(), labels=range(0, n_classes))

    # get the train and test accuracy from the model
    #return accuracy.result(), compute_iou(conf_matrix)
    #print("get metrics done")
    return accuracy, compute_iou(conf_matrix)

# computes the miou given a confusion amtrix
def compute_iou(conf_matrix):
    intersection = np.diag(conf_matrix)
    ground_truth_set = conf_matrix.sum(axis=1)
    predicted_set = conf_matrix.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection
    IoU = intersection / union.astype(np.float32)
    IoU[np.isnan(IoU)] = 0
    print(IoU)
    miou = np.mean(IoU)
    '''
    print(ground_truth_set)
    miou_no_zeros=miou*len(ground_truth_set)/np.count_nonzero(ground_truth_set)
    print ('Miou without counting classes with 0 elements in the test samples: '+ str(miou_no_zeros))
    '''
    return miou

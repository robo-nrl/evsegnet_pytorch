import os
os.environ['CUDA_VISIBLE_DEVICES']="2,"

import numpy as np
#import tensorflow as tf
#import tensorflow.contrib.eager as tfe
import torch
import torch.nn.functional as F
#import torchvision
import os
import sys
sys.path.insert(0, '/home/min/a/sdasbisw/Desktop/PROJECTS/evcoop/Ev-SegNet-master/')
import nets.Network_1 as Segception
import utils.Loader_pytorch as loader
from utils.utils_pytorch_bsforinf import lr_decay, convert_to_tensors, get_metrics
#from utils.utils import , restore_state, init_model, preprocess, get_params, 
import argparse
import shutil
#import matplotlib.pyplot as plt
#from torchvision.utils import make_grid



torch.cuda.empty_cache()

# enable eager mode
#tf.enable_eager_execution()
#tf.set_random_seed(7)
random_seed = 1 # or any of your favorite number 
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
#torch.backends.cudnn.enabled = False 
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)

from datetime import datetime


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
device = get_default_device()

#device ='cuda:2'
print("device:", device)
print("GPU card number:", torch.cuda.current_device())

def save_checkpoint(state, save_best_model, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(save_best_model,filename))
    shutil.copyfile(os.path.join(save_best_model,filename), os.path.join(save_best_model,'model_best.pth.tar'))
    #name_best_model = os.path.join(save_path,'model_best.pth.tar')

# Trains the model for certains epochs on a dataset
def train(loader, model, epochs=5, batch_size=2, show_loss=False, augmenter=None, lr=None, init_lr=2e-4,
          ##saver=None, variables_to_optimize=None, 
          evaluation=True, save_best_model='/home/min/a/sdasbisw/Desktop/PROJECTS/evcoop/Ev-SegNet-master/results'#, preprocess_mode=None
          ):
    training_samples = len(loader.image_train_list)
    steps_per_epoch = int((training_samples / batch_size) + 1)
    #steps_per_epoch=4
    print("Steps per epoch:",steps_per_epoch)
    best_miou = 0
    #model = model.to(device)
    init_lr = 1

    #print("hi")
    

   

    for epoch in range(epochs):  # for each epoch
        model.train()
        lr = lr_decay(lr, init_lr, 1e-2, epoch, epochs - 1)  # compute the new lr
        print('epoch: ' + str(epoch) + '. Learning rate: ' + str(lr))
        acc_loss = 0
        
        for step in range(steps_per_epoch):  # for every batch
            #print("step:",step)
            optimizer = torch.optim.Adam(model.parameters(), lr)
            optimizer.zero_grad()
            #with tf.GradientTape() as g:
            # get batch
            x, y, mask = loader.get_batch(size=batch_size, train=True, augmenter=augmenter)
            print("x", x.shape)
            print("y", y.shape)

            x= np.transpose(x, (0,3,2,1))
            y= np.transpose(y, (0,3,2,1))
            mask = np.transpose(mask, (0, 2, 1))

            #x = preprocess(x, mode=preprocess_mode)
            [x, y, mask] = convert_to_tensors([x, y, mask])

            x= x.to(device)
            y = y.to(device)
            mask =mask.to(device)

            
            #print("shape mask:", mask.shape)

            
            #x= x.to('cuda:2')
            #y = y.to('cuda:2')
            #mask =mask.to('cuda:2')

            #print("Batch loaded to GPU")

            y_, aux_y_ = model(x,# training=True, 
            aux_loss=True)  # get output of the model

            print(y.shape)
            #print("y:", y.dtype)

            #print(y_[:0])
            print(y_.shape)
            #print("y_:", y_.dtype)

            #loss = tf.losses.softmax_cross_entropy(y, y_, weights=mask)  # compute loss
            #l = torch.nn.CrossEntropyLoss(weight = mask)
            def l(y, y_):
                y1 = torch.reshape(y, [y.shape[3] * y.shape[2] * y.shape[0], y.shape[1]])
                #print(y1.shape)
                y1_ = torch.reshape(y_, [y_.shape[3] * y_.shape[2] * y_.shape[0], y_.shape[1]])
                loss = F.cross_entropy(y1_, y1)
                print("y1;",y1[:0])
                print("y1_;",y1_[:0])
                #loss = -torch.sum(F.log_softmax(y1_, dim=1) * y1, dim=1)
                #loss = torch.sum(- y1 * F.log_softmax(y1_, -1), -1)
                #mean_loss = loss.mean()
                #print("mean:", mean_loss.item())
                #return mean_loss
                return loss


            loss = l(y, y_)  # compute loss
            loss_aux = l(y, aux_y_)  # compute loss
            loss = 1*loss + 0.8*loss_aux
            acc_loss +=loss.item()

            #print('Training loss for step '+ str(step) + ' : ' + str(loss.item()))
            

            # Gets gradients and applies them
            loss.backward()
            optimizer.step()
            #grads = g.gradient(loss, variables_to_optimize)
            #optimizer.apply_gradients(zip(grads, variables_to_optimize))
            #print('Training loss: ' + str(loss.item()))
        epoch_loss = acc_loss / steps_per_epoch
        print("Loss for epoch " + str(epoch+1) + ' / ' + str(epochs) + " : " + str(epoch_loss))
            
        if ((epoch+1)%100==0):    
            if evaluation:
                model.eval()
                print("Calculating validation metrics")
                with torch.no_grad():
                # get metrics
                #print("Calculating training metrics")
                #train_acc, train_miou = get_metrics(loader, model, loader.n_classes, train=True, #preprocess_mode=preprocess_mode
                #)
                #print('Train accuracy: ' + str(train_acc))
                #print('Train miou: ' + str(train_miou))
                # print('') 

                
                    test_acc, test_miou = get_metrics(loader, model, loader.n_classes, train=False, flip_inference=False,scales=[1]#, preprocess_mode=preprocess_mode
                )


                print('Validation accuracy: ' + str(test_acc))
                print('Validation miou: ' + str(test_miou))
                print('')

                # save model 
                if test_miou > best_miou:
                    print('Best model updated.')
                    print('=> Everything will be saved to {}'.format(args.save_best_model))
                    best_miou = test_miou
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_miou': best_miou
                        }, save_best_model)

        loader.suffle_segmentation()  # shuffle training set


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Dataset path", default='/home/min/a/sdasbisw/Desktop/PROJECTS/evcoop/Ev-SegNet-master/dataset_our_codification/')
    parser.add_argument("--model_path", help="pretrained", default='/home/min/a/sdasbisw/Desktop/PROJECTS/evcoop/Ev-SegNet-master/results/model_best.pth.tar')
    parser.add_argument("--save_best_model",help="Save results path",default='/home/min/a/sdasbisw/Desktop/PROJECTS/evcoop/Ev-SegNet-master/results_2')
    parser.add_argument("--n_classes", help="number of classes to classify", default=6)
    parser.add_argument("--batch_size", help="batch size", default=8)
    parser.add_argument("--epochs", help="number of epochs to train", default=500)
    parser.add_argument("--width", help="number of epochs to train", default=352)
    parser.add_argument("--height", help="number of epochs to train", default=224)
    parser.add_argument("--lr", help="init learning rate", default=1e-3)
    parser.add_argument("--n_gpu", help="number of the gpu", default=2)
    args = parser.parse_args()

    n_gpu = int(args.n_gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(n_gpu)

    num_classes = int(args.n_classes)
    batch_size = int(args.batch_size)
    epochs = int(args.epochs)
    width =  int(args.width)
    height =  int(args.height)
    lr = float(args.lr)

    channels = 6 # input of 6 channels
    channels_image = 0
    channels_events = channels - channels_image
    #folder_best_model = args.model_path
    save_best_model =  args.save_best_model
    pretrained_model = args.model_path
    #name_best_model = os.path.join(folder_best_model,'best')
    dataset_path = args.dataset
       
    if not os.path.exists(save_best_model):
        os.makedirs(save_best_model)


    loader = loader.Loader(dataFolderPath=dataset_path, n_classes=num_classes, problemType='segmentation',width=width, height=height, channels=channels_image, channels_events=channels_events)

 
    # build model and optimizer
    #import pdb; pdb.set_trace()
    model = Segception.Segception_small(num_classes=num_classes, weights=None, input_shape=(channels, None, None))
    print("model created")
    start_time = datetime.now()
    model = model.to(device)
    #model = model.to('cuda:0')
    print("model moved to GPU")
    #print(model)
    end_time = datetime.now()
    print('Duration for model transfer: {}'.format(end_time - start_time))
    #print(model.parameters())
    #model = torch.nn.DataParallel(model).to(device)
    #print("to device")
    # optimizer
    #learning_rate = tfe.Variable(lr)
    learning_rate= lr
    #optimizer = tf.train.AdamOptimizer(learning_rate)
    #optimizer = torch.optim.Adam(model.parameters(), learning_rate)

    # Init models (optional, just for get_params function)
    #init_model(model, input_shape=(batch_size, width, height, channels))

    #variables_to_restore = model.variables #[x for x in model.variables if 'block1_conv1' not in x.name]
    #variables_to_save = model.variables
    #variables_to_optimize = model.variables

    # Init saver. can use also ckpt = tfe.Checkpoint((model=model, optimizer=optimizer,learning_rate=learning_rate, global_step=global_step)
    #saver_model = tfe.Saver(var_list=variables_to_save)
    #restore_model = tfe.Saver(var_list=variables_to_restore)
    

    # restore if model saved and show number of params
    #restore_state(restore_model, name_best_model)
    try:
        if not pretrained_model:
            raise Exception("You need to pass pretrained model path")

        model_data = torch.load(pretrained_model)
        #model = model(args).to(args.device)
        model.load_state_dict(model_data['state_dict'])
        #get_params(model)
    except Exception as e:
        print('Model not loaded: ' + str(e))


    train(loader=loader, model=model, epochs=epochs, batch_size=batch_size, augmenter='segmentation', lr=learning_rate,
          init_lr=lr, #saver=saver_model, variables_to_optimize=variables_to_optimize, 
          save_best_model=save_best_model,
          evaluation=True
          #, 
          #preprocess_mode=None
          )

    # Test best model
    print('Testing model')
    with torch.no_grad():
        test_acc, test_miou = get_metrics(loader, model, loader.n_classes, train=False, flip_inference=True, scales=[1, 0.75, 1.5],
                                      write_images=False)
    print('Test accuracy: ' + str(test_acc))
    print('Test miou: ' + str(test_miou))

    #train_acc, train_miou = get_metrics(loader, model, loader.n_classes, train=True, preprocess_mode=preprocess_mode)
    #print('Train accuracy: ' + str(train_acc.numpy()))
    #print('Train miou: ' + str(train_miou))

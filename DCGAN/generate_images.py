import os, time
import torch
import torch.optim as optim
from torch.autograd import Variable
import logging
import utils
from model import netChairs as net
import numpy as np
import scipy.misc

'''
In this file I will generate 30,000 images from a trained GAN model
for further estimation of their inception score.
'''

def load_model(model_dir, restore_file, D_model, G_model):
    restore_path = os.path.join(model_dir, restore_file + '.pth.tar')
    logging.info("Restoring parameters from {}".format(restore_path))
    utils.load_checkpoint(restore_path, D_model, G_model)

def generate_images(N, path):

    if not os.path.exists(path):
        print("Directory for generated images does not exist! Making directory {}".format(output_folder))
        os.mkdir(output_folder)
    else:
        print("Directory for generated images exists! ")

    #examples = torch.randn((N, 100)).view(-1, 100, 1, 1)
    examples = torch.FloatTensor(N, 100, 1, 1).normal_(0, 1)
    examples = examples.cuda() if param_cuda else examples
    examples = Variable(examples, volatile=True)
    G_model.eval()
    test_images = G_model(examples)
    for i in range(N):
        test_images[i]
        #print(np.shape(test_images[i]))
        scipy.misc.toimage(test_images[i].data.cpu().numpy()).save(path+'/gen_image_'+str(i)+'.jpg')
        if i%1000 == 0:
            print('1000 images added')


if __name__ == '__main__':


    # numer of images to generate'
    images = 30000

    # folder and file with saved weights of the model
    model_dir = 'imagenet_folder'
    restore_file = 'last'

    # file to save the generated images
    output_folder = 'generated_images_30k_imagenet'

    # use GPU if available
    param_cuda = torch.cuda.is_available()
    if param_cuda:
            print('Im using cuda')
    else:
        print('Im not using cuda')

    # Define the models
    G_model = net.generator(128).cuda() if param_cuda else net.generator(128)
    D_model = net.discriminator(128).cuda() if param_cuda else net.discriminator(128)

    # load model parameters from file.
    load_model(model_dir, restore_file, D_model, G_model)

    generate_images(images, output_folder)


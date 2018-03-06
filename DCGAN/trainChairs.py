import os, time
import matplotlib.pyplot as plt
import itertools
import pickle
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import logging
from tqdm import tqdm
import utils
import model.netChairs as net
import model.data_loader as data_loader
import matplotlib.colors as colors
import numpy as np
from PIL import Image
#from evaluate import evaluate
import scipy.misc
import torch.nn.functional as F


def train(G_model, D_model, G_optimizer, D_optimizer, loss_fn, train_loader, metrics):

    # set models to training mode
    #G_model.train()
    #D_model.train()

    # Use tqdm for progress bar
    with tqdm(total=len(train_loader)) as t:

        for inputs_real in train_loader:

            # move to GPU if available


            # train discriminator D:

            # define real and fake inputs and labels, convert to variables
            mini_batch = inputs_real.size()[0]
            labels_real = torch.ones(mini_batch)

            labels_fake = torch.zeros(mini_batch)
            #inputs_fake = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)

            inputs_fake = torch.FloatTensor(mini_batch, 100, 1, 1).normal_(0, 1)
            #print(inputs_fake)

            inputs_real, labels_real, inputs_fake, labels_fake = Variable(inputs_real), Variable(labels_real),Variable(inputs_fake), Variable(labels_fake)


            # compute D_model with real input, and compute loss
            D_output = D_model(inputs_real).squeeze()
            D_model_real_loss = loss_fn(D_output, labels_real)

            # compute D_model with fake input from generator, and compute loss
            G_output = G_model(inputs_fake)

            D_output = D_model(G_output).squeeze()
            D_model_fake_loss = loss_fn(D_output, labels_fake)

            # D_model train loss as the sum of real + fake:
            #D_model_fake_score = D_model.data.mean()
            D_model_train_loss = D_model_real_loss + D_model_fake_loss
            # clear previous gradients, compute gradients of all variables wrt train loss
            D_model.zero_grad()
            D_model_train_loss.backward()
            D_optimizer.step()


            # train generator G:


            #create new fake inputs, convert to variables
            inputs_fake = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
            inputs_fake = Variable(inputs_fake)

            # compute D_model with fake input from generator, and compute loss
            G_output = G_model(inputs_fake)
            D_output = D_model(G_output).squeeze()
            G_model_train_loss = loss_fn(D_output, labels_real)

            # clear previous gradients, compute gradients of all variables wrt loss
            G_model.zero_grad()
            G_model_train_loss.backward()
            G_optimizer.step()

            print("D loss: " + str(D_model_train_loss.data[0]))
            print("G loss: " + str(G_model_train_loss.data[0]))

            t.update()

        #return actual losses
        return D_model_train_loss.data[0], G_model_train_loss.data[0]


def train_and_evaluate(G_model, D_model, G_optimizer, D_optimizer, loss_fn, train_loader, metrics, train_epoch, model_dir, restore_file=None):
    """Train the model and evaluate every epoch"""

    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(model_dir, restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, D_model, G_model, D_optimizer, G_optimizer)

    print('training start!')
    start_time = time.time()

    D_model_losses = []
    G_model_losses = []
    train_hist = {}
    train_hist['D_model_losses'] = []
    train_hist['G_model_losses'] = []
    train_hist['per_epoch_ptimes'] = []
    train_hist['total_ptime'] = []
    num_iter = 0

    # results save folder
    if not os.path.isdir('Chairs_results'):
        os.mkdir('Chairs_results')
    if not os.path.isdir('Chairs_results/Random_results'):
        os.mkdir('Chairs_results/Random_results')
    if not os.path.isdir('Chairs_results/Fixed_results'):
        os.mkdir('Chairs_results/Fixed_results')

    print('length dataloader ' + str(len(train_loader)))

    for epoch in range(train_epoch):

        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, train_epoch))
        epoch_start_time = time.time()

        # compute number of batches in one epoch (one full pass over the training set)
        D_model_loss, G_model_loss = train(G_model, D_model, G_optimizer, D_optimizer, loss_fn, train_loader, metrics)
        D_model_losses.append(D_model_loss)
        G_model_losses.append(G_model_loss)

        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time

        #prints in every epoch:
        print("iteration number "+str(epoch))
        print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime, torch.mean(torch.FloatTensor(D_model_losses)), torch.mean(torch.FloatTensor(G_model_losses))))

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'D_model_state_dict': D_model.state_dict(),
                               'G_model_state_dict': G_model.state_dict(),
                               'D_optim_dict': D_optimizer.state_dict(),
                               'G_optim_dict': G_optimizer.state_dict()},
                               is_best=False,
                               checkpoint = model_dir)


        #save test pictures after every epoch:
        p = 'Chairs1_results/Random_results/LSUN_DCGAN_' + str(epoch + 1) + '.png'
        fixed_p = 'Chairs1_results/Fixed_results/LSUN_DCGAN_' + str(epoch + 1) + '.png'
        show_result((epoch+1), save=True, path=p, isFix=False)
        show_result((epoch+1), save=True, path=fixed_p, isFix=True)
        train_hist['D_model_losses'].append(torch.mean(torch.FloatTensor(D_model_losses)))
        train_hist['G_model_losses'].append(torch.mean(torch.FloatTensor(G_model_losses)))
        train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

    end_time = time.time()
    total_ptime = end_time - start_time
    train_hist['total_ptime'].append(total_ptime)

    print("Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f" % (torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), train_epoch, total_ptime))
    print("Training finish!... save learned parameters")
    torch.save(G_model.state_dict(), "Chairs_results/generator_param.pkl")
    torch.save(D_model.state_dict(), "Chairs_results/discriminator_param.pkl")
    with open('Chairs_results/train_hist.pkl', 'wb') as f:
        pickle.dump(train_hist, f)


def show_result(num_epoch, show = False, save = False, path = 'result.png', isFix=False):

    fixed_z_ = torch.randn((5 * 5, 100)).view(-1, 100, 1, 1)    # fixed noise
    fixed_z_ = Variable(fixed_z_, volatile=True)

    z_ = torch.randn((5*5, 100)).view(-1, 100, 1, 1)
    z_ = Variable(z_, volatile=True)

    G_model.eval()
    if isFix:
        test_images = G_model(fixed_z_)
    else:
        test_images = G_model(z_)
    G_model.train()
    print(test_images.size())
    size_figure_grid = 5
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(5*5):
        i = k // 5
        j = k % 5
        #ax[i, j].cla()
        print(test_images[k].shape)
        plot_image = test_images[k].data.cpu().numpy()
        print(np.shape(plot_image))
        plot_image = (plot_image - plot_image.min())/(plot_image.max() - plot_image.min())
        plot_image = np.transpose(plot_image,(1, 2, 0))
        ax[i, j].imshow(plot_image)

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')
    plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()



if __name__ == '__main__':

    # training
    batch_size = 128
    lr = 0.0002
    train_epoch = 200
    data_dir = 'new_images'
    dataset = "chairs"
    model_dir = 'model_folder'

    print(dataset)
    # Define the models
    G_model = net.generator(128)
    D_model = net.discriminator(128)

    #Initialize weights
    G_model.weight_init(mean=0.0, std=0.02)
    D_model.weight_init(mean=0.0, std=0.02)

    #G_model.cuda()
    #D_model.cuda()

    #Define optimizers
    G_optimizer = optim.Adam(G_model.parameters(), lr=lr, betas=(0.5, 0.999))
    D_optimizer = optim.Adam(D_model.parameters(), lr=lr, betas=(0.5, 0.999))

    # fetch loss function and metrics
    loss_fn = net.loss_fn
    metrics = net.metrics

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    # data_loader
    train_loader = data_loader.fetch_dataloader(data_dir, batch_size, dataset)
    print('dataset length '+str(len(train_loader.dataset)))

    ''' try to plot images directly from files and also from pythorch dataset
    
    filenames = os.listdir(data_dir)
    filenames = [os.path.join(data_dir, f) for f in filenames]
    imageX = Image.open(filenames[8])
    imageX = np.asarray(imageX)
    #print(imageX)
    print(type(imageX))
    print(np.shape(imageX))
    plt.imshow(imageX)
    plt.show()
    
    image = train_loader.dataset[8]
    print(type(image))
    image = image.numpy()
    print(image)
    print("image", type(image))
    image = (image - image.min())/(image.max() - image.min())
    image = np.transpose(image,(1,2,0))
    #print(image)
    #print(np.shape(image))
    plt.imshow(image)
    plt.show()
    '''


    # Train the model
    logging.info("Starting training for {} epoch(s)".format(train_epoch))
    train_and_evaluate(G_model, D_model, G_optimizer, D_optimizer, loss_fn, train_loader, metrics, train_epoch, model_dir, restore_file=None)

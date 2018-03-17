import os, time
import pickle
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import logging
from tqdm import tqdm
import utils
from model import netChairs as net
from model import data_loader as data_loader
import numpy as np
from PIL import Image



def train(G_model, D_model, G_optimizer, D_optimizer, loss_fn, train_loader, metrics, param_cuda):

    # set models to training mode
    G_model.train()
    D_model.train()

    # Use tqdm for progress bar
    with tqdm(total=len(train_loader)) as t:

        for inputs_real in train_loader:

            # move to GPU if available
            if param_cuda:
                train_batch = train_batch.cuda(async=True)

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


def train_and_evaluate(dataset, G_model, D_model, G_optimizer, D_optimizer, loss_fn, train_loader, metrics, train_epoch, model_dir, restore_file=None):
    """Train the model and evaluate every epoch"""

    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(model_dir, restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, D_model, G_model, D_optimizer, G_optimizer)

    # check training starting time
    start_time = time.time()

    # here we are going to save the losses.
    D_model_losses = []
    G_model_losses = []
    train_hist = {}
    train_hist['D_model_losses'] = []
    train_hist['G_model_losses'] = []
    train_hist['per_epoch_ptimes'] = []
    train_hist['total_ptime'] = []

    # folder for saving the images
    if not os.path.isdir(dataset + '_results'):
        os.mkdir(dataset + '_results')
    if not os.path.isdir(dataset + '_results/Random_results'):
        os.mkdir(dataset + '_results/Random_results')
    if not os.path.isdir(dataset + '_results/Fixed_results'):
        os.mkdir(dataset + '_results/Fixed_results')


    print('length dataloader ' + str(len(train_loader)))

    for epoch in range(train_epoch):

        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, train_epoch))
        epoch_start_time = time.time()

        # compute number of batches in one epoch (one full pass over the training set)
        D_model_loss, G_model_loss = train(G_model, D_model, G_optimizer, D_optimizer, loss_fn, train_loader, metrics, param_cuda)
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
        p = dataset + '_results/Random_results/pretrain_' + str(epoch + 1) + '.png'
        fixed_p = dataset + '_results/Fixed_results/pretrain_' + str(epoch + 1) + '.png'
        utils.show_result(G_model, (epoch+1), save=True, path=p, isFix=False)
        utils.show_result(G_model, (epoch+1), save=True, path=fixed_p, isFix=True)

        # add losses to the training history
        train_hist['D_model_losses'].append(torch.mean(torch.FloatTensor(D_model_losses)))
        train_hist['G_model_losses'].append(torch.mean(torch.FloatTensor(G_model_losses)))
        train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

    end_time = time.time()
    total_ptime = end_time - start_time
    train_hist['total_ptime'].append(total_ptime)

    print("Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f" % (torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), train_epoch, total_ptime))
    print("Training finish!... save learned parameters")
    torch.save(G_model.state_dict(), dataset + "_results/generator_param.pkl")
    torch.save(D_model.state_dict(), dataset + "_results/discriminator_param.pkl")

    # save training history
    with open(dataset + '_results/train_hist.pkl', 'wb') as f:
        pickle.dump(train_hist, f)
    # plot training history
    utils.show_train_hist(train_hist, save=True, path=dataset + '_results/_train_hist.png')


if __name__ == '__main__':

    # training
    batch_size = 128
    lr = 0.0002
    train_epoch = 2
    #data_dir = 'new_images'
    #dataset = "Chairs"
    #model_dir = 'model_folder'
    data_dir = 'Imagenet'
    dataset = 'Imagenet'
    model_dir = 'imagenet_folder'

    # use GPU if available
    param_cuda = torch.cuda.is_available()

    # Define the models
    G_model = net.generator(128).cuda() if param_cuda else net.generator(128)
    D_model = net.discriminator(128).cuda() if param_cuda else net.discriminator(128)

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

    # Set the logger
    utils.set_logger(os.path.join(model_dir, 'train.log'))

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
    
    image = train_loader.dataset[500]
    print(type(image))
    image = image.numpy()
    print(image)
    print(np.shape(image))
    print("image", type(image))
    image = (image - image.min())/(image.max() - image.min())
    image = np.transpose(image,(1,2,0))
    #print(image)
    #print(np.shape(image))
    plt.imshow(image)
    plt.show()
    '''

    '''
    train_hist = {}
    with open('Chairs_results/train_hist.pkl', 'rb') as input:
        train_hist = pickle.load(input)
    print(train_hist.keys())
    show_train_hist(train_hist, save=True, path='Chairs_results/_train_hist.png')
    '''

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(train_epoch))
    train_and_evaluate(dataset, G_model, D_model, G_optimizer, D_optimizer, loss_fn, train_loader, metrics, train_epoch, model_dir, restore_file=None)

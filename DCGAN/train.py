import os, time
import torch
import torch.optim as optim
from torch.autograd import Variable
import logging
from tqdm import tqdm
import utils
from model import net
#from model import net_chairs_partial_freeze as net
from model import data_loader




def train(G_model, D_model, G_optimizer, D_optimizer, loss_fn, train_loader, param_cuda, train_hist):

    # set models to training mode
    G_model.train()
    D_model.train()

    # save losses in every epoch
    D_model_losses = []
    G_model_losses = []

    # Use tqdm for progress bar
    with tqdm(total=len(train_loader)) as t:

        for i, inputs_real in enumerate(train_loader):

            # move to GPU if available
            inputs_real = inputs_real.cuda(async=True) if param_cuda else inputs_real

            # train discriminator D:

            # define real and fake inputs and labels, convert to variables
            mini_batch = inputs_real.size()[0]
            labels_real = torch.ones(mini_batch).cuda() if param_cuda else torch.ones(mini_batch)
            labels_fake = torch.zeros(mini_batch).cuda() if param_cuda else torch.zeros(mini_batch)
            inputs_fake = torch.FloatTensor(mini_batch, 100, 1, 1).normal_(0, 1)
            inputs_fake = inputs_fake.cuda() if param_cuda else inputs_fake

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

            '''
            Here I can put a threshold to make the discriminator learn slower at first,
            only update if loss is bigger thar a threshold
            if D_model_train_loss.data[0] >= 0.1:
            '''
            D_model.zero_grad()
            D_model_train_loss.backward()
            D_optimizer.step()

            # train generator G:

            #create new fake inputs, convert to variables
            inputs_fake = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
            inputs_fake = inputs_fake.cuda() if param_cuda else inputs_fake
            inputs_fake = Variable(inputs_fake)

            # compute D_model with fake input from generator, and compute loss
            G_output = G_model(inputs_fake)
            D_output = D_model(G_output).squeeze()
            G_model_train_loss = loss_fn(D_output, labels_real)

            # clear previous gradients, compute gradients of all variables wrt loss
            G_model.zero_grad()
            G_model_train_loss.backward()
            G_optimizer.step()

            # add losses to training history of this epoch
            D_model_losses.append(D_model_train_loss.data[0])
            G_model_losses.append(G_model_train_loss.data[0])

            #print losses in console
            print("D loss: " + str(D_model_train_loss.data[0]))
            print("G loss: " + str(G_model_train_loss.data[0]))

            t.update()

        # save the mean loss of every epoch
        train_hist['D_model_mean_losses'].append(torch.mean(torch.FloatTensor(D_model_losses)))
        train_hist['G_model_mean_losses'].append(torch.mean(torch.FloatTensor(G_model_losses)))

        # return losses
        return train_hist

        # return D_model_train_loss.data[0], G_model_train_loss.data[0], train_hist


def train_and_evaluate(param_cuda, dataset, G_model, D_model, G_optimizer, D_optimizer, loss_fn, train_loader, train_epoch, model_dir, restore_file=None):
    '''Train the model and evaluate every epoch'''

    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(model_dir, restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, D_model, G_model, D_optimizer, G_optimizer)

    # check training starting time
    start_time = time.time()

    # here we are going to save the losses of every epoch
    train_hist = {}
    train_hist['D_model_mean_losses'] = []
    train_hist['G_model_mean_losses'] = []
    train_hist['per_epoch_ptimes'] = []
    train_hist['total_ptime'] = []

    # folder for saving the images
    if not os.path.isdir(dataset + '_results'):
        os.mkdir(dataset + '_results')

    for epoch in range(train_epoch):

        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, train_epoch))
        epoch_start_time = time.time()

        # compute number of batches in one epoch (one full pass over the training set)
        train_hist = train(G_model, D_model, G_optimizer, D_optimizer, loss_fn, train_loader, param_cuda, train_hist)

        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time

        #prints in every epoch:
        print("iteration number "+str(epoch))
        print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime, torch.mean(torch.FloatTensor(train_hist['D_model_mean_losses'])), torch.mean(torch.FloatTensor(train_hist['G_model_mean_losses']))))

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'D_model_state_dict': D_model.state_dict(),
                               'G_model_state_dict': G_model.state_dict(),
                               'D_optim_dict': D_optimizer.state_dict(),
                               'G_optim_dict': G_optimizer.state_dict()},
                               is_best=False,
                               checkpoint =  dataset + '_results/')

        #save test pictures after every epoch:
        p = dataset + '_results/result_epoch_' + str(epoch + 1) + '.png'
        utils.show_result(param_cuda, G_model, (epoch+1), p, save=True)

        # add epoch time to the training history
        train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

    end_time = time.time()
    total_ptime = end_time - start_time
    train_hist['total_ptime'].append(total_ptime)

    print("Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f" % (torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), train_epoch, total_ptime))
    print("Training finish!... save learned parameters")

    # plot training history
    utils.show_train_hist(train_hist, save=True, path=dataset + '_results/_train_hist.png')


if __name__ == '__main__':

    # training
    batch_size = 128
    lr = 0.0002
    train_epoch = 10
    data_dir = 'chairs_dataset_3300'

    #dataset: 'Chairs', 'Imagenet', 'MNIST'
    dataset = "Chairs"

    # model_dir: folder where the pretrain model is.
    model_dir = 'pretrain_600epoch'

    #restore_file: None if not pretrained weights. If loading weights: restore_file = 'last'
    restore_file = None

    # use GPU if available
    param_cuda = torch.cuda.is_available()

    # Define the models
    G_model = net.generator(128).cuda() if param_cuda else net.generator(128)
    D_model = net.discriminator(128).cuda() if param_cuda else net.discriminator(128)

    #Initialize weights
    G_model.weight_init(mean=0.0, std=0.02)
    D_model.weight_init(mean=0.0, std=0.02)

    #Define optimizers
    G_optimizer = optim.Adam(G_model.parameters(), lr=lr, betas=(0.5, 0.999))
    D_optimizer = optim.Adam(D_model.parameters(), lr=lr, betas=(0.5, 0.999))

    '''
    If freezing some layers, optimize only the other layers.
    
    #paramsG = list(G_model.deconv4.parameters()) + list(G_model.deconv4_bn.parameters()) + list(G_model.deconv5.parameters())
    #G_optimizer = optim.Adam(paramsG, lr=lr, betas=(0.5, 0.999))
    #paramsD = list(D_model.conv4.parameters()) + list(D_model.conv4_bn.parameters()) + list(D_model.conv5.parameters())
    #D_optimizer = optim.Adam(paramsD, lr=lr, betas=(0.5, 0.999))
    '''

    # fetch loss function
    loss_fn = net.loss_fn

    # Set the logger
    utils.set_logger(os.path.join(model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    # data_loader
    train_loader = data_loader.fetch_dataloader(data_dir, batch_size, dataset)
    print('dataset length '+str(len(train_loader.dataset)))

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(train_epoch))
    train_and_evaluate(param_cuda, dataset, G_model, D_model, G_optimizer, D_optimizer, loss_fn, train_loader, train_epoch, model_dir, restore_file=restore_file)

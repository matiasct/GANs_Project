import json
import logging
import os
import shutil
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import itertools
import torch

    
def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_checkpoint(state, is_best, checkpoint):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'.
    """
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        print("Checkpoint Directory exists! ")
    torch.save(state, filepath)


def load_checkpoint(checkpoint, D_model, G_model, D_optimizer=None, G_optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    """
    if not os.path.exists(checkpoint):
        raise("File doesn't exist {}".format(checkpoint))
    print(checkpoint)
    checkpoint = torch.load(checkpoint)
    D_model.load_state_dict(checkpoint['D_model_state_dict'])
    G_model.load_state_dict(checkpoint['G_model_state_dict'])

    if D_optimizer:
        D_optimizer.load_state_dict(checkpoint['D_optim_dict'])
    if G_optimizer:
        G_optimizer.load_state_dict(checkpoint['G_optim_dict'])

    return checkpoint


def show_train_hist(hist, path, show = False, save = False):

    x = range(len(hist['D_model_mean_losses']))
    y1 = hist['D_model_mean_losses']
    y2 = hist['G_model_mean_losses']
    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')
    plt.xlabel('Iter')
    plt.ylabel('Loss')
    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()
    if save:
        plt.savefig(path)
    if show:
        plt.show()
    else:
        plt.close()


def show_result(param_cuda, G_model, num_epoch, path, show = False, save=False):

    z_ = torch.randn((5*5, 100)).view(-1, 100, 1, 1)
    z_ = z_.cuda() if param_cuda else z_
    z_ = Variable(z_, volatile=True)

    G_model.eval()
    test_images = G_model(z_)

    size_figure_grid = 5
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(5*5):
        i = k // 5
        j = k % 5
        plot_image = test_images[k].data.cpu().numpy()
        plot_image = (plot_image - plot_image.min())/(plot_image.max() - plot_image.min())
        plot_image = np.transpose(plot_image,(1, 2, 0))
        ax[i, j].imshow(plot_image)

    if save:
        label = 'Epoch {0}'.format(num_epoch)
        fig.text(0.5, 0.04, label, ha='center')
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

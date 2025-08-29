from torch.nn import Module, ModuleList, Linear, ReLU, Sigmoid
from torch.nn import BCEWithLogitsLoss
from torch import Tensor
import torch.nn.functional as F
import torch
import pandas as pd
from losses import sigmoid_focal_loss, weighted_binary_cross_entropy

# model definition module is the root class for all neural networks
class MLP(Module):
    # define model elements
    def __init__(self, input_size, n_nodes=[4, 4, 4, 4], activation="sig"):
        super(MLP, self).__init__()
        # add list of layers
        self.layers = ModuleList()
        # init the concurrent input nodes
        next_input = input_size
        # fill list with hidden layers
        for nodes in n_nodes:
            self.layers.append(Linear(next_input, nodes))
            next_input = nodes
            if activation == "relu":
                self.layers.append(ReLU())
            if activation == "sig":
                self.layers.append(Sigmoid())
        # assign final output layer
        self.layers.append(Linear(next_input, 1))

    # forward propagation
    def propagate(self, X):
        if torch.isnan(X).any():
            print("NAN detected in model input")
        # concatenate data through layers
        for i, layer in enumerate(self.layers):
            #X = torch.nan_to_num(X)
            X = layer(X)
        # return result
        if torch.isnan(X).any():
            print("NAN detected in model output")
        return X

    # calculate the loss
    def calc_loss(self, pred, targets, pos_weight, loss_type, gamma):
        if loss_type == "focal":
            loss = sigmoid_focal_loss(
                alpha=pos_weight / 100, gamma=gamma, inputs=pred, targets=targets
            )
        if loss_type == "bce":
            loss = weighted_binary_cross_entropy(targets, pos_weight)
        return loss

    # implementation of early stopping method patience as number of eapochs to wait till validation loss incerases
    def early_stop(self, curr_validation_loss: list = [], patience: int = 10):
        # condition that values are comparable wait until patience size is met
        if len(curr_validation_loss) < patience:
            return False
        # get first and last validation loss value
        first = curr_validation_loss[-patience]
        last = curr_validation_loss[-1]
        # if whithin the epoch comparison first value lower last value stop
        if first < last:
            print("best validation loss:", last)
            return True
        return False


import math
import random
import itertools
import torch
import torch.nn as nn

# these imports are from DeepGMM package
from models.mlp_model import MLPModel
from optimizers.oadam import OAdam
from scenarios.toy_scenarios import Standardizer, AGMMZoo

from methods.mnist_x_model_selection_method import MNISTXModelSelectionMethod
from methods.mnist_xz_model_selection_method import MNISTXZModelSelectionMethod
from methods.mnist_z_model_selection_method import MNISTZModelSelectionMethod
from scenarios.abstract_scenario import AbstractScenario
from models.cnn_models import LeakySoftmaxCNN, DefaultCNN


SCENARIOS_NAMES = ["mnist_x", "mnist_z", "mnist_xz"]
SCENARIO_METHOD_CLASSES = {
    "mnist_x": MNISTXModelSelectionMethod,
    "mnist_z": MNISTZModelSelectionMethod,
    "mnist_xz": MNISTXZModelSelectionMethod,
}

ENABLE_CUDA = True


def main():
    num_train = 10000
    num_dev = 10000
    num_test = 10000
    num_epochs = 500
    batch_size = 1000
    
    scenario_name="mnist_xz"
    print("\nLoading " + scenario_name + "...")
    scenario = AbstractScenario(filename="data/" + scenario_name + "/main.npz")
    scenario.info()
    scenario.to_tensor()
    scenario.to_cuda()

    train = scenario.get_dataset("train")
    dev = scenario.get_dataset("dev")
    test = scenario.get_dataset("test")
    
    x_train, z_train, y_train = train.x, train.z, train.y
    x_dev, z_dev, y_dev, g_of_x_oracle_dev = dev.x, dev.z, dev.y, dev.g
    x_test, z_test, y_test, g_of_x_oracle_test = test.x, test.z, test.y, test.g

    # scenario name can be e.g. "abs", "linear", "sin", "step" to replicate
    # the respective scenarios from the paper
#     scenario_name = "step"
    # create data from respective scenario
#     scenario = Standardizer(AGMMZoo(scenario_name, two_gps=False,
#                                     n_instruments=2))
#     scenario.setup(num_train=num_train, num_dev=num_dev, num_test=num_test)
#     scenario.to_tensor()
#     if torch.cuda.is_available() and ENABLE_CUDA:
#         scenario.to_cuda()

#     x_train, z_train, y_train, _, _ = scenario.get_train_data()
#     x_dev, z_dev, y_dev, g_of_x_oracle_dev, _ = scenario.get_dev_data()
#     x_test, z_test, y_test, g_of_x_oracle_test, _ = scenario.get_test_data()

    # set up f and g models and optimizers
#     g = MLPModel(input_dim=1, layer_widths=[20, 3],
#                  activation=nn.LeakyReLU).double()
#     f = MLPModel(input_dim=2, layer_widths=[20],
#                  activation=nn.LeakyReLU).double()
    g = DefaultCNN(cuda=ENABLE_CUDA)
    f = DefaultCNN(cuda=ENABLE_CUDA)

    if torch.cuda.is_available() and ENABLE_CUDA:
        g = g.cuda()
        f = f.cuda()
    g_optimizer = OAdam(g.parameters(), lr=0.00005, betas=(0.5, 0.9)) #was 0.00001
    f_optimizer = OAdam(f.parameters(), lr=0.000025, betas=(0.5, 0.9)) #0.000005
    # train models using DeepGMM algorithm
    g = train_deep_gmm(g=g, f=f, g_optimizer=g_optimizer,
                       f_optimizer=f_optimizer, num_epochs=num_epochs,
                       batch_size=batch_size, x_train=x_train, z_train=z_train,
                       y_train=y_train, verbose=True, print_freq=20,
                       x_dev=x_dev, z_dev=z_dev, y_dev=y_dev,
                       g_of_x_oracle_dev=g_of_x_oracle_dev)

    # test output g function on test data
    test_mse = calc_mse_safe_test(x_test, g_of_x_oracle_test, g,
                             batch_size=batch_size)
    torch.save({'model':g.state_dict()}, 'g_mnist.pth')
    print("MSE on test data: %f" % test_mse)
    print("")


def calc_game_objective(g, f, x, z, y):
    # calculate the tuple of objective functions that the g and f networks
    # respectively are minimizing
    g_of_x = torch.squeeze(g(x))
    f_of_z = torch.squeeze(f(z))
    y = torch.squeeze(y)
    epsilon = g_of_x - y

    moment = f_of_z.mul(epsilon).mean()
    f_reg = 0.25 * (f_of_z ** 2).mul(epsilon ** 2).mean()
    return moment, -moment + f_reg


def calc_obj_safe(x, z, y, g, f, batch_size):
    # this function is written to be safe with large amount of data when using
    # CUDA, to avoid memory issues form computing on entire data at once
    num_data = x.shape[0]
    num_batch = math.ceil(num_data / batch_size)
    obj_total = 0
    for b in range(num_batch):
        if b < num_batch - 1:
            batch_idx = list(range(b*batch_size, (b+1)*batch_size))
        else:
            batch_idx = list(range(b*batch_size, num_data))
        x_batch = x[batch_idx]
        z_batch = z[batch_idx]
        y_batch = y[batch_idx]
        g_obj, _ = calc_game_objective(g, f, x_batch, z_batch, y_batch)
        obj_total += float(g_obj.detach().cpu()) * len(batch_idx) / num_data
    return obj_total


def calc_mse_safe(x, g_of_x_oracle, g, batch_size):
    # this function is written to be safe with large amount of data when using
    # CUDA, to avoid memory issues form computing on entire data at once
    num_data = x.shape[0]
    num_batch = math.ceil(num_data / batch_size)
    mse_total = 0
    g_of_x_oracle = torch.squeeze(g_of_x_oracle)
    for b in range(num_batch):
        if b < num_batch - 1:
            batch_idx = list(range(b*batch_size, (b+1)*batch_size))
        else:
            batch_idx = list(range(b*batch_size, num_data))
        x_batch = x[batch_idx]
        g_of_x_batch = torch.squeeze(g(x_batch))
        g_of_x_oracle_batch = g_of_x_oracle[batch_idx]
        mse = ((g_of_x_batch - g_of_x_oracle_batch) ** 2).mean()
        mse_total += float(mse.detach().cpu()) * len(batch_idx) / num_data
    return mse_total

def calc_mse_safe_test(x, g_of_x_oracle, g, batch_size):
    # this function is written to be safe with large amount of data when using
    # CUDA, to avoid memory issues form computing on entire data at once
    num_data = x.shape[0]
    num_batch = math.ceil(num_data / batch_size)
    mse_total = 0
    g_of_x_oracle = torch.squeeze(g_of_x_oracle)
    import pdb;pdb.set_trace()
    for b in range(num_batch):
        if b < num_batch - 1:
            batch_idx = list(range(b*batch_size, (b+1)*batch_size))
        else:
            batch_idx = list(range(b*batch_size, num_data))
        x_batch = x[batch_idx]
        g_of_x_batch = torch.squeeze(g(x_batch))
        g_of_x_oracle_batch = g_of_x_oracle[batch_idx]
        mse = ((g_of_x_batch - g_of_x_oracle_batch) ** 2).mean()
        mse_total += float(mse.detach().cpu()) * len(batch_idx) / num_data
    return mse_total

def train_deep_gmm(g, f, g_optimizer, f_optimizer, num_epochs, batch_size,
                   x_train, z_train, y_train, verbose=False, x_dev=None,
                   z_dev=None, y_dev=None, g_of_x_oracle_dev=None,
                   print_freq=20):
    # train our g function using DeepGMM algorithm

    num_data = x_train.shape[0]
    for epoch in range(num_epochs):
            if verbose and epoch % print_freq == 0:
                # print out current game objective and mse on dev data
                dev_obj = calc_obj_safe(x_dev, z_dev, y_dev, g, f,
                                        batch_size=batch_size)
                dev_mse = calc_mse_safe(x_dev, g_of_x_oracle_dev, g,
                                        batch_size=batch_size)
                print("epoch %d, dev objective = %f, dev mse = %f"
                      % (epoch, dev_obj, dev_mse))

            # decide random order of data for batches in this epoch
            num_batch = math.ceil(num_data / batch_size)
            train_idx = list(range(num_data))
            random.shuffle(train_idx)
            idx_iter = itertools.cycle(train_idx)

            # loop through training data in batches
            for _ in range(num_batch):
                batch_idx = [next(idx_iter) for _ in range(batch_size)]
                x_batch = x_train[batch_idx]
                z_batch = z_train[batch_idx]
                y_batch = y_train[batch_idx]
                g_obj, f_obj = calc_game_objective(
                    g, f, x_batch, z_batch, y_batch)

                # do single first order optimization step on f and g
                g_optimizer.zero_grad()
                g_obj.backward(retain_graph=True)
                g_optimizer.step()

                f_optimizer.zero_grad()
                f_obj.backward()
                f_optimizer.step()

    return g


if __name__ == "__main__":
    main()

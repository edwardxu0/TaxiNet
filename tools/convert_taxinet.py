#!/usr/bin/env python
import os
import argparse
import numpy as np
import h5py
import tensorflow as tf
from tensorflow.python.framework import tensor_util

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


import pathlib

import matplotlib.pyplot as plt


def _parse_args():
    parser = argparse.ArgumentParser(description="TaxiNet tool.")
    parser.add_argument("task", type=str, choices=["train", "soi", "gen_prop"])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "gpu"])
    parser.add_argument("--log_interval", type=int, default=500)
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--result_dir", default="./results/taxinet")
    parser.add_argument("--taxinet_root", default="lib/SoI/soi_artifact/data/taxi")
    parser.add_argument(
        "--soi_model_path", default="./lib/SoI/soi_artifact/data/taxi/32x16_688.pb"
    )
    parser.add_argument("--property_root", default="properties/taxinet/")

    return parser.parse_args()


class TaxiNetDataset(Dataset):
    def __init__(self, path, partition):
        super(Dataset, self).__init__()
        dataset = h5py.File(path, "r")
        if partition == "train":
            self.X = torch.from_numpy(np.array(dataset["X_train"], dtype=np.float32))
            self.Y = torch.from_numpy(np.array(dataset["y_train"], dtype=np.float32))
        elif partition == "test":
            self.X = torch.from_numpy(np.array(dataset["X_val"], dtype=np.float32))
            self.Y = torch.from_numpy(np.array(dataset["y_val"], dtype=np.float32))
        else:
            assert False
        self.X = self.X.view(-1, 1, 16, 32)

    def __len__(self):
        # print(self.X.shape, self.Y.shape)
        assert len(self.X) == len(self.Y)
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class TaxiNet(nn.Module):
    def __init__(self, verbose=False):
        super(TaxiNet, self).__init__()

    def load_soi_weight(self, path):
        print(f"Loading weights/bias from {path}")
        with tf.compat.v1.Session() as sess:
            with tf.io.gfile.GFile(path, "rb") as f:
                graph_def = tf.compat.v1.GraphDef()
                graph_def.ParseFromString(f.read())
                sess.graph.as_default()
                tf.import_graph_def(graph_def, name="")
                graph_nodes = [n for n in graph_def.node]

                names = [
                    x.name
                    for x in graph_nodes
                    if x.op == "Const" and "Reshape" not in x.name
                ]

                print("Nodes: ", names)

                idx = 0
                for t in graph_nodes:
                    if t.op == "Const" and "Reshape" not in t.name:
                        print(f"{idx}: {t.name}")
                        data = tensor_util.MakeNdarray(t.attr["value"].tensor)
                        if "kernel" in t.name:
                            print("  loading weight ...")
                            print(data.shape)
                            if len(data.shape) == 4:
                                new_shape = (3, 2, 0, 1)
                                # new_shape = self.layers[idx].weight.shape
                            elif len(data.shape) == 2:
                                new_shape = (1, 0)
                                # new_shape = (2, 8)
                            else:
                                assert False
                            data_t = np.transpose(data, new_shape)
                            # data_t = np.reshape(data, new_shape)
                            print(
                                f"    Need shape: {self.layers[idx].weight.shape}, data shape: {data.shape}, converted data shape: {data_t.shape}"
                            )
                            new_data = torch.tensor(data_t)

                            self.layers[idx].weight.data = new_data

                        elif "bias" in t.name:
                            print("  loading bias ...")
                            new_data = torch.tensor(data)
                            self.layers[idx].bias.data = new_data
                            idx += 1
                        else:
                            assert False


class TaxiNet32x16(TaxiNet):
    def __init__(self, verbose=False):
        super(TaxiNet32x16, self).__init__()
        stride = (2, 2)
        padding = (1, 1)
        self.conv0 = nn.Conv2d(1, 4, 3, stride=stride, padding=padding)
        self.conv1 = nn.Conv2d(4, 4, 3, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(4, 4, 3, stride=stride, padding=padding)
        self.conv3 = nn.Conv2d(4, 4, 3, stride=stride, padding=padding)
        self.conv4 = nn.Conv2d(4, 4, 3, stride=(1, 1), padding=padding)
        self.fc1 = nn.Linear(8, 2)

        self.layers = [
            self.conv0,
            self.conv1,
            self.conv2,
            self.conv3,
            self.conv4,
            self.fc1,
        ]

        if verbose:
            print("--------------------")
            print("Parameter shapes:")
            for l in self.layers:
                print(l.weight.data.shape)
            print("--------------------")

    def forward(self, x, verbose=False):
        if verbose:
            print("--------------------")
            print("Data shapes:")
        x = self.conv0(x)
        if verbose:
            print("0", x.shape)
        x = F.relu(x)
        x = self.conv1(x)
        if verbose:
            print("1", x.shape)
        x = F.relu(x)
        x = self.conv2(x)
        if verbose:
            print("2", x.shape)
        x = F.relu(x)
        x = self.conv3(x)
        if verbose:
            print("3", x.shape)
        x = F.relu(x)
        x = self.conv4(x)
        if verbose:
            print("4", x.shape)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        if verbose:
            print("5", x.shape)
        x = self.fc1(x)
        if verbose:
            print("--------------------")
        return x


def load_data(path):
    training_data_file = h5py.File(path, "r")
    X_train = np.array(training_data_file["X_train"])
    Y_train = np.array(training_data_file["y_train"])
    X_test = np.array(training_data_file["X_val"])
    Y_test = np.array(training_data_file["y_val"])
    return X_train, Y_train, X_test, Y_test


def gen_prop(args):
    data_path = os.path.join(args.taxinet_root, "training_data/32x16.h5")
    # data_train = TaxiNetDataset(data_path, "train")
    data_test = TaxiNetDataset(data_path, "test")
    # train_loader = DataLoader(data_train, batch_size=64, shuffle=True)
    test_loader = DataLoader(data_test, batch_size=1000, shuffle=False)

    id = 0
    eps = 0.02
    gamma = 0.5

    pathlib.Path(args.property_root).mkdir(exist_ok=True, parents=True)
    npy_img = next(enumerate(test_loader))[1][id][0].data.numpy().reshape(1, 1, 16, 32)

    npy_path = os.path.join(os.getcwd(), args.property_root, f"{id}_{eps}.npy")
    prop_path = os.path.join(os.getcwd(), args.property_root, f"{id}_{eps}.dnnp")
    np.save(npy_path, npy_img)

    lines = [
        "from dnnv.properties import *",
        "import numpy as np",
        "",
        'N = Network("N")',
        f'x = Image("{npy_path}")',
        "",
        f"epsilon = {eps}",
        f"gamma = {gamma}",
        "output = N(x)",
        "lb = output - gamma",
        "ub = output + gamma",
        "",
        "Forall(",
        "    x_,",
        "    Implies(",
        "        ((x - epsilon) < x_ < (x + epsilon)),",
        "        (lb < N(x_) < ub),",
        "    ),",
        ")",
    ]
    lines = [x + "\n" for x in lines]
    open(prop_path, "w").writelines(lines)
    print(f"Property saved to {prop_path}.")


def soi(args):
    data_path = os.path.join(args.taxinet_root, "training_data/32x16.h5")
    # data_path = os.path.join(args.taxinet_root, "training_data/16x8.h5")
    # data_train = TaxiNetDataset(data_path, "train")
    data_test = TaxiNetDataset(data_path, "test")
    # train_loader = DataLoader(data_train, batch_size=64, shuffle=True)
    test_loader = DataLoader(data_test, batch_size=1000, shuffle=False)

    # process_tf_graph(taxi_root + "32x16_688.pb")
    # process_tf_graph(taxi_root + "64x32_2752.pb")
    # process_tf_graph(taxi_root + "KJ_64x32_2048.pb")

    x, y = next(enumerate(test_loader))[1]

    model = TaxiNet32x16(verbose=True)
    model.load_soi_weight(args.soi_model_path)
    x = torch.tensor(x[0].reshape(1, 1, 16, 32), dtype=torch.float)
    print(
        "Testing image:",
        x.shape,
        "pred:",
        model(x, verbose=True).data,
        "label:",
        y[0],
    )

    test_epoch(args, model, args.device, test_loader, 0)

    if args.save_model:
        pathlib.Path(args.result_dir).mkdir(exist_ok=True, parents=True)
        model_path = os.path.splitext(args.soi_model_path)[0].split("/")[-1] + ".onnx"
        model_path = os.path.join(args.result_dir, model_path)
        print(f"Saving model to {model_path}.")
        torch.onnx.export(model, x, model_path, verbose=True)


def train(args):
    data_path = os.path.join(args.taxinet_root, "training_data/32x16.h5")

    data_train = TaxiNetDataset(data_path, "train")
    data_test = TaxiNetDataset(data_path, "test")
    train_loader = DataLoader(data_train, batch_size=64, shuffle=True)
    test_loader = DataLoader(data_test, batch_size=1000, shuffle=False)

    model = TaxiNet32x16()

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    for epoch in range(1, args.epochs + 1):
        train_epoch(args, model, args.device, train_loader, optimizer, epoch)
        test_epoch(args, model, args.device, test_loader, epoch)
        scheduler.step()


def train_epoch(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        loss = nn.MSELoss()(output, target)

        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


def test_epoch(args, model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)

        loss = nn.MSELoss()(output, target)
        test_loss += loss

    test_loss /= len(test_loader)
    print(
        "[Test]:",
        epoch,
        test_loss.detach().numpy(),
        output[-1].detach().numpy(),
        target[-1].numpy(),
    )


def main():
    args = _parse_args()
    if args.task == "soi":
        soi(args)
    elif args.task == "train":
        train(args)
    elif args.task == "gen_prop":
        gen_prop(args)


if __name__ == "__main__":
    main()

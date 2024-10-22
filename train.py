import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import tqdm
from matplotlib import pyplot as plt
from torch.optim import Adam
from torch.utils.data import DataLoader

from wikiart import WikiArtDataset, WikiArtModel

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", help="configuration file", default="config.json")

args = parser.parse_args()

config = json.load(open(args.config))["model_train"]

trainingdir = config["trainingdir"]
testingdir = config["testingdir"]
device = config["device"]

traindataset = WikiArtDataset(trainingdir, device)


def train(epochs=3, batch_size=32, modelfile=None, device="cpu"):
    loader = DataLoader(traindataset, batch_size=batch_size, shuffle=True)

    model = WikiArtModel().to(device)
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = nn.NLLLoss().to(device)

    loss_for_training = []

    for epoch in range(epochs):
        print("Starting epoch {}".format(epoch))
        loss_at_step = []
        for _, batch in enumerate(tqdm.tqdm(loader)):
            X, y = batch
            y = y.to(device)
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss_at_step.append(loss.item())
            loss.backward()
            optimizer.step()

        epoch_loss = sum(loss_at_step) / len(loss_at_step)
        loss_for_training.append(epoch_loss)
        print(f"Mean loss for epoch {epoch} is {epoch_loss}")

    fig, ax = plt.subplots()
    ax.set_title(f"Training loss for {epochs} epochs")
    ax.set_ylabel("Mean loss for epoch")
    ax.set_xlabel("Epoch")
    ax.set_xticks([x for x in range(epochs)])
    ax.plot(loss_for_training)
    fig.savefig(Path("images") / "training_loss.png")

    if modelfile:
        torch.save(model.state_dict(), modelfile)

    return model


model = train(
    config["epochs"], config["batch_size"], modelfile=config["modelfile"], device=device
)

import argparse
import json

import torch
import torch.nn as nn
import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader

from wikiart import WikiArtDataset, WikiArtModel

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", help="configuration file", default="config.json")

args = parser.parse_args()

config = json.load(open(args.config))

trainingdir = config["trainingdir"]
testingdir = config["testingdir"]
device = config["device"]

traindataset = WikiArtDataset(trainingdir, device)

print(traindataset.imgdir)

the_image, the_label = traindataset[5]
print(the_image, the_image.size())


def train(epochs=3, batch_size=32, modelfile=None, device="cpu"):
    loader = DataLoader(traindataset, batch_size=batch_size, shuffle=True)

    model = WikiArtModel().to(device)
    optimizer = Adam(model.parameters(), lr=0.01)
    criterion = nn.NLLLoss().to(device)

    for epoch in range(epochs):
        print("Starting epoch {}".format(epoch))
        accumulate_loss = 0
        for _, batch in enumerate(tqdm.tqdm(loader)):
            X, y = batch
            y = y.to(device)
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            accumulate_loss += loss
            optimizer.step()

        print("In epoch {}, loss = {}".format(epoch, accumulate_loss))

    if modelfile:
        torch.save(model.state_dict(), modelfile)

    return model


model = train(
    config["epochs"], config["batch_size"], modelfile=config["modelfile"], device=device
)

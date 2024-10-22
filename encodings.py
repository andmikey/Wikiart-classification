import argparse
import json

import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import tqdm
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from torch.optim import Adam
from torch.utils.data import DataLoader

from wikiart import WikiArtAutoencoder, WikiArtDataset


def train(traindataset, epochs=3, batch_size=32, device="cpu"):
    loader = DataLoader(traindataset, batch_size=batch_size, shuffle=True)

    model = WikiArtAutoencoder().to(device)
    optimizer = Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss().to(device)

    for epoch in range(epochs):
        print("Starting epoch {}".format(epoch))
        accumulate_loss = 0
        for _, batch in enumerate(tqdm.tqdm(loader)):
            X, cls = batch
            optimizer.zero_grad()
            output = model(X)
            # Loss is comparing generated image to actual image
            loss = criterion(output, X)
            loss.backward()
            accumulate_loss += loss
            optimizer.step()

        # TODO should plot the epoch loss over time
        print("In epoch {}, loss = {}".format(epoch, accumulate_loss))

    return model


def plot_embeddings(model, testdataset, output_file):
    # Run the model on the test dataset to get clusterings
    # (Yes, doing it one-by-one is quite inefficient, but I found this easiest for
    # understanding what was happening)
    loader = DataLoader(testdataset, batch_size=1, shuffle=False)

    # Create lists of [class_idxes], [encoded_images]
    arr = []
    label_arr = []

    cls_indexes = testdataset.classes

    for _, batch in enumerate(loader):
        img, classes = batch
        # Returns shape [batch_size, 1, 10, 10]
        encoded_img = model.encoder(img).to("cpu").detach().numpy()
        classes = classes.to("cpu").detach().numpy()
        # Flatten image for PCA
        img = encoded_img[0, 0, :].flatten()
        cls = classes[0]
        arr.append(img)
        label_arr.append(cls_indexes[cls])

    components = np.array(arr)

    # PCA reduction on encoded images
    pca = PCA(n_components=2)
    X_fit = pca.fit_transform(components)

    # Plot encoded images grouped by class
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.color_palette("mako")
    sns.scatterplot(x=X_fit[:, 0], y=X_fit[:, 1], hue=label_arr, ax=ax)
    fig.savefig("out.png")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c", "--config", help="configuration file", default="config.json"
    )
    parser.add_argument("-t", "--train-model", default=True)
    parser.add_argument("-s", "--use-style-embeddings", default=False)

    args = parser.parse_args()

    config = json.load(open(args.config))["autoencoder"]

    trainingdir = config["trainingdir"]
    testingdir = config["testingdir"]
    device = config["device"]

    traindataset = WikiArtDataset(trainingdir, device)
    testdataset = WikiArtDataset(testingdir, device)

    # Only train model if requested in the command-line arguments
    if args.train_model:
        model = train(
            traindataset,
            config["epochs"],
            config["batch_size"],
            modelfile=config["modelfile"],
            device=device,
        )

        if config["modelfile"]:
            torch.save(model.state_dict(), model)

    # Plot embeddings
    # Load model from file if not traning
    if not args.train_model:
        model = WikiArtAutoencoder.to(device)
        model.load_state_dict(torch.load(config["modelfile"], weights_only=True))

    model.eval()

    plot_embeddings(model, testdataset, config["embeddings_output_file"])


if __name__ == "__main__":
    main()

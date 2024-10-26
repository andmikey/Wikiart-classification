import argparse
import json
from pathlib import Path

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


def train(traindataset, epochs=3, batch_size=32, device="cpu", save_dir=None):
    loader = DataLoader(traindataset, batch_size=batch_size, shuffle=True)

    model = WikiArtAutoencoder().to(device)
    optimizer = Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss().to(device)

    loss_for_training = []

    for epoch in range(epochs):
        print("Starting epoch {}".format(epoch))
        loss_at_step = []
        for _, batch in enumerate(tqdm.tqdm(loader)):
            X, cls = batch
            optimizer.zero_grad()
            output = model(X)
            # Loss is comparing generated image to actual image
            loss = criterion(output, X)
            loss_at_step.append(loss.item())
            loss.backward()
            optimizer.step()

        epoch_loss = sum(loss_at_step) / len(loss_at_step)
        loss_for_training.append(epoch_loss)
        print(f"Mean loss for epoch {epoch} is {epoch_loss}")

    # Plot training loss
    fig, ax = plt.subplots()
    ax.set_title(f"Training loss for {epochs} epochs")
    ax.set_ylabel("Mean loss for epoch")
    ax.set_xlabel("Epoch")
    ax.set_xticks([x for x in range(epochs)])
    ax.plot(loss_for_training)
    fig.savefig(save_dir / "embedding_model_training_loss.png")

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
    fig.savefig(output_file)


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
            device=device,
            save_dir=Path(config["additional_outputs_dir"]),
        )

        if config["modelfile"]:
            torch.save(model.state_dict(), config["modelfile"])

    # Plot embeddings
    # Load model from file if not traning
    if not args.train_model:
        model = WikiArtAutoencoder.to(device)
        model.load_state_dict(torch.load(config["modelfile"], weights_only=True))

    model.eval()

    plot_embeddings(model, testdataset, config["embeddings_output_file"])


if __name__ == "__main__":
    main()

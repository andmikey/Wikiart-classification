import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import tqdm
from matplotlib import pyplot as plt
from torch.optim import Adam
from torch.utils.data import DataLoader

from wikiart import WikiArtAutoencoder, WikiArtDataset, WikiArtModel


def generate_class_embeddings(model, traindataset, batch_size, device="cpu"):
    train_loader = DataLoader(traindataset, batch_size=batch_size, shuffle=True)

    embeddings_for_cls = defaultdict(list)

    for _, batch in enumerate(train_loader):
        X, cls = batch
        cls = cls.to(device)  # [32]
        output = model.embedding(X)  # [32, 100]
        # Gather the embeddings for each class
        for i in range(len(cls)):
            cls_idx = int(cls[i].cpu().numpy())
            embedding = output[i].cpu()
            embeddings_for_cls[cls_idx].append(embedding)

    # Now average the class embeddings
    class_avgs = {}
    for i in embeddings_for_cls:
        class_avgs[i] = torch.mean(torch.stack(embeddings_for_cls[i]), axis=0).detach()

    num_classes = max(class_avgs)
    embeddings_arr = torch.zeros(num_classes + 1, 100).to(device)
    for i in class_avgs:
        embeddings_arr[i] = class_avgs[i]

    return embeddings_arr


def train_autoencoder(
    traindataset, class_embeddings, epochs=3, batch_size=32, device="cpu", save_dir=None
):
    loader = DataLoader(traindataset, batch_size=batch_size, shuffle=True)

    model = WikiArtAutoencoder(use_embedding=True).to(device)
    optimizer = Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss().to(device)

    loss_for_training = []

    for epoch in range(epochs):
        print("Starting epoch {}".format(epoch))
        loss_at_step = []
        for _, batch in enumerate(tqdm.tqdm(loader)):
            X, cls = batch
            embeddings = class_embeddings[cls]
            optimizer.zero_grad()
            output = model(X, embeddings)
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
    fig.savefig(save_dir / "embedding_training_loss_with_style.png")

    return model


def main():
    # This script needs to be run on CPU
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c", "--config", help="configuration file", default="config.json"
    )

    args = parser.parse_args()

    config = json.load(open(args.config))["style_embeddings"]

    trainingdir = config["trainingdir"]
    testingdir = config["testingdir"]
    device = config["device"]

    traindataset = WikiArtDataset(trainingdir, device)
    testdataset = WikiArtDataset(testingdir, device)

    # Use trained model to get average class embeddings for images in training set
    style_model = WikiArtModel().to(device)
    style_model.load_state_dict(
        torch.load(config["embedding_model"], weights_only=True)
    )
    style_model.eval()
    class_embeddings = generate_class_embeddings(
        style_model, traindataset, config["batch_size"], device
    )
    # Now train autoencoder using the class style embeddings
    autoencoder = train_autoencoder(
        traindataset,
        class_embeddings,
        config["epochs"],
        config["batch_size"],
        device=device,
        save_dir=Path(config["additional_outputs_dir"]),
    )

    if config["trained_model"]:
        torch.save(autoencoder.state_dict(), config["trained_model"])

    # Try generating an image


if __name__ == "__main__":
    main()

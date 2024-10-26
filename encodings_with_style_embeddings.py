import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import tqdm
from matplotlib import pyplot as plt
from torch.optim import Adam
from torch.utils.data import DataLoader

from wikiart import WikiArtAutoencoder, WikiArtDataset, WikiArtModel


def generate_class_embeddings(model, traindataset, batch_size, device):
    train_loader = DataLoader(traindataset, batch_size=batch_size, shuffle=True)

    embeddings_for_cls = defaultdict(lambda: torch.zeros(100).to(device))
    cls_ct = defaultdict(lambda: 0)

    for _, batch in enumerate(train_loader):
        # This bit gets OOM issues if run on GPU, works fine on CPU
        # Haven't been able to fix it
        torch.cuda.empty_cache()
        X, cls = batch
        output = model.embedding(X)  # [32, 100]
        # Gather the embeddings for each class
        for i in range(len(cls)):
            cls_idx = int(cls[i].cpu().numpy())
            cls_ct[cls_idx] += 1
            embedding = output[i]
            embeddings_for_cls[cls_idx] = (
                embedding.detach() + embeddings_for_cls[cls_idx]
            )

            del embedding
        del X, cls, output

    # Now average the class embeddings
    class_avgs = {}
    for i in embeddings_for_cls:
        class_avgs[i] = embeddings_for_cls[i] / cls_ct[i]

    num_classes = max(class_avgs)
    embeddings_arr = np.zeros((num_classes + 1, 100))
    for i in class_avgs:
        embeddings_arr[i] = class_avgs[i].cpu().detach().numpy()

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
            embeddings = torch.from_numpy(class_embeddings[cls]).cuda().to(device)
            optimizer.zero_grad()
            # print(f"Image device: {X.get_device()}")
            # print(f"Embeddings device: {embeddings.get_device()}")
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
    parser.add_argument("-t", "--train-model", default=True)

    args = parser.parse_args()

    config = json.load(open(args.config))["style_embeddings"]

    trainingdir = config["trainingdir"]
    testingdir = config["testingdir"]
    device = config["device"]

    traindataset = WikiArtDataset(trainingdir, device)
    testdataset = WikiArtDataset(testingdir, device)

    if args.train_model:
        print("Training a model")
        # Use trained model to get average class embeddings for images in training set
        print("Generating style embeddings from train set")
        style_model = WikiArtModel().to(device)
        style_model.load_state_dict(
            torch.load(
                config["embedding_model"],
                weights_only=True,
                map_location=torch.device(device),
            )
        )
        style_model.eval()
        class_embeddings = generate_class_embeddings(
            style_model,
            WikiArtDataset(trainingdir, device),
            config["batch_size"],
            device,
        )
        if config["class_embeddings"]:
            with open(config["class_embeddings"], "wb") as f:
                pickle.dump(class_embeddings, f)

        # Now train autoencoder using the class style embeddings
        print("Training autoencoder with style embeddings")
        model = train_autoencoder(
            traindataset,
            class_embeddings,
            config["epochs"],
            config["batch_size"],
            device=device,
            save_dir=Path(config["additional_outputs_dir"]),
        )

        if config["trained_model"]:
            torch.save(model.state_dict(), config["trained_model"])

    else:
        print("Using existing weights")
        # Load pre-trained model
        if config["trained_model"]:
            model = WikiArtAutoencoder(use_embedding=True).to(device)
            model.load_state_dict(
                torch.load(config["trained_model"], weights_only=True)
            )
            model.eval()

        # Load pre-trained class embeddings
        with open(config["class_embeddings"], "rb") as f:
            class_embeddings = pickle.load(f)

    # Try generating some images
    img1, lbl1 = testdataset[0]
    img2, lbl2 = testdataset[300]
    print(lbl1, testdataset.classes[lbl1], lbl2, testdataset.classes[lbl2])

    img1_with_own_class = (
        model(img1.unsqueeze(0), class_embeddings[lbl1].unsqueeze(0))[0]
        .detach()
        .permute((1, 2, 0))
    )
    img1_with_diff_class = (
        model(img1.unsqueeze(0), class_embeddings[lbl1 + 1].unsqueeze(0))[0]
        .detach()
        .permute((1, 2, 0))
    )

    img2_with_own_class = (
        model(img2.unsqueeze(0), class_embeddings[lbl2].unsqueeze(0))[0]
        .detach()
        .permute((1, 2, 0))
    )
    img2_with_diff_class = (
        model(img2.unsqueeze(0), class_embeddings[lbl2 + 1].unsqueeze(0))[0]
        .detach()
        .permute((1, 2, 0))
    )

    # Print out the image w/ style embedding, image w/ style embedding of next class
    fig, axs = plt.subplots(2, 3, figsize=(10, 10))

    axs[0][0].set_title(f"Original image")
    axs[0][0].imshow(img1.permute((1, 2, 0)) / 255)
    axs[0][1].set_title(f"With {testdataset.classes[lbl1]} style")
    axs[0][1].imshow(img1_with_own_class / 255)
    axs[0][2].set_title(f"With {testdataset.classes[lbl1+1]} style")
    axs[0][2].imshow(img1_with_diff_class / 255)

    axs[1][0].set_title(f"Original image")
    axs[1][0].imshow(img2.permute((1, 2, 0)) / 255)
    axs[1][1].set_title(f"With {testdataset.classes[lbl2]} style")
    axs[1][1].imshow(img2_with_own_class / 255)
    axs[1][2].set_title(f"With {testdataset.classes[lbl2+1]} style")
    axs[1][2].imshow(img2_with_diff_class / 255)

    plt.savefig("images/style_experiments.png")


if __name__ == "__main__":
    main()

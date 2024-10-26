import os
import random
from collections import defaultdict

import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
from torchvision.io import read_image

# Set a random seed for reproducibility
random.seed(42)

# Resampling
# If < 0, only a proportion of training images will be used
# If > 0, images will be repeated
RESAMPLE_DICT = {
    "Impressionism": 0.176289,
    "Realism": 0.233645,
    "Romanticism": 0.345722,
    "Expressionism": 0.354925,
    "Post_Impressionism": 0.422833,
    "Baroque": 0.554785,
    "Art_Nouveau_Modern": 0.581395,
    "Symbolism": 0.589102,
    "Abstract_Expressionism": 1,
    "Northern_Renaissance": 1,
    "Cubism": 1,
    "Naive_Art_Primitivism": 1,
    "Rococo": 1,
    "Color_Field_Painting": 1,
    "Mannerism_Late_Renaissance": 2,
    "Early_Renaissance": 2,
    "Pop_Art": 2,
    "Minimalism": 2,
    "High_Renaissance": 2,
    "Ukiyo_e": 2,
    "Fauvism": 2,
    "Contemporary_Realism": 4,
    "Pointillism": 5,
    "New_Realism": 10,
    "Synthetic_Cubism": 11,
    "Action_painting": 22,
    "Analytical_Cubism": 27,
}


class WikiArtImage:
    def __init__(self, imgdir, label, filename):
        self.imgdir = imgdir
        self.label = label
        self.filename = filename
        self.image = None
        self.loaded = False

    def get(self):
        if not self.loaded:
            self.image = read_image(
                os.path.join(self.imgdir, self.label, self.filename)
            ).float()
            self.loaded = True

        return self.image


class WikiArtDataset(Dataset):
    def __init__(self, imgdir, device="cpu"):
        walking = os.walk(imgdir)
        filedict = {}
        indices = []
        classes = set()
        class_counts = defaultdict(lambda: 0)

        # Run through the files in alphabetical order to ensure consistency between runs (os.walk is random order
        # by default)
        for item in sorted(walking):
            arttype = os.path.basename(item[0])  # Name of class
            if arttype in ["train", "test"]:
                # Don't sample the train/test directories
                # (tbh os.walk isn't really the best choice here, but I'll keep it
                # to keep parity with the example code)
                continue
            resample_prob = RESAMPLE_DICT[arttype]  # Probability of resampling
            artfiles = item[2]  # All the files in that class folder
            for art in artfiles:
                # Iterate over each file in the class folder
                filedict[art] = WikiArtImage(imgdir, arttype, art)
                classes.add(arttype)

                # Resample to adjust class probabilities
                if resample_prob >= 1:
                    # Add multiple examples of this art piece
                    class_counts[arttype] += resample_prob
                    indices.extend([art] * resample_prob)
                else:
                    rand_sample = random.random()
                    if resample_prob >= rand_sample:
                        class_counts[arttype] += 1
                        indices.append(art)

        self.filedict = filedict
        self.imgdir = imgdir
        self.indices = indices
        self.classes = list(classes)
        self.device = device

        # Print new class counts after resampling
        # for c in class_counts:
        #     print(f"{class_counts[c]} {c}")

    def __len__(self):
        # Since we've resampled, use self.indices
        # rather than self.filedict to get the dataloader length
        return len(self.indices)

    def __getitem__(self, idx):
        imgname = self.indices[idx]
        imgobj = self.filedict[imgname]
        ilabel = self.classes.index(imgobj.label)
        image = imgobj.get().to(self.device)

        return image, ilabel


class WikiArtModel(nn.Module):
    def __init__(self, num_classes=27):
        super().__init__()

        # Use this to get the art style embedding
        self.embedding = nn.Sequential(
            nn.Conv2d(3, 3, (4, 4)),
            nn.Dropout(0.01),
            nn.ReLU(),
            nn.Conv2d(3, 3, (4, 4)),
            nn.Dropout(0.01),
            nn.ReLU(),
            nn.Conv2d(3, 1, (4, 4), padding=2),
            nn.MaxPool2d((4, 4), padding=2),
            nn.Flatten(),
            nn.BatchNorm1d(103 * 103),
            nn.Linear(103 * 103, 100),
            nn.Dropout(0.01),
            nn.ReLU(),
        )

        self.output_classes = nn.Sequential(
            nn.Linear(100, num_classes), nn.LogSoftmax(dim=1)
        )

    def forward(self, image):
        return self.output_classes(self.embedding(image))


class WikiArtAutoencoder(nn.Module):
    def __init__(self, use_embedding=False):
        super().__init__()
        self.use_embedding = use_embedding
        # Splitting into encoding/decoding like this means we can easily pull just the encoder once the model is trained
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=5, stride=2),
            nn.BatchNorm2d(3),
            nn.Dropout2d(),
            nn.Conv2d(3, 3, kernel_size=5, stride=2),
            nn.BatchNorm2d(3),
            nn.Dropout2d(),
            nn.Conv2d(3, 3, kernel_size=5, stride=2),
            nn.BatchNorm2d(3),
            nn.Dropout2d(),
            nn.Conv2d(3, 1, kernel_size=5, stride=2),
            nn.BatchNorm2d(1),
            nn.Dropout2d(),
            nn.Conv2d(1, 1, kernel_size=5, stride=2),
        )  # [batch_size, 1, 10, 10]
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1, 1, kernel_size=5, stride=2),
            nn.Dropout2d(),
            nn.BatchNorm2d(1),
            nn.ConvTranspose2d(1, 3, kernel_size=5, stride=2),
            nn.Dropout2d(),
            nn.BatchNorm2d(3),
            nn.ConvTranspose2d(3, 3, kernel_size=5, stride=2),
            nn.Dropout2d(),
            nn.BatchNorm2d(3),
            # output_padding is needed to bring the image size back to [416, 416]
            nn.ConvTranspose2d(3, 3, kernel_size=5, stride=2, output_padding=1),
            nn.Dropout2d(),
            nn.BatchNorm2d(3),
            nn.ConvTranspose2d(3, 3, kernel_size=5, stride=2, output_padding=1),
            nn.Dropout2d(),
            nn.BatchNorm2d(3),
        )  #  [batch_size, 3, 416, 416]

    def forward(self, image, style_embedding=None):
        if self.use_embedding and style_embedding:
            batch_size = image.shape[0]
            encoded = self.encoder(image)  # [batch_size, 1, 10, 10]
            flattened = encoded.view(batch_size, 100)  # [batch_size, 100]
            # Concat the two layers
            concatted = torch.cat(
                [flattened, style_embedding], axis=1
            )  # [batch_size, 200]
            linear = nn.Linear(200, 100)(concatted)  # [batch_size, 100]
            # Reshape to 1x10x10 and pass through the decoder
            embedding_layer = linear.view(
                batch_size, 1, 10, 10
            )  # [batch_size, 1, 10, 10]

            return self.decoder(embedding_layer)
        else:
            return self.decoder(self.encoder(image))

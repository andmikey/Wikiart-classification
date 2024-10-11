# WikiArt classification 

## Assignment
### Bonus A - Make the in-class example actually learn something (5 points)

Yes! We are actually starting with a bonus task! That is, you can skip this if you don't want to.  The in-class WikiArt demo did not actually learn the intended function.  As in, we got a maximum of 2% multiclass accuracy before convergence.  Update the model and architecture to see if you can get better performance.  If you get better than 5% accuracy, you can get the points on this task.  However, you are not allowed to use any pretrained models or transformers, just the basic PyTorch classes. You can copypaste your code from Assignment 1 too, it's OK!  Report on what you did and whether it worked.

### Part 0 - Documentation (3 points)

This isn't really a part, but we'll evaluate documentation quality, as a treat -- of course, you have to document to get any points at all, as we have to know what you did, but we'll also subjectively evaluate documentation quality on a scale like this:

0 - very difficult to figure out what you did

3 - easy to follow, understand, and concise while being descriptive

with 1 and 2 being somewhere in the middle.  This is, of course, a somewhat subjective rating, but it's to incentivize paying attention to how you report things.  It's only a handful of points.

### Part 1 - Fix class imbalance (5 points)

The data has very imbalanced classes which may lead to problematic performance.  From Bonus A, or from scratch as you like, train the model after finding a way to address the class imbalance and test the model for classification accuracy.  You can use any method you like to address the class imbalance.  Briefly document the method and the results in a Markdown report in the README of your repo or linked from the README.

### Part 2 - Autoencode and cluster representations (13 points)

We will do things other than classification in this task, so you should create a *different* training script and a new model class to allow for it (as in, a script separate from train.py and a new class in wikiart.py if you're using the in-class code as a base).

    Using the standard PyTorch classes, develop an autoencoder that produces compressed representations of each image in the dataset. (Hint: the representations are going to be the activations of one of the model layers after training. Also consider what loss you might need to use.) Document and explain what you did concisely as well as how you measured the progress of the autoencoder.
    Save and cluster the representations using clustering methods from scikit-learn. (You may have to convert the tensors back to numpy).  Using matplotlib, graph the clusters dimensionality-reduced to 2D (by PCA, t-SNE, SVD,or any other method), colour and/or shape-coded by their class.  The script will have to output an image, which you can add to your report.  Do art styles cluster well in your model? 

Document everything concisely.

### Part 3 - Generation/style transfer (9 points)

In this part, you're going to write another script that augments the autoencoder in part 2 to make a crude art generator conditioned on trained embeddings for the art styles.  That is, the model will take another input, that represents the art style alongside the input image.  Then you're going to test this with "mismatched" input artworks and style embeddings and see what the output looks like. (It won't be good, you need a much more elaborate network like a Generative Adversarial Network (GAN) and a lot of training to make this work.)  Report what you did in a concise manner alongside subjective impressions of the output (including if it's trivial, like it does nothing or it looks like noise).  

### Bonus B - Generation but with a pre-trained model (10 points)

Write a script to fine-tune an existing pre-trained model from HuggingFace so that it generates art in the styles in the WikiArt dataset.  You can use LoRA to make it more compressed and efficient (the tools for LoRA should be installed on the mltgpu machines.  You might need mltgpu-2 for this, depending on how you implement it.  Document what you did in the report and subjective impressions of the generation quality.
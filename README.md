# Adversarial Images RootedCon 2018

The objective of this project is to show how we can programatically create an adversarial image using tensorflow, i.e., be able to fool a computer vision trained model, and make it return the classification that we wont for an image. 
The inception V3 trained by Google is the model choosen for this project.

Example:

The program classify a rifle in a correct way:
![Alt text](/screenshots/screenshot01.png?raw=true)

And makes little changes on the image so cannot be detected by human eye, but enough to fool the model. In the example, we want the inception V3 to classify our rifle as a Hot Dog:
![Alt text](/screenshots/screenshot02.png?raw=true)

## Setup

You'lle need python, and an environment with tensorflow, pillow, numpy and matplotlib. 
I'f you're new to python, I recommend you to download the anaconda version.

Download the code, and download the inception V3 from google. You'll find it here:
https://github.com/tensorflow/models/tree/master/research/slim

Unzip the file into the root folder of the project, this is the pretrained model.

## Using the project

Once you have all the setup done, try running `python main.py` to check the provided example.
If you want to test with your own photos and target classes, do as follow:

Put the image in the input folder.
Edit main.py, you'll find some configuration parameters:

![Alt text](/screenshots/screenshot03.png?raw=true)

Change the imgPath to your chosen image.

For the targetClass, go to the file inception.json, that is the file containing the 1000 different classes that the trained model is able to identify, choose your target class, and substract 2 from the line number of the class. Example, if the class is 'hot dog' that you can find in the line 936, so the targetClass should be 934

![Alt text](/screenshots/screenshot04.png?raw=true)




import PIL
from classifier import Classifier
from show import show

imgPath = './rifle2.jpg'
targetClass = 934
maxSteps = 100
learningRate = 0.1
maxEpsilon = 2.0/255.0
maxLoss = 0.01

classifier = Classifier(299, 299, 1000)
print('Loading inception v3')
classifier.restore('inception_v3.ckpt', 'imagenet.json')
print('Inception v3 loaded')
image = PIL.Image.open(imgPath)
classification, img = classifier.classify(image)
topprobs, toplabels = classifier.getBest(classification)
show(img, toplabels, topprobs)

adv = classifier.calculateAdversarial(img, targetClass, maxSteps, learningRate, maxEpsilon, maxLoss)
classification, img = classifier.classify(adv)
topprobs, toplabels = classifier.getBest(classification)
show(img, toplabels, topprobs)

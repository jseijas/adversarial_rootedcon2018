import json
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
import numpy as np

class Classifier(object):
  def __init__(self, inputWidth, inputHeight, numClasses):
    tf.logging.set_verbosity(tf.logging.ERROR)
    self.inputWidth = inputWidth
    self.inputHeight = inputHeight
    self.numClasses = 1000
    self.session = tf.InteractiveSession()
    self.defaultImage = tf.Variable(tf.zeros((self.inputWidth, self.inputHeight, 3)))
    self.initTopology()
  
  def initTopology(self):
    preprocessed = tf.multiply(tf.subtract(tf.expand_dims(self.defaultImage, 0), 0.5), 2.0)
    arg_scope = nets.inception.inception_v3_arg_scope(weight_decay = 0.0)
    with slim.arg_scope(arg_scope):
      logits, _ = nets.inception.inception_v3(preprocessed, self.numClasses + 1, is_training = False, reuse = False)
      self.logits = logits[:,1:]
      self.probs = tf.nn.softmax(logits)

  def restore(self, modelFileName, labelsFileName):
    self.saver = tf.train.Saver([
      var for var in tf.global_variables()
      if var.name.startswith('InceptionV3/')
    ])
    self.saver.restore(self.session, modelFileName)
    with open(labelsFileName) as f:
      self.labels = json.load(f)

  def prepareImage(self, image):
    if hasattr(image, 'width'):
      isWide = image.width > image.height
      if (isWide):
        newWidth = int(image.width * self.inputWidth / image.height)
        newHeight = int(image.height * self.inputHeight / image.width)  
      else:
        newWidth = self.inputWidth
        newHeight = self.inputHeight
      img = image.resize((newWidth, newHeight)).crop((0, 0, self.inputWidth, self.inputHeight))
      return (np.asarray(img) / 255.0).astype(np.float32)
    else:
      return image

  def classify(self, image):
    img = self.prepareImage(image)
    p = self.session.run(self.probs, feed_dict={self.defaultImage: img})[0]
    return p, img

  def getBest(self, classification):
    topten = list(classification.argsort()[-10:][::1])
    topprobs = classification[topten]
    toplabels = [self.labels[i - 1][:15] for i in topten]
    return topprobs, toplabels

  def calculateAdversarial(self, image, targetClass, maxSteps, lr, maxEpsilon, maxLoss):
    x = tf.placeholder(tf.float32, (self.inputWidth, self.inputHeight, 3))
    xPrime = self.defaultImage
    assignOp = tf.assign(xPrime, x)
    learningRate = tf.placeholder(tf.float32, ())
    yPrime = tf.placeholder(tf.int32, ())
    labels = tf.one_hot(yPrime, 1000)
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=[labels])
    optimStep = tf.train.GradientDescentOptimizer(learningRate).minimize(loss, var_list=[xPrime])
    epsilon = tf.placeholder(tf.float32, ())
    below = x - epsilon
    above = x + epsilon
    projected = tf.clip_by_value(tf.clip_by_value(xPrime, below, above), 0, 1)
    with tf.control_dependencies([projected]):
      projectStep = tf.assign(xPrime, projected)
    self.session.run(assignOp, feed_dict={x: image})

    for i in range(maxSteps):
      _, loss_value = self.session.run([optimStep, loss], feed_dict={learningRate: lr, yPrime: targetClass})
      self.session.run(projectStep, feed_dict={x: image, epsilon: maxEpsilon})
      print('step %d, losss=%g' % (i+1, loss_value))
      if (loss_value < maxLoss):
        break
    return xPrime.eval()

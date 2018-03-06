import matplotlib.pyplot as plt
from matplotlib import gridspec

def show(image, toplabels, topprobs):
  fig = plt.figure(figsize=(8, 6))
  spec = gridspec.GridSpec(1, 3, width_ratios=[2, 1, 1])
  axImage = plt.subplot(spec[0])
  axImage.set_xticks([])
  axImage.set_yticks([])

  axProbs = plt.subplot(spec[2])
  axImage.imshow(image, interpolation='nearest', aspect='auto')
  fig.sca(axImage)
  plt.sca(axProbs)
  barlist = axProbs.barh(range(10), topprobs)
  plt.yticks(range(10), toplabels, rotation='horizontal')
  fig.subplots_adjust(bottom=0.3)
  plt.show()

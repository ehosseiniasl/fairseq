import matplotlib.pyplot as plt
import numpy as np

bleu = np.load('bleu.npy').ravel()
logprob = np.load('logprob.npy').ravel()

plt.xlabel('oracle BLEU')
plt.ylabel('logprob')
plt.plot(bleu, logprob, 'o')

plt.show()

# Generating adversarial examples for MNIST

Done as coursework for Mathematics in Action (2016) at the University of Edinburgh.  
See the report for results.

Reproducing results from Goodfellow, Ian J., Jonathon Shlens, and Christian Szegedy. "Explaining and harnessing adversarial examples." arXiv preprint arXiv:1412.6572 (2014).

Code based on Theano tutorials: https://github.com/Newmu/Theano-Tutorials  

MNIST data from Yann LeCun's website: http://yann.lecun.com/exdb/mnist/   

## Usage  
1. To train the simple neural network, set TRAINING to True. The weights are then saved with cPickle and for following runs can be loaded from file by setting TRAINING back to False. 
2. The magnitude of adversarial noise added can be set as the eps parameter.

## Comments
One thing to point out from the report is that once the L1 and L2 regularization were added to the model, they were tested on the adversarial examples generated from the network without them. I realized this only after submitting. Hence the results in the report do not directly show how regularized network handles its own adversarial examples. By running it again it seems that the correct results are: 

| Dataset | Epsilon | Accuracy |
| --- | --- | --- |
| l1+l2 adversarial | 0.10 | 79.36%  |
| l1+l2 adversarial | 0.25 | 76.30%  |

But I did these hastily and they do sound like a very good improvement, so perhaps check them before believing them.






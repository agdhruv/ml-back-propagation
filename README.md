# ml-back-propagation

This is an object-oriented implementation of the back propogation algorithm for machine learning from the grounds up (only pre-installed python libraries have been used).

This implementation is only for a single hidden layer, but variable number of neurons in that layer. This is because in practice, this algorithm does not work for a deep network due to the vanishing gradient problem.

## Steps to run

The program needs to be given the number of hidden layer neurons as a command line argument.

Run using **Python 2.7**. Since it does not require any external package, there is no `requirements.txt`.

Steps to run:
```bash
$ python learn.py 12
```
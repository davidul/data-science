import numpy as np
import matplotlib.pyplot as plt

# Softmax function
def softmax(z):
    nom = np.exp(z)
    den = np.sum(np.exp(z))
    return nom/den
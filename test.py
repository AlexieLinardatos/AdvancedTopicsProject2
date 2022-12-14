
import numpy as np
import matplotlib.pyplot as plt

x = [0,1]
y1 = [0.98,0.99]

figure, axis = plt.subplots(2, 1)

axis[0].plot(x,y1,'o')
axis[0].set_title("Accuracy")

y2 = [0.98,0.99]

axis[1].plot(x,y2,'o')
axis[1].set_title("Loss")

plt.savefig('figure.png')
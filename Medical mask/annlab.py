import tensorflow as tf
import matplotlib.pyplot as plt
x = tf.linspace(0.0, 1.0, 100)

weight = 2.0
bias = 0.5

def linear_model(x, w, b):
    return w * x + b

y = linear_model(x, weight, bias)

plt.plot(x, y, label=f'y = {weight}x + {bias}')

new_weights = [1.0, 3.0]

for w in new_weights:
    y_new = linear_model(x, w, bias)
    plt.plot(x, y_new, label=f'y = {w}x + {bias}')
    
new_biases = [-0.5, 1.5]
for b in new_biases:
    y_new = linear_model(x, weight, b)
    plt.plot(x, y_new, label=f'y = {weight}x + {b}')
    
plt.xlabel('Input (x)')
plt.ylabel('Output (y)')
plt.title('Effect of Weights and Bias on Linear Model Output')
plt.legend()
plt.grid(True)
plt.show()
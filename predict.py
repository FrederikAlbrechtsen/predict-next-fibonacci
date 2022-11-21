import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

def split_sequence(seq, n_steps):
    x,y = [], []
    
    for i in range(len(seq)):
        last_index = i + n_steps
        
        if last_index > len(seq) - 1:
            break
        
        seq_x, seq_y = seq[i:last_index], seq[last_index]
        
        x.append(seq_x)
        y.append(seq_y)
    
    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
    return x,y

def F(n):
    if n == 0: 
        return 0
    elif n == 1: 
        return 1
    else: 
        return F(n - 1) + F(n - 2)

# The 13 first fibonacci numbers
data = [1,1,2,3,5,8,13,21,34,55,89,144,233]
n_steps = 5
x, y = split_sequence(data, n_steps)

# LSTM need input of the form [batch, timesteps, features]
# hence we need to shape the the data x
n_features = 1
x = x.reshape((x.shape[0], x.shape[1], n_features))

model = tf.keras.Sequential()
model.add(layers.LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
model.add(layers.Dense(1))

# Compile the model
model.compile(
            optimizer=tf.keras.optimizers.Adam(0.01), 
            loss=tf.keras.losses.MeanSquaredError(), 
            metrics=['accuracy']
)

# Train the model
model.fit(x, y, epochs=200, verbose=1)

model.summary()
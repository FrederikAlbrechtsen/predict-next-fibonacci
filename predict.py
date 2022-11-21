import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

def split_sequence(sequence: list, n_steps: int):
    x,y = [], []
    
    for i in range(len(sequence)):
        last_index = i + n_steps
        
        if last_index > len(sequence) - 1:
            break
        
        sequence_x, sequence_y = sequence[i:last_index], sequence[last_index]
        
        x.append(sequence_x)
        y.append(sequence_y)
    
    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
    return x,y

def fibonacci(n: int):
    if n == 0: 
        return 0
    elif n == 1: 
        return 1
    else: 
        return fibonacci(n - 1) + fibonacci(n - 2)

# The 13 first fibonacci numbers
data = [1,1,2,3,5,8,13,21,34,55,89,144,233]
n_steps = 5
x, y = split_sequence(data, n_steps)

# Reshape the data x
n_features = 1
x = x.reshape((x.shape[0], x.shape[1], n_features))

model = tf.keras.Sequential()
model.add(layers.LSTM(64, activation='relu', input_shape=(n_steps, n_features)))
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

# Predictions
test_data = np.array([89,144,233,377,610], dtype=np.float32)
test_data = test_data.reshape((1, n_steps, n_features))

# Predicting the 16th number
predict_next_fib = model.predict(test_data, verbose=1)

converted = [[int(num) for num in sub] for sub in predict_next_fib]

flat_list = [item for sublist in converted for item in sublist]

strings = [str(integer) for integer in flat_list]
a_string = "".join(strings)
predicted_fib = int(a_string)

print("The predicted next fibonacci number (the 16th): ", predicted_fib)

# The correct 16th fibonacci number
print("The actual 16th fibonacci number:", fibonacci(16))

# The difference from the predicted fibonacci number and the correct
print("Difference is", abs(predicted_fib - fibonacci(16)))
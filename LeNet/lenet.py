import tensorflow as tf

def create_lenet5():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters= 6, strides= (1,1),
        kernel_size= (5,5), activation= 'tanh', input_shape=(28, 28, 1)), # input_shape=(28, 28, 1), since I used MNIST data for training
        
        tf.keras.layers.AveragePooling2D(pool_size= (2,2),strides= (2,2)),
        
        tf.keras.layers.Conv2D(filters= 6, strides= (1,1),
        kernel_size= (5,5), activation= 'tanh'),

        tf.keras.layers.AveragePooling2D(pool_size= (2,2),strides = (2,2)),
        
        tf.keras.layers.Dense(120, activation= 'relu'),
        
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(84, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    return model

if __name__ == "__main__":
    model = create_lenet5()
    model.summary()

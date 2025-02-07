import tensorflow as tf
import pandas as pd
import numpy as np

# Procesing the Data
train = pd.read_csv('/home/kandarpa-sarkar/Desktop/MNIST/mnist_train.csv')
test = pd.read_csv('/home/kandarpa-sarkar/Desktop/MNIST/mnist_test.csv')

X_train = train.drop('label', axis= 1).values
y_train = train['label'].values

X_test = test.drop('label', axis= 1).values
y_test = test['label'].values

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

y_train = np.eye(10)[y_train]
y_test = np.eye(10)[y_test]

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

X_train/= 255.0 # Normalize data
X_test/= 255.0

def LeNet_tf(Input_shape):
    model = tf.keras.Sequential()

    # Convolution Layer 1
    model.add(tf.keras.layers.Conv2D(filters= 6, strides= (1,1),
        kernel_size= (5,5), activation= 'tanh', input_shape= Input_shape))
    
    # Subsampling Layer 1
    model.add(tf.keras.layers.AveragePooling2D(pool_size= (2,2),strides= (2,2)))

    # Convolution Layer 2
    model.add(tf.keras.layers.Conv2D(filters= 6, strides= (1,1),
        kernel_size= (5,5), activation= 'tanh'))

    # Subsampling Layer 2
    model.add(tf.keras.layers.AveragePooling2D(pool_size= (2,2),strides = (2,2)))    

    model.add(tf.keras.layers.Dense(120, activation= 'relu'))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(84, activation= 'relu'))
    model.add(tf.keras.layers.Dense(10, activation= 'softmax'))


    model.compile(loss= 'categorical_crossentropy', optimizer = tf.keras.optimizers.RMSprop(learning_rate = 0.001), metrics= ['accuracy'])
    
    return model


lenet = LeNet_tf((28,28,1))
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='model/best_model.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max', 
    verbose=1
)
history = lenet.fit(X_train, y_train, epochs= 5, batch_size=64, validation_data = (X_test, y_test), callbacks=[checkpoint_callback])

model = tf.keras.models.load_model("model/best_model.keras")


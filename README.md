Course material from Pluralsight course: [
Build a Machine Learning Workflow with Keras TensorFlow 2.0](https://app.pluralsight.com/library/courses/build-machine-learning-workflow-keras-tensorflow/table-of-contents)

# Requirements
- Python 3.8
- see requirements.txt

# Key points

## Training an NN model in Keras:
After preparing features, scaling, train test val split:
1. Build model - define layers, infer input shape, dropout, activation
2. Instantiate model
3. Compile model - define loss, optimizer, evaluation metrics
e.g. 
```python
from tensorflow import keras
from tensorflow.keras import layers

tf.keras.backend.clear_session() # Destroys the current TF graph session and creates a new one - use if using same Python kernel

def build_and_compile_model():
    model = tf.keras.Sequential([
        layers.Dense(32, activation='relu', input_shape=[len(X_train.columns)]), # fully connected layer with 32 neurons
        layers.Dropout(0.25)
        layers.Dense(64, activation='relu'), 
        layers.Dense(1) #output layer
    ])
    
    optimizer = tf.keras.optimizers.Adam(0.001) 

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(), # appropriate loss function for binary classification,
                  optimizer=optimizer,
                  metrics=['accuracy', 
                       tf.keras.metrics.Precision(0.5), # 0.5 probability threshold
                       tf.keras.metrics.Recall(0.5),])
    return model
```
4. Inspect model structure - `model.summmary()`, `keras.utils.plot_model(model, 'model.png', show_shapes=True)`, `model.layers()`
5. Fit model - define n_epochs, validation_split, any callbacks - `training_history = model.fit()`
    * View training history - `training_history.history.keys()`
6. Make predictions - `model.predict(X_test).flatten()`

## Larger datasets
Convert datasets to tf.data.Dataset format - suitable for large data sets (enables distributed training processes)
```python
dataset_train = tf.data.Dataset.from_tensor_slices((x_train.values, y_train.values))
dataset_train = dataset_train.batch(16) # batch size 16 records

dataset_train.shuffle(128) # shuffle training data with buffer size 128
```
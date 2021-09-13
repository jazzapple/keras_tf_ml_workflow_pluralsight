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
def build_and_compile_model():
    
    inputs = tf.keras.Input(shape=(x_train.shape[1],)) # define input layer

     # every layer instance is a callable. Accepts inputs and returns a tensor
    x = layers.Dense(16, activation='relu')(inputs) # fully connected layer

    x = layers.Dropout(0.3)(x)

    x = layers.Dense(8, activation='relu')(x)

    predictions = layers.Dense(1, activation='sigmoid')(x) # final layer with 1 neuron with probability
    
    # instantiate model
    model = tf.keras.Model(inputs=inputs, outputs=predictions)
    
    model.summary()
    
    model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001), # adaptive learning optimizer. learning rate 0.001
              loss=tf.keras.losses.BinaryCrossentropy(), # appropriate loss function for binary classification
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
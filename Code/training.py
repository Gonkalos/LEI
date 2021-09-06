import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
import matplotlib.pyplot as plt
import itertools

'''
Define the metric Precision
Percentage of correct classifications from all values classified as positive
'''
def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

'''
Define the metric Recall
Percentage of positive classes correctly classified
'''
def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

'''
Define Dice loss function
'''
def dice_loss(y_true, y_pred, smooth=1e-6):
    # convert types
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    # Dice coefficient
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    coefficient = (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)
    # Dice loss
    return 1 - coefficient

'''
Compile model and fit it to data
'''
def compile_fit(model, loss, config, x_train, y_train, x_val, y_val):
    learning_rate, epochs, batch_size = config
    # compile model
    if loss == 'binary_crossentropy':
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
                      loss='binary_crossentropy',
                      metrics=[precision, recall])
    elif loss == 'dice':
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
                      loss=dice_loss,
                      metrics=[precision, recall])
    # fit model to data
    history = model.fit(x_train, y_train, 
              validation_data=(x_val, y_val),                                                                                 
              epochs=epochs, batch_size=batch_size, shuffle=True)
    # plot learning curves
    plot_learning_curves(history.history['loss'], history.history['val_loss'])
    return model

'''
Predict changes from a pair of images
'''
def predict_changes(model, images, image_size):
    # reshape input data to fit the model
    input_data = images.reshape(-1, image_size, image_size, 2)
    # predict changes
    prediction = model.predict(input_data)
    prediction = prediction > 0.5
    prediction = prediction.reshape(image_size, image_size)
    return prediction

'''
Plot learning curves
'''
def plot_learning_curves(loss, val_loss):
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

'''
Generate hiperparameters configurations
'''
def generate_configs(learning_rate, epochs, batch_size):
    configs = [learning_rate, epochs, batch_size]
    configs = list(itertools.product(*configs))
    print('Generated %s different configurations' % (len(configs)))
    return configs 

'''
Grid Search for hiperparameters
'''
def grid_search(model, loss, configs, x_train, y_train, x_val, y_val, x_test, y_test):
    # evaluate configs
    df_scores = pd.DataFrame(columns = ['learning_rate', 'epochs', 'batch_size', 'loss', 'precision', 'recall'])
    for config in configs:
        model = compile_fit(model, loss, config, x_train, y_train, x_val, y_val)
        metrics = model.evaluate(x_test, y_test)
        new_row = {'learning_rate':config[0], 'epochs':config[1], 'batch_size':config[2], 'loss':metrics[0], 'precision':metrics[1], 'recall':metrics[2]}
        df_scores = df_scores.append(new_row, ignore_index=True)
    # store scores
    df_scores.to_csv('./scores.csv')
import os
import time
import logging
import datetime as dt
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, Input
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model

from core.utils import image_output_dir

class LSTMTimeSeriesModel:
    '''
    Class for building the LSTM
    '''
    def __init__(self):
        self.model = None

    def load_model(self, filepath):
        '''
        Loading the model from a filepath
        '''      
        logging.info(f"Loading model from {filepath}")
        self.model = load_model(filepath)
    
    def build_model(self, config):
        '''
        Function to build the model from a config file
        '''
        logging.info("[MODEL]: Building model...")
        now = time.time()
        #
        # input_layer = Input(shape=(None,))
        bottom = Sequential()
        # bottom.add(input_layer)
        for layer in config['model']['layers']:
            units = layer['units'] if 'units' in layer else None
            dropout = layer['dropout'] if 'dropout' in layer else None
            activation = layer['activation'] if 'activation' in layer else None
            seq_len = layer['seq_len'] - 1 if 'seq_len' in layer else None         
            num_features = layer['num_features'] if 'num_features' in layer else None
            layer_type = layer['type'] if 'type' in layer else None
            return_seq = layer['return_seq'] if 'return_seq' in layer else None
            
            if layer_type == 'Dense':
                bottom.add(Dense(units=units, activation=activation))
            elif layer_type == "LSTM":
                bottom.add(LSTM(units=units,
                                    activation=activation, 
                                    input_shape=(seq_len, num_features),
                                    return_sequences=return_seq
                                   ))
            elif layer_type == "GRU":
                bottom.add(GRU(units=units,
                                    activation=activation,
                                    input_shape=(seq_len, num_features),
                                    return_sequences=return_seq
                                    ))
            elif layer_type == "Dropout":
                bottom.add(Dropout(rate=dropout))

        regressor_output = Dense(3, activation='tanh', name='regressor') (bottom.output)
        classfier_output = Dense(2, activation='softmax', name='classifier') (bottom.output)

        self.model = Model(inputs=bottom.input, outputs=[
            # bottom.output,
            regressor_output,
            classfier_output
        ])

        self.model.summary()
        plot_model(self.model, os.path.join(image_output_dir,"model.png"), show_shapes=True)

        self.model.compile(loss={
            # 'dense': config['model']['loss'],
            'regressor': config['model']['loss'],
            'classifier': 'categorical_crossentropy'
        },
                           optimizer=config['model']['optimizer'])
        
        time_taken = time.time() - now    
        logging.info(f"Model Building complete in {time_taken//60} min and {(time_taken % 60):.1f} s")
    
    def train(self, x_train, y_train_regressor, y_train_classifier, config):
        '''
        Function to train model
        '''
        epochs = config["training"]["epochs"]
        batch_size = config["training"]["batch_size"]
        save_dir = config["model"]["save_dir"]
        
        save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
        callbacks = [
            ModelCheckpoint(filepath=save_fname, **config["model"]["checkpoint_params"]),            
            ReduceLROnPlateau(**config["model"]["reduce_lr_params"]),          
            EarlyStopping(**config["model"]["early_stopping_params"]),  
        ]
        logging.info("[MODEL]: Training started")
        history = self.model.fit(
                    x_train,
            # y_train_regressor,
                    {
                        'regressor': y_train_regressor,
                        'classifier': y_train_classifier
                    },
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=config["training"]["val_split"],
                    callbacks=callbacks        
                )
        self.model.save(save_fname)
        
        logging.info(f"Model training completed. Model saved to {save_fname}")
        
        return history
    
    def predict_point_by_point(self, data):
        '''
        Making one prediction for each sequence
        '''
        logging.info('[MODEL]: Predicting Point-by-Point...')
        predicted = self.model.predict(data)
        # predicted = np.reshape(predicted, (predicted.size,))
        
        return predicted
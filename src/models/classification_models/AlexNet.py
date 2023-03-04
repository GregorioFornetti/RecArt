
import tensorflow as tf

from classification_models.Model import Model

class SimpleNeuralNetwork(Model):

    def _init_attributes(self):
        '''
        Inicializa os atributos que serão utilizados nos métodos genéricos do modelo

        Atualmente é necessário definer as variáveis:
        - model_name
        - img_shape
        '''
        self.model_name = 'AlexNet'
        self.img_shape = (224, 224, 3)
    
    def _create_neural_network(self):
        '''
        Inicializa a estrutura da rede neural
        '''
        self.model = tf.keras.models.Sequential()

        self.model.add(tf.keras.layers.Conv2D(96, 11, strides=4, padding='same'))
        self.model.add(tf.keras.layers.Lambda(tf.nn.local_response_normalization))
        self.model.add(tf.keras.layers.Activation('relu'))

        self.model.add(tf.keras.layers.MaxPooling2D(3, strides=2))

        self.model.add(tf.keras.layers.Conv2D(256, 5, strides=4, padding='same'))
        self.model.add(tf.keras.layers.Lambda(tf.nn.local_response_normalization))
        self.model.add(tf.keras.layers.Activation('relu'))

        self.model.add(tf.keras.layers.MaxPooling2D(3, strides=2))

        self.model.add(tf.keras.layers.Conv2D(384, 3, strides=4, padding='same'))
        self.model.add(tf.keras.layers.Activation('relu'))

        self.model.add(tf.keras.layers.Conv2D(384, 3, strides=4, padding='same'))
        self.model.add(tf.keras.layers.Activation('relu'))

        self.model.add(tf.keras.layers.Conv2D(256, 3, strides=4, padding='same'))
        self.model.add(tf.keras.layers.Activation('relu'))

        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(4096, activation='relu'))
        self.model.add(tf.keras.layers.Dropout(0.5))

        self.model.add(tf.keras.layers.Dense(4096, activation='relu'))
        self.model.add(tf.keras.layers.Dropout(0.5))

        self.model.add(tf.keras.layers.Dense(self.OUTPUT_SIZE, activation='sigmoid'))

        self.model.compile(optimizer='adam',
                           loss='binary_crossentropy',
                           metrics=['accuracy'])

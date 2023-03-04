
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
        self.model_name = 'SimpleNeuralNetwork'
        self.img_shape = (256, 256, 3)
    
    def _create_neural_network(self):
        '''
        Inicializa a estrutura da rede neural
        '''
        self.model = tf.keras.models.Sequential()

        self.model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.img_shape))
        self.model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        self.model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        self.model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))

        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(64, activation='relu'))
        self.model.add(tf.keras.layers.Dense(self.OUTPUT_SIZE, activation='sigmoid'))

        self.model.compile(optimizer='adam',
                           loss='binary_crossentropy',
                           metrics=['accuracy'])


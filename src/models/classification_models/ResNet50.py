
import tensorflow as tf

from classification_models.Model import Model

class ResNet50(Model):

    def _init_attributes(self):
        '''
        Inicializa os atributos que serão utilizados nos métodos genéricos do modelo

        Atualmente é necessário definer as variáveis:
        - model_name
        - img_shape
        '''
        self.model_name = 'ResNet50'
        self.img_shape = (256, 256, 3)
    
    def _create_neural_network(self):
        '''
        Inicializa a estrutura da rede neural
        '''
        self.model = tf.keras.applications.ResNet50(classes=self.OUTPUT_SIZE, weights=None, input_shape=self.img_shape, classifier_activation='sigmoid')
        self.model.compile(optimizer='adam',
                           loss='binary_crossentropy',
                           metrics=['accuracy'])

        

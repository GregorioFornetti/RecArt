
import numpy as np

from classification_models.Model import Model

class OnlyZerosModel(Model):

    def _init_attributes(self):
        '''
        Inicializa os atributos que serão utilizados nos métodos genéricos do modelo

        Atualmente é necessário definer as variáveis:
        - model_name
        - img_shape
        '''
        self.model_name = 'OnlyZeros'
        self.img_shape = (256, 256, 3)
    
    def _create_neural_network(self):
        '''
        Inicializa a estrutura da rede neural
        '''
        pass

    def predict_proba(self, img_path):
        return np.zeros(self.OUTPUT_SIZE)

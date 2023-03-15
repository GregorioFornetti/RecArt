from rec_models.RecModel import RecModel
from classification_models.ResNet50 import ResNet50

class RecResNet50(RecModel):

    def _init_params(self):
        '''
        Inicializa os parâmetros que serão utilizados no __init__ para inicializar o
        modelo.

        Os atributos que precisam ser inicializados são:
        - model
        '''
        self.model = ResNet50()
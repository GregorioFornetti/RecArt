from rec_models.RecModel import RecModel
from classification_models.KNN import KNN

class RecKNN(RecModel):

    def _init_params(self):
        '''
        Inicializa os parâmetros que serão utilizados no __init__ para inicializar o
        modelo.

        Os atributos que precisam ser inicializados são:
        - model
        '''
        self.model = KNN()
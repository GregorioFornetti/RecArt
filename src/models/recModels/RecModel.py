from abc import ABC, abstractclassmethod

def RecModel(ABC):

    @abstractclassmethod
    def _init_params(self):
        '''
        Inicializa os parâmetros que serão utilizados no __init__ para inicializar o
        modelo.
        '''
        pass

    def __init__(self):
        '''
        Inicializa o modelo de recomendações
        '''
        pass

    def predict(self, image_path):
        '''
        Irá prever os rotulos/classes da imagem fornecida, e recomendar imagens semelhantes

        ---

        ## Parâmetros

        ---

        image_path: String
            Caminho relativo até a imagem que será classificada

        --- 

        ## Retorno

        ---

        Um dicionário seguindo a seguinte estrutura:
            {
            "predictions": { 
                "selected":
                    [
                        {
                            "nome": <nome movimento artístico>
                            "probability": <probabilidade da imagem pertencer ao movimento ártistico (valor real entre 0 e 1)>
                        }
                        ...
                    ]
                "not_selected":
                    [
                        {
                            "nome": <nome movimento artístico>
                            "probability": <probabilidade da imagem pertencer ao movimento ártistico (valor real entre 0 e 1)>
                        }
                    ]
                }
            "similarImages": <DataFrame contendo informações de similaridade da imagem atual para cada imagem no dataset. Esse
                              DataFrame vai estar ordenado para que a primeira imagem seja a mais semelhante e assim por diante.>
            }
        '''
        pass
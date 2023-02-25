from abc import ABC, abstractclassmethod

class Model(ABC):

    @abstractclassmethod
    def __init__(self, load=True):
        '''
        Responsável por inicializar a rede neural.

        ## Parâmetros

        ---

        load: Boolean (default=True)
            Caso seja verdadeiro, irá carregar o modelo de rede neural salvo, caso contrário, irá
            instanciar a rede neural com pesos padronizados (sendo necessário treina-la).
        ---
        '''
        pass
    
    def __load(self):
        '''
        Carrega o modelo salvo em "save_path"
        '''
        pass

    def fit(self, data, batch_size=32, epochs=10, save=True):
        '''
        Treina o modelo

        ---

        ## Parâmetros

        ---

        data: DataFrame
            DataFrame que será utilizado para treinamento do modelo.

        ---

        batch_size: int (default=32)
            Tamanho do lote de treinamento
        
        ---
        
        epochs: int (default=10)
            Número de epochs para o treinamento
        
        ---

        save: Boolean (default=True)
            Se for verdadeiro, salvará os pesos aprendidos
        
        ---
        '''
        pass

    def predict(self, img_path):
        '''
        Lê e ajusta a imagem e faz predições.

        ## Parâmetros

        ---

        img_path: String
            Caminho relativo para a imagem que irá ser classificada
        
        ---

        ## Retorno

        ---

        Retorna uma lista de valores que podem ser 0 ou 1. Se o valor for 0, a imagem não faz parte da classe, se for 1, sim.
        Ex: lista[0] = 1 (a imagem pertence ao gênero na posição 0) e lista[0] = 0 (a imagem não pertence ao gênero)
        '''
        pass

    def predict_proba(self, img_path):
        '''
        Lê e ajusta a imagem e faz predições.

        ## Parâmetros

        ---

        img_path: String
            Caminho relativo para a imagem que irá ser classificada
        
        ---

        ## Retorno

        ---

        Retorna uma lista de probabilidades, variando de 0 a 1. Cada valor na lista é a probabilidade de pertencer a uma classe.
        Ex: lista[0] = 0.61 (61% de chance da imagem pertencer a classe na posição 0)
        '''
        pass

    def evaluate(self, data):
        '''
        Calcula as métricas de avaliação para a rede neural. Salva os resultados no caminho
        de "save" do modelo.

        ## Parâmetros

        ---

        data: DataFrame
            DataFrame de teste, o qual as métricas serão calculadas
        
        ---

        ## Retorno

        ---

        Retorna uma tupla contendo dois DataFrames, o primeiro DataFrame com métricas para cada classe, e o segundo
        um DataFrame de métricas gerais. Esses dois DataFrames serão salvos no "save_path" do modelo.
        '''
        pass

    def save_imgs_predictions(self, data):
        '''
        Salva as predições de probabilidades para cada imagem

        ## Parâmetros

        ---

        data: DataFrame
            DataFrame completo, para predizer as probabilidades de cada imagem pertencer a cada movimento
            ártistico
        
        --- 

        Não há retorno, apenas irá salvar no "save_path" os resultados obtidos para cada imagem.
        '''
        pass
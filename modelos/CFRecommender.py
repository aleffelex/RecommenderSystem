import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_distances


class CFTraing(object):
        
    def prepare_matrix(self, ratings):
        #Criando uma tabela dinâmica esparsa com usuários em linhas e itens em colunas
        users_books_pivot_matrix_df = ratings.pivot(index='User-ID',columns='ISBN',values='Book-Rating').fillna(0)
        users_books_pivot_matrix = users_books_pivot_matrix_df.values
        users_ids = list(users_books_pivot_matrix_df.index)
        columns = users_books_pivot_matrix_df.columns
        return users_books_pivot_matrix, users_ids, columns
    
    def training_model(self, ratings):
        pivot_matrix, users_ids, columns = self.prepare_matrix(ratings)
        # Calculando a matriz de distâncias com: cosine_distances(ratings_train)
        pivot_sparse_matrix_dist = 1 - cosine_distances(pivot_matrix)
        # Prevendo os ratings
        # Isso é feito através da multiplicação da matriz de distâncias com a matriz de ratings
        pred = pivot_sparse_matrix_dist.dot(pivot_matrix) / np.array([np.abs(pivot_sparse_matrix_dist).sum(axis=1)]).T
        #Convertendo a matriz reconstruída em um dataframe do Pandas
        preds_df = pd.DataFrame(pred, columns = columns, index=users_ids).transpose()
        return preds_df
    

class CFRecommender(object):
    
    MODEL_NAME = 'Filtros Colaborativos'
    
    def __init__(self, cf_predictions_df, items_df=None):
        self.cf_predictions_df = cf_predictions_df
        self.items_df = items_df
        
    def retorna_nome_modelo(self):
        return self.MODEL_NAME
        
    def recomenda_itens(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        # Obter e classificar as previsões do usuário
        sorted_user_predictions = self.cf_predictions_df[user_id].sort_values(ascending=False) \
                                    .reset_index().rename(columns={user_id: 'Book-Rating'})

        # Recomende os livros com a classificação mais alta prevista que o usuário ainda não leu.
        recommendations_df = sorted_user_predictions[~sorted_user_predictions['ISBN'].isin(items_to_ignore)] \
                               .sort_values('Book-Rating', ascending = False) \
                               .head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how = 'left', 
                                                          left_on = 'ISBN', 
                                                          right_on = 'ISBN')[['Book-Rating', 'ISBN', 'Book-Title', 'Book-Author']]


        return recommendations_df
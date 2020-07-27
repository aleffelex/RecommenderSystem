import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

class CFTraingSVD(object):
        
    def prepare_matrix(self, ratings):
        #Criando uma tabela dinâmica esparsa com usuários em linhas e itens em colunas
        users_books_pivot_matrix_df = ratings.pivot(index='User-ID',columns='ISBN',values='Book-Rating').fillna(0)
        users_books_pivot_matrix = users_books_pivot_matrix_df.values
        users_ids = list(users_books_pivot_matrix_df.index)
        columns = users_books_pivot_matrix_df.columns
        return users_books_pivot_matrix, users_ids, columns
    
    def training_model(self, ratings, NUMBER_OF_FACTORS_MF):
        pivot_matrix, users_ids, columns = self.prepare_matrix(ratings)
        pivot_sparse_matrix = csr_matrix(pivot_matrix)
        #Executa a fatoração da matriz da matriz do item do usuário original
        U, sigma, Vt = svds(pivot_sparse_matrix, k = NUMBER_OF_FACTORS_MF)
        sigma = np.diag(sigma)
        predicted_ratings = np.dot(np.dot(U, sigma), Vt)
        predicted_ratings_norm = (predicted_ratings - predicted_ratings.min()) / (predicted_ratings.max() - predicted_ratings.min())
        preds_df = pd.DataFrame(predicted_ratings_norm, columns = columns, index=users_ids).transpose()
        return preds_df

class CFRecommenderSVD:
    
    MODEL_NAME = 'Filtros Colaborativos SVD'
    
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

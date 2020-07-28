
class PopularityRecommender(object):
    
    NOME_MODELO = 'Popularidade'
    
    def __init__(self, ratings, itens_df=None):
        self.ratings = ratings
        self.itens_df = itens_df
        
    def retorna_nome_modelo(self):
        return self.NOME_MODELO
        
    def recomenda_itens(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        # Recomemda os itens mais populares que os usuários não viram ainda
        popularidade_df = self.ratings.groupby('ISBN')['Book-Rating'].mean().sort_values(ascending=False).reset_index()
        recommendations_df = popularidade_df[~popularidade_df['ISBN'].isin(items_to_ignore)].sort_values('Book-Rating', ascending = False).head(topn)

        if verbose:
            if self.itens_df is None:
                raise Exception('"itens_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.itens_df, how = 'left', 
                                                        left_on = 'ISBN', 
                                                        right_on = 'ISBN')[['Book-Rating', 'ISBN', 'Book-Title', 'Book-Author']]


        return recommendations_df

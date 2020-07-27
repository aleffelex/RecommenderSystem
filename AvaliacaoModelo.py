import pandas as pd

class AvaliacaoModelo(object):
    
    def __init__(self, ratings_test_index, ratings_train_index):
        self.ratings_test_index = ratings_test_index
        self.ratings_train_index = ratings_train_index

    def retorna_itens_avaliados(self, user_id, ratings):
        # Obter os dados do usuário e mesclar as informações do filme.
        itens_avaliados = ratings.loc[user_id]['ISBN']
        return set(itens_avaliados if type(itens_avaliados) == pd.Series else [itens_avaliados])

    def _verify_hit_top_n(self, item_id, itens_recomendados, topn):        
        try:
            index = next(i for i, c in enumerate(itens_recomendados) if c == item_id)
        except:
            index = -1
        hit = int(index in range(0, topn))
        return hit, index

    def avaliar_modelo_usuario(self, model, user_id):
        #Colocando os Itens no Conjunto de Teste
        itens_avaliados_teste = self.ratings_test_index.loc[user_id]
        if type(itens_avaliados_teste['ISBN']) == pd.Series:
            user_itens_avaliados_test = set(itens_avaliados_teste['ISBN'])
        else:
            user_itens_avaliados_test = set([itens_avaliados_teste['ISBN']])  
        itens_avaliados_count_test = len(user_itens_avaliados_test)

        #Obtendo uma lista de recomendações classificadas de um modelo para um determinado usuário
        user_recs_df = model.recomenda_itens(user_id,items_to_ignore=self.retorna_itens_avaliados(user_id, 
                                                                                            self.ratings_train_index), 
                                               topn=100)

        hits_at_5_count = 0
        hits_at_10_count = 0
        #Para cada item que o usuário avaliou no conjunto de testes
        for item_id in user_itens_avaliados_test:
                                                   
            valid_recs = user_recs_df['ISBN'].values
            #Verificando se o item classificado atual está entre os top-N itens recomendados
            hit_at_5, index_at_5 = self._verify_hit_top_n(item_id, valid_recs, 5)
            hits_at_5_count += hit_at_5
            hit_at_10, index_at_10 = self._verify_hit_top_n(item_id, valid_recs, 10)
            hits_at_10_count += hit_at_10

        #Recall é a taxa dos itens avaliados classificados entre os Top-N itens recomendados, 
        #quando misturado com um conjunto de itens não relevantes
        recall_at_5 = hits_at_5_count / float(itens_avaliados_count_test)
        recall_at_10 = hits_at_10_count / float(itens_avaliados_count_test)

        user_metrics = {'hits@5_count':hits_at_5_count, 
                          'hits@10_count':hits_at_10_count, 
                          'avaliacoes_count': itens_avaliados_count_test,
                          'recall@5': recall_at_5,
                          'recall@10': recall_at_10}
        return user_metrics

    def avaliar_modelo(self, model):
        people_metrics = []
        #with tqdm(total=len(ratings_test_index.index.unique().values)) as pbar:
        for idx, user_id in enumerate(list(self.ratings_test_index.index.unique().values)):
            user_metrics = self.avaliar_modelo_usuario(model, user_id)  
            user_metrics['_user_id'] = user_id
            people_metrics.append(user_metrics)
                #pbar.write('%d users processed' % idx)
                #pbar.update(1)
        print('%d usuários processados' % idx)

        detailed_results_df = pd.DataFrame(people_metrics) \
                            .sort_values('avaliacoes_count', ascending=False)

        global_recall_at_5 = detailed_results_df['hits@5_count'].sum() / float(detailed_results_df['avaliacoes_count'].sum())
        global_recall_at_10 = detailed_results_df['hits@10_count'].sum() / float(detailed_results_df['avaliacoes_count'].sum())

        global_metrics = {'modelName': model.retorna_nome_modelo(),
                            'recall@5': global_recall_at_5,
                            'recall@10': global_recall_at_10}    
        return global_metrics, detailed_results_df
    


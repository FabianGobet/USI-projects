from sklearn.metrics import silhouette_score
from k_means import k_means_clustering
from hierarchical import agglomerative_clustering
import pandas as pd
import sys

def silhouette(path_to_featurevec_csv: str, clustering_csv_path: str = None, max_clusters: int = None, min_clusters: int = 2):
    df_feature_vector = pd.read_csv(path_to_featurevec_csv).drop('method_name', axis=1).values
    
    if clustering_csv_path: 
        df_clustering = pd.read_csv(clustering_csv_path).drop('method_name', axis=1).values.ravel()
        return silhouette_score(df_feature_vector, df_clustering)
    else:
        kmean_dict = {}
        agglom_dict = {}
        agglom_dict_single = {}
        for k in range(min_clusters, max_clusters + 1):
            df_clustering = k_means_clustering(n_clusters=k, path_to_featurevec_csv=path_to_featurevec_csv, get=True,save=True).drop('method_name', axis=1).values.ravel()
            kmean_dict[k] = silhouette_score(df_feature_vector, df_clustering)
            df_clustering = agglomerative_clustering(path_to_featurevec_csv=path_to_featurevec_csv, n_clusters=k, get = True,save=True).drop('method_name', axis=1).values.ravel()
            agglom_dict[k] = silhouette_score(df_feature_vector, df_clustering)
            df_clustering = agglomerative_clustering(path_to_featurevec_csv=path_to_featurevec_csv, n_clusters=k, get = True,save=True,linkage='single').drop('method_name', axis=1).values.ravel()
            agglom_dict_single[k] = silhouette_score(df_feature_vector, df_clustering)
        return kmean_dict, agglom_dict, agglom_dict_single

def utility_print(kmean_dict, agglom_dict, agglom_single_dict):
    print('Agglomerative complete:')
    for k, v in agglom_dict.items():
        print(f'{k}, {v:.3f}')
    print('Agglomerative single:')
    for k, v in agglom_single_dict.items():
        print(f'{k}, {v:.3f}')
    print('\nK-means:')
    for k, v in kmean_dict.items():
        print(f'{k}, {v:.3f}')

if __name__ == '__main__':
    if len(sys.argv)<3:
        print('Usage: python silhouette.py path_to_featurevec_csv min_clusters max_clusters [clustering_csv_path]\nIf clustering_csv_path is provided, min_clusters and max_clusters are ignored.')
        sys.exit() 
        
    path_to_featurevec_csv = sys.argv[1]
    if len(sys.argv)>4:
        clustering_csv_path = sys.argv[4]
        print(f'{silhouette(path_to_featurevec_csv, clustering_csv_path):.3f}')
    else:
        max_clusters = int(sys.argv[3])
        min_clusters = int(sys.argv[2])
        utility_print(*silhouette(path_to_featurevec_csv, max_clusters=max_clusters, min_clusters=min_clusters))
    
    

from sklearn.cluster import AgglomerativeClustering
import pandas as pd
import sys

def agglomerative_clustering(path_to_featurevec_csv: str, n_clusters: int, save: bool = False, get: bool = False, linkage = 'complete') -> None:
    df = pd.read_csv(path_to_featurevec_csv)
    X = df.drop('method_name', axis=1).values
    kmeans = AgglomerativeClustering(n_clusters=n_clusters,linkage=linkage).fit_predict(X)

    df_for_csv = df['method_name'].to_frame()
    df_for_csv['cluster_id'] = kmeans  
    cols = df_for_csv.columns.tolist()
    df_for_csv = df_for_csv[cols[-1:] + cols[:-1]]
    
    class_name = path_to_featurevec_csv.split('/')[-1].split('.')[0]
    if save:
        df_for_csv.sort_values(by='cluster_id').to_csv(class_name + '_agglomerative_'+str(linkage)+'_'+str(n_clusters)+'.csv', index=False)
    if get:
        return df_for_csv



if __name__ == '__main__':
    if not len(sys.argv) > 2:
        print('Usage: python k_means.py <path_to_featurevec_csv> <n_clusters> [optional: linkage = \'single \']')
        sys.exit(1)

    if len(sys.argv) == 4:
        agglomerative_clustering(sys.argv[1], int(sys.argv[2]), linkage=sys.argv[3])
    else:
        agglomerative_clustering(sys.argv[1], int(sys.argv[2]))
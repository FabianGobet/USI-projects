from sklearn.cluster import KMeans
import pandas as pd
import sys

def k_means_clustering(n_clusters: int, path_to_featurevec_csv: str = None, save: bool = False, get:bool = False) -> None:
    df = pd.read_csv(path_to_featurevec_csv)

    X = df.drop('method_name', axis=1).values
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(X)

    df_for_csv = df['method_name'].to_frame()
    df_for_csv['cluster_id'] = kmeans  
    cols = df_for_csv.columns.tolist()
    df_for_csv = df_for_csv[cols[-1:] + cols[:-1]]
    
    class_name = path_to_featurevec_csv.split('/')[-1].split('.')[0]
    if save:
       df_for_csv.sort_values(by='cluster_id').to_csv(class_name + '_kmeans_'+str(n_clusters)+'.csv', index=False)
    if get:
        return df_for_csv


if __name__ == '__main__':
    if not len(sys.argv) > 2:
        print('Usage: python k_means.py <path_to_featurevec_csv> <n_clusters>')
        sys.exit(1)
    path_to_directory = sys.argv[1]
    k_means_clustering(path_to_featurevec_csv=int(sys.argv[2]), n_clusters=sys.argv[1],save=True)
import numpy as np
import importlib.util
import os
import contextlib
import numpy as np
import sys
from sklearn.manifold import TSNE
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

spec = importlib.util.spec_from_file_location("search_data", "search-data.py")
search_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(search_data)
get_sims = search_data.get_sims

class Metrics:
    """
    A class to calculate and store precision and recall metrics for different models.

    Attributes
    ----------
    measures : dict
        Dictionary containing SubMetrics instances for each model ('fv', 'tfidf', 'lsi', 'd2v').

    Methods
    -------
    add_metric(list_metrics_across_models)
        Adds precision and recall metrics for each model from a list of metrics.
    get_avg_precision_recall()
        Returns the average precision and recall for each model.
    """
    class SubMetrics:
        """
        A helper class to store precision and recall metrics for a specific model.

        Attributes
        ----------
        metrics : dict
            Dictionary with keys 'precision' and 'recall', each storing a list of respective metric values.

        Methods
        -------
        add_metric(metric)
            Adds a precision and recall metric to the respective lists.
        """
        def __init__(self):
            self.metrics = {'precision': [], 'recall': []}
            
        def add_metric(self, metric):
            """
            Adds a precision and recall metric to the lists.

            Parameters
            ----------
            metric : tuple
                Tuple containing precision and recall values.
            """
            self.metrics['precision'].append(metric[0])
            self.metrics['recall'].append(metric[1])
    
    def __init__(self):
        self.measures = {
            'fv': Metrics.SubMetrics(), 
            'tfidf': Metrics.SubMetrics(), 
            'lsi': Metrics.SubMetrics(), 
            'd2v': Metrics.SubMetrics()
        }
        
    def add_metric(self, list_metrics_across_models):
        """
        Adds precision and recall metrics across multiple models.

        Parameters
        ----------
        list_metrics_across_models : list of tuple
            List containing tuples of precision and recall metrics for each model.
        """
        for m,(p,r) in zip(self.measures.keys(), list_metrics_across_models):
            self.measures[m].add_metric((p,r))
    
    def get_avg_precision_recall(self):
        """
        Computes the average precision and recall for each model.

        Returns
        -------
        dict
            Dictionary with model names as keys and tuples of (average precision, average recall) as values.
        """
        avg = {}
        for m in self.measures.keys():
            avg[m] = (np.mean(self.measures[m].metrics['precision']), np.mean(self.measures[m].metrics['recall']))
        return avg
        

def get_data_from_file(file_path):
    """
    Load data from a file where each entry is separated by a newline.

    Parameters
    ----------
    file_path : str
        Path to the input file.

    Returns
    -------
    list of list of str
        List containing entries, where each entry is a list of strings.
    """
    data = []
    with open(file_path) as f:
        entry = []
        for line in f:
            if line == '\n':
                data.append(entry)
                entry = []
            else:
                entry.append(line.strip())
        if entry:
            data.append(entry)
    return data

def get_metrics(data, top_n=5):
    """
    Calculate precision and recall metrics for each query.

    Parameters
    ----------
    data : list of list of str
        List of entries, where each entry contains query, name, and file path.
    top_n : int, optional
        Number of top results to consider; defaults to 5.

    Returns
    -------
    Metrics
        An instance of the Metrics class containing precision and recall for each model.
    """
    metrics = Metrics()
    emb_vecs = {'lsi': [], 'd2v': []}
    for query, name, file in data: 
        temp_metric = []
        fv_results, tfidf_results, lsi_results, d2v_results, lsi_emb_vecs, d2v_emb_vectors = get_sims(query, top_n)
        emb_vecs['lsi'].append(lsi_emb_vecs)
        emb_vecs['d2v'].append(d2v_emb_vectors)
        for r in [fv_results, tfidf_results, lsi_results, d2v_results]:
            submetric = (0,0)
            for i,(eval_name, eval_file) in enumerate(zip(r['name'].values.tolist(), r['file'].values.tolist()), start=1):
                if eval_name == name and eval_file == file:
                    submetric = (i/5, 1)
                    break
            temp_metric.append(submetric)
        metrics.add_metric(temp_metric)
    return metrics, emb_vecs

def plot_tsne(algo_emb_vecs, title):
    """
    Plot a t-SNE visualization of embedding vectors for multiple queries.

    This function uses t-SNE to reduce the dimensionality of the embedding vectors for visualization. Each query's embeddings
    are plotted in a 2D scatter plot, with different colors representing different queries.

    Parameters
    ----------
    algo_emb_vecs : list of list of numpy.ndarray
        A list where each element is a list of embedding vectors for a specific query.
    title : str
        Title of the plot, which is also used as the filename for saving the plot.

    Returns
    -------
    None
        Saves the t-SNE plot as a PNG file with the specified title and displays it.
    """
    merged = []
    labels = []
    for idx, s in enumerate(algo_emb_vecs):
        merged.extend(s)
        labels.extend([f'query {idx+1}'] * len(s))

    merged = np.vstack(merged)
    #print(merged.shape)
    
    tsne = TSNE(n_components=2, perplexity=4, n_iter=3000)
    tsne_results = tsne.fit_transform(merged)
    
    df = pd.DataFrame(tsne_results, columns=['x', 'y'])
    df['label'] = labels
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='x', y='y', hue='label', palette=sns.color_palette("hsv", len(set(labels))), data=df, legend='full')

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()
    plt.title(title+'t-SNE visualization')
    plt.savefig(title.strip()+'_tsne.png', bbox_inches='tight')
    #plt.show()
    
    
if __name__ == '__main__':
    
    n=5
    file_path='ground-truth-unique.txt'
    
    if len(sys.argv)==2:
        file_path = sys.argv[1]
    elif len(sys.argv)==3:
        n = int(sys.argv[2])
    elif len(sys.argv)>3:
        print("Usage: python prec-recall.py [ground-truth.txt] [top_n]")
        print("Example: python prec-recall.py ./ground-truth-unique.txt 5")
        sys.exit()
    
    data = get_data_from_file(file_path)
    with contextlib.redirect_stdout(open(os.devnull, 'w')):
        metrics, emb_vecs = get_metrics(data, n)
        
    metrics = metrics.get_avg_precision_recall()
    labels = {'fv': 'FREQ', 'tfidf': 'TF-IDF', 'lsi': 'LSI', 'd2v': 'Doc2Vec'}
    with open('precision_recall.txt', 'w') as f:
        for k,v in labels.items():
            s = f'{v}:'
            s = f'{s}\nAverage precision = {metrics[k][0]}'
            s = f'{s}\nRecall = {metrics[k][1]}\n'
            f.write(s+'\n')
            print(s)
            
    plot_tsne(emb_vecs['lsi'], 'LSI ')
    plot_tsne(emb_vecs['d2v'], 'Doc2Vec ')
        
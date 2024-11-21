import pandas as pd
import re
import pickle
import numpy as np
import sys
import os
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import gensim as gs

EXTRA_STOP_WORDS = ['value', "'", 'of', 'graph', 'shape', 'call', 'input', 'tests', 'size', 'name', 'type', '>>>', 'this', 'output', 'test', 'to', 'self', 'returns', 'main', 'get', 'tf', 'args', 'the', 'if', '', 'def', 'variable', 'config', 'compute']
STOP_WORDS = set(stopwords.words('english')).union(EXTRA_STOP_WORDS)
AFTER_FILTERS = ['Args:', 'Returns:', 'Returns', 'Args', 'Also see', 'Also see:', 'Example usage:', '>>>']
PS = PorterStemmer()
SAVE_PATH = './generated_files'
CHECK_FILES = ['/bag_of_words', '/corpus_bow_fv', '/tfidf', '/tfidf_corpus', '/lsi', '/lsi_corpus', '/d2v', '/corpus_processed']

def split_camel_case_underscore_clean(sample):
    """
    Split camel case and underscore-separated words, remove digits and special characters.

    Parameters
    ----------
    sample : str
        The input string to be processed.

    Returns
    -------
    str
        Processed string with spaces between camel case words and cleaned of digits and special characters.
    """
    sample1 = re.sub('([a-z])([A-Z])', r'\1 \2', sample)
    sample1 = re.sub('_', ' ', sample1)
    sample1 = re.sub(r'\d+', '', sample1)
    sample1 = re.sub(r'[^\w\s]',' ',sample1)
    return sample1

def remove_everything_after(sentence, clauses):
    """
    Truncate a sentence at the first occurrence of any clause from a list.

    Parameters
    ----------
    sentence : str
        The sentence to be truncated.
    clauses : list of str
        List of clauses to serve as truncation points.

    Returns
    -------
    str
        The truncated sentence, with content after the first matched clause removed.
    """
    for clause in clauses:
        if clause in sentence:
            sentence = sentence.split(clause)[0]
    return sentence

def create_entity(name,file,comment):
    """
    Generate a list of processed, stemmed words from the name, file, and comment fields, excluding stop words.

    Parameters
    ----------
    name : str
        The name to be processed, typically representing an entity's identifier.
    file : str
        The file path to extract filename-related words.
    comment : str or None
        Comment to extract descriptive words, if available.

    Returns
    -------
    list of str
        List of stemmed and filtered words representing the entity, based on its name, file, and comment.
    """
    name1 = split_camel_case_underscore_clean(name).split(' ')
    file1 = split_camel_case_underscore_clean(file.split('/')[-1].strip('.py')).split(' ')

    pre_final=''
    if comment is not None:
        pre_final=comment
        pre_final = remove_everything_after(pre_final, AFTER_FILTERS)
        pre_final = split_camel_case_underscore_clean(pre_final)
        
    pre_final = pre_final.split(' ')

    final = []
    for pre_w in pre_final+name1+file1:
        w = PS.stem(pre_w.lower())
        if w not in STOP_WORDS:
            final.append(w)

    return final

def get_corpus_processed(df, save_path=SAVE_PATH):
    """
    Generate a processed corpus from a DataFrame and save it to a file if a path is provided.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing data for corpus generation, with columns 'name', 'file', and 'comment'.
    save_path : str, optional
        Path to save the processed corpus; defaults to SAVE_PATH.

    Returns
    -------
    list of list of str
        List of processed, filtered, and stemmed tokens for each row in the DataFrame.
    """
    corpus_processed = []
    for i in range(len(df)):
        row = df.iloc[i]
        comment = row['comment']
        corpus_processed.append(create_entity(row['name'], row['file'], comment if isinstance(comment, str) else ''))
    if save_path is not None:
        with open(save_path+"/corpus_processed", 'wb') as f:
            pickle.dump(corpus_processed, f)
    return corpus_processed

'''
def get_corpus_lower_stemmed(df, save_path=SAVE_PATH):
    """
    Generate a corpus with lowercased and stemmed comments from a DataFrame and save it if a path is provided.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing comments to be processed and stemmed.
    save_path : str, optional
        Path to save the lowercased and stemmed corpus; defaults to SAVE_PATH.

    Returns
    -------
    list of str
        List of lowercased and stemmed comment strings.
    """
    corpus_lower_stemmed = []
    for i in range(len(df)):
        row = df.iloc[i]
        comment = ''
        if isinstance(row['comment'], str):
            comment = ' '.join([PS.stem(w.lower()) for w in row['comment'].split(' ')])
        corpus_lower_stemmed.append(comment)
    if save_path is not None:
        with open(save_path+"/corpus_lower_stemmed", 'wb') as f:
            pickle.dump(corpus_lower_stemmed, f)
    return corpus_lower_stemmed
'''

def query_pipeline(query):
    """
    Process a query string by splitting, lowercasing, stemming, and filtering out stop words.

    Parameters
    ----------
    query : str
        Query string to be processed.

    Returns
    -------
    list of str
        List of stemmed and filtered tokens from the query.
    """
    query1 = split_camel_case_underscore_clean(query)
    query1 = query1.split(' ')
    final = []
    for pre_w in query1:
        w = PS.stem(pre_w.lower())
        if w not in STOP_WORDS:
            final.append(w)
    return final

def get_bag_of_words(processed_sentences, save_path=SAVE_PATH):
    """
    Create a bag-of-words model from a list of tokenized sentences and save it if a path is provided.

    Parameters
    ----------
    processed_sentences : list of list of str
        List of tokenized and processed sentences.
    save_path : str, optional
        Path to save the bag-of-words model; defaults to SAVE_PATH.

    Returns
    -------
    gensim.corpora.Dictionary
        Bag-of-words dictionary model based on the processed sentences.
    """
    bag_of_words = gs.corpora.Dictionary(processed_sentences)
    if save_path is not None:
        with open(save_path+"/bag_of_words", 'wb') as f:
            pickle.dump(bag_of_words, f)
    return bag_of_words

def get_corpus_fv(corpus_processed, bag_of_words, save_path=SAVE_PATH):
    """
    Generate a frequency vector (bag-of-words) representation of the corpus and save it if a path is provided.

    Parameters
    ----------
    corpus : list of list of str
        Corpus of tokenized sentences.
    bag_of_words : gensim.corpora.Dictionary
        Bag-of-words dictionary model.
    save_path : str, optional
        Path to save the frequency vector corpus; defaults to SAVE_PATH.

    Returns
    -------
    list of list of tuple
        Corpus in frequency vector (bag-of-words) format.
    """
    corpus_bow_fv = [bag_of_words.doc2bow(sentence) for sentence in corpus_processed]
    if save_path is not None:
        with open(save_path+"/corpus_bow_fv", 'wb') as f:
            pickle.dump(corpus_bow_fv, f)
    return corpus_bow_fv

def evaluate_query_fv(corpus_bow_fv, query_bow, bag_of_words):
    """
    Compute similarity scores for a query using a frequency vector representation of the corpus.

    Parameters
    ----------
    corpus_bow_fv : list of list of tuple
        Frequency vector representation of the corpus.
    query_bow : list of tuple
        Query in frequency vector format.
    bag_of_words : gensim.corpora.Dictionary
        Bag-of-words dictionary model.

    Returns
    -------
    numpy.ndarray
        Array of similarity scores for the query against the corpus.
    """
    index = gs.similarities.SparseMatrixSimilarity(corpus_bow_fv, num_features=len(bag_of_words))
    sims = index[query_bow]
    return sims
    
def get_tfidf_model(corpus_bow_fv, save_path=SAVE_PATH):
    """
    Generate a TF-IDF model and transform the corpus, saving both if a path is provided.

    Parameters
    ----------
    corpus_bow_fv : list of list of tuple
        Frequency vector representation of the corpus.
    save_path : str, optional
        Path to save the TF-IDF model and its transformed corpus; defaults to SAVE_PATH.

    Returns
    -------
    tuple
        A tuple containing the TF-IDF model and the transformed corpus in TF-IDF format.
    """
    tfidf = gs.models.TfidfModel(corpus_bow_fv)
    tfidf_corpus = tfidf[corpus_bow_fv]
    if save_path is not None:
        with open(save_path+"/tfidf", 'wb') as f:
            pickle.dump(tfidf, f)
        with open(save_path+"/tfidf_corpus", 'wb') as f:
            pickle.dump(tfidf_corpus, f)
    return tfidf, tfidf_corpus

def evaluate_query_tfidf(tfidf, query_bow, tfidf_corpus, num_features):
    """
    Compute similarity scores for a query using a TF-IDF model representation of the corpus.

    Parameters
    ----------
    tfidf : gensim.models.TfidfModel
        TF-IDF model for the corpus.
    query_bow : list of tuple
        Query in frequency vector format.
    tfidf_corpus : list
        Transformed corpus in TF-IDF format.
    num_features : int
        Number of features in the TF-IDF model.

    Returns
    -------
    numpy.ndarray
        Array of similarity scores for the query against the TF-IDF transformed corpus.
    """
    index = gs.similarities.SparseMatrixSimilarity(tfidf_corpus, num_features=num_features)
    return index[tfidf[query_bow]]

def get_lsi_model(tfidf_corpus, bag_of_words, save_path=SAVE_PATH, num_topics=300):
    """
    Generate an LSI (Latent Semantic Indexing) model from a TF-IDF transformed corpus and save it if a path is provided.

    Parameters
    ----------
    tfidf_corpus : list
        Transformed corpus in TF-IDF format.
    bag_of_words : gensim.corpora.Dictionary
        Bag-of-words dictionary model.
    save_path : str, optional
        Path to save the LSI model and its transformed corpus; defaults to SAVE_PATH.
    num_topics : int, optional
        Number of topics for the LSI model; defaults to 300.

    Returns
    -------
    tuple
        A tuple containing the LSI model and the transformed corpus in LSI format.
    """
    lsi = gs.models.LsiModel(tfidf_corpus, id2word=bag_of_words, num_topics=num_topics)
    lsi_corpus = lsi[tfidf_corpus]
    if save_path is not None:
        with open(save_path+"/lsi", 'wb') as f:
            pickle.dump(lsi, f)
        with open(save_path+"/lsi_corpus", 'wb') as f:
            pickle.dump(lsi_corpus, f)
    return lsi, lsi_corpus

def evaluate_query_lsi(lsi, lsi_corpus, tfidf, query_bow):
    """
    Compute similarity scores for a query using an LSI model.

    Parameters
    ----------
    lsi : gensim.models.LsiModel
        LSI model for the corpus.
    lsi_corpus : list
        Transformed corpus in LSI format.
    tfidf : gensim.models.TfidfModel
        TF-IDF model used to transform the query prior to LSI.
    query_bow : list of tuple
        Query in frequency vector format.

    Returns
    -------
    numpy.ndarray
        Array of absolute similarity scores for the query against the LSI-transformed corpus.
    """
    index = gs.similarities.MatrixSimilarity(lsi_corpus)
    return abs(index[lsi[tfidf[query_bow]]])

def get_d2v_model(corpus_processed, save_path=SAVE_PATH, vector_size=300, num_epochs=40, min_count=2):
    """
    Generate a Doc2Vec model from a processed corpus and save it if a path is provided.

    Parameters
    ----------
    corpus_processed : list of list of str
        List of tokenized and processed sentences, where each sentence is a list of words.
    save_path : str, optional
        Path to save the trained Doc2Vec model; defaults to SAVE_PATH.
    vector_size : int, optional
        Dimensionality of the feature vectors; defaults to 300.
    num_epochs : int, optional
        Number of training epochs; defaults to 40.
    min_count : int, optional
        Ignores all words with total frequency lower than this; defaults to 2.

    Returns
    -------
    gensim.models.Doc2Vec
        Trained Doc2Vec model.
    """
    dv_docs = [gs.models.doc2vec.TaggedDocument(doc, [i]) for i, doc in enumerate(corpus_processed)]
    d2v = gs.models.Doc2Vec(vector_size=vector_size, min_count=min_count, epochs=num_epochs)
    d2v.build_vocab(dv_docs)
    d2v.train(dv_docs, total_examples=d2v.corpus_count, epochs=d2v.epochs)
    if save_path is not None:
        with open(save_path+"/d2v", 'wb') as f:
            pickle.dump(d2v, f)
    return d2v

def evaluate_query_d2v(d2v, query_processed, df, topn=5):
    """
    Retrieve the top similar documents for a query_processed using a Doc2Vec model.

    Parameters
    ----------
    d2v : gensim.models.Doc2Vec
        Trained Doc2Vec model.
    query_processed : str
        Tokenized query
    df : pandas.DataFrame
        DataFrame containing document data.
    topn : int, optional
        Number of top results to retrieve; defaults to 5.

    Returns
    -------
    pandas.DataFrame
        DataFrame of top similar documents based on the query.
    """
    tv = d2v.infer_vector(query_processed)
    d2v_results = []
    d2v_emb_vectors = []
    for r,e in d2v.dv.most_similar([tv], topn=topn):
        d2v_results.append(r)
        d2v_emb_vectors.append(e)
    return df.iloc[d2v_results]

def load_models(dir_path=SAVE_PATH):
    """
    Load pre-trained models and data files from the specified directory.

    Parameters
    ----------
    dir_path : str, optional
        Directory path where the models and data files are stored; defaults to SAVE_PATH.

    Returns
    -------
    tuple
        Tuple containing loaded DataFrame and models:
        (df, bag_of_words, corpus_bow_fv, tfidf, tfidf_corpus, lsi, lsi_corpus, d2v).
    """
    df = pd.read_csv(dir_path+'/data.csv')
    bag_of_words = pickle.load(open(dir_path+'/bag_of_words', 'rb'))
    corpus_bow_fv = pickle.load(open(dir_path+'/corpus_bow_fv', 'rb'))
    tfidf = pickle.load(open(dir_path+'/tfidf', 'rb'))
    lsi = pickle.load(open(dir_path+'/lsi', 'rb'))
    lsi_corpus = pickle.load(open(dir_path+'/lsi_corpus', 'rb'))
    tfidf_corpus = pickle.load(open(dir_path+'/tfidf_corpus', 'rb'))
    d2v = pickle.load(open(dir_path+'/d2v', 'rb'))
    corpus_processed = pickle.load(open(dir_path+'/corpus_processed', 'rb'))
    
    return df, bag_of_words, corpus_bow_fv, tfidf, tfidf_corpus, lsi, lsi_corpus, d2v, corpus_processed

def pick_top_n(df, sims, n=5):
    """
    Retrieve the top N results from a DataFrame based on similarity scores.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing document data.
    sims : numpy.ndarray
        Array of similarity scores.
    n : int, optional
        Number of top results to retrieve; defaults to 5.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the top N results sorted by similarity.
    """
    ordered_indexes = np.argsort(sims, axis=0)[::-1][:n]
    return df.iloc[ordered_indexes], ordered_indexes
        
def print_results(results, algo):
    """
    Print the results with file location and line number using the specified algorithm.

    Parameters
    ----------
    results : pandas.DataFrame
        DataFrame of search results.
    algo : str
        Name of the algorithm used for generating the results.

    Returns
    -------
    None
    """
    print(f"Using {algo}:")
    for i in range(len(results)):
        print(f"{i+1}. {results.iloc[i]['name']} in {results.iloc[i]['file']} at line {results.iloc[i]['line']}")
    print()

def check_files_exist():
    """
    Verify the existence of required model files and directories. If files are missing, generate and save models; otherwise, load existing models.

    This function checks if the specified models and data files are present in the SAVE_PATH directory. If any required files are missing,
    it creates the necessary files and models. If all files are present, it loads the models from disk.

    Returns
    -------
    tuple
        A tuple containing loaded DataFrame and models:
        (df, bag_of_words, corpus_bow_fv, tfidf, tfidf_corpus, lsi, lsi_corpus, d2v).

    Notes
    -----
    If 'data.csv' is missing, the function will terminate with an error message.
    """
    make_models = False
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
        make_models = True
    else:
        for file in CHECK_FILES:
            if not os.path.exists(SAVE_PATH+file):
                make_models = True
                break
            
    if make_models:
        if not os.path.exists(SAVE_PATH+'/data.csv'):
            print('data.csv file not found in ./generated_files')
            sys.exit(1)
        else:
            print('Creating models...\n')
            df = pd.read_csv(SAVE_PATH+'/data.csv')
            corpus_processed = get_corpus_processed(df, SAVE_PATH)
            bag_of_words = get_bag_of_words(corpus_processed, SAVE_PATH)
            corpus_bow_fv = get_corpus_fv(corpus_processed, bag_of_words, SAVE_PATH)
            tfidf, tfidf_corpus = get_tfidf_model(corpus_bow_fv, SAVE_PATH)
            lsi, lsi_corpus = get_lsi_model(tfidf_corpus, bag_of_words, SAVE_PATH)
            d2v = get_d2v_model(corpus_processed, SAVE_PATH)
    else:
        print('Loading models...\n')
        df, bag_of_words, corpus_bow_fv, tfidf, tfidf_corpus, lsi, lsi_corpus, d2v, corpus_processed = load_models(SAVE_PATH)
    
    return df, bag_of_words, corpus_bow_fv, tfidf, tfidf_corpus, lsi, lsi_corpus, d2v, corpus_processed

def get_sims(query, n=5):
    """
    Retrieve the top similar documents for a query using multiple models (frequency vector, TF-IDF, LSI, and Doc2Vec).

    This function loads or creates necessary models and data, processes the query, and evaluates similarity scores using four methods:
    frequency vector (FV), TF-IDF, LSI, and Doc2Vec. It then returns the top N results from each method.

    Parameters
    ----------
    query : str
        Query string to evaluate.
    n : int, optional
        Number of top results to retrieve; defaults to 5.

    Returns
    -------
    tuple
        A tuple containing DataFrames of the top N similar documents for each method:
        (fv_results, tfidf_results, lsi_results, d2v_results).
    """
    df, bag_of_words, corpus_bow_fv, tfidf, tfidf_corpus, lsi, lsi_corpus, d2v, corpus_processed = check_files_exist()
    query_processed = query_pipeline(query)
    query_bow = bag_of_words.doc2bow(query_processed)
    
    sims = evaluate_query_fv(corpus_bow_fv, query_bow, bag_of_words)
    fv_results, _ = pick_top_n(df, sims, n)
    
    sims = evaluate_query_tfidf(tfidf, query_bow, tfidf_corpus, len(bag_of_words))
    tfidf_results, _ = pick_top_n(df, sims, n)
    
    sims = evaluate_query_lsi(lsi, lsi_corpus, tfidf, query_bow)
    lsi_results, lsi_ordered_indexes = pick_top_n(df, sims, n)
    
    lsi_emb_vecs = []
    for idx in lsi_ordered_indexes:
        vec = []
        for t in lsi_corpus[idx]:
            vec.append(t[1])
        lsi_emb_vecs.append(vec)
   
    d2v_results = evaluate_query_d2v(d2v, query_processed, df, n)
    d2v_emb_vectors = [d2v.infer_vector(corpus_processed[i]) for i in d2v_results.index]
    
    return fv_results, tfidf_results, lsi_results, d2v_results, lsi_emb_vecs, d2v_emb_vectors



if __name__ == '__main__':
    n=5
    query=sys.argv[1]
    
    if len(sys.argv)==3:
        n = int(sys.argv[2])
    elif len(sys.argv)!=2:
        print("Usage: python search-data.py query [top_n]")
        print("Example: python search-data.py 'this function returns the sum of two numbers' 5")
        sys.exit()
        
    
    fv_results, tfidf_results, lsi_results, d2v_results, _, _ = get_sims(query, n)
    
    print(f"Results for query:\n'{query}'\n")
    print_results(fv_results,'Frequency Vectors')
    print_results(tfidf_results, 'TF-IDF')
    print_results(lsi_results, 'LSI')
    print_results(d2v_results, 'Doc2Vec')
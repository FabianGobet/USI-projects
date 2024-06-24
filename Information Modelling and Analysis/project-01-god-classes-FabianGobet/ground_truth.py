import pandas as pd
import sys

def get_ground_truth(path_featurevec_csv: str, path_keywords_list:str, save_path = './') -> None:

    with open(path_keywords_list, 'r') as f:
        keywords_list = f.read().splitlines()
    df = pd.read_csv(path_featurevec_csv)

    ground_truths = {}
    for method in df['method_name'].values:
        for keyword in keywords_list:
            if keyword in method.lower():
                if keyword not in ground_truths:
                    ground_truths[keyword] = []
                ground_truths[keyword].append(method)
                break
        else:
            if 'none' not in ground_truths:
                ground_truths['none'] = []
            ground_truths['none'].append(method)

    df_to_csv = df['method_name'].to_frame()
    df_to_csv['cluster_id'] = -1
    cols = df_to_csv.columns.tolist()
    df_to_csv = df_to_csv[cols[-1:] + cols[:-1]]
    for i, keyword in enumerate(ground_truths):
        for method in ground_truths[keyword]:
            df_to_csv.loc[df_to_csv['method_name'] == method, 'cluster_id'] = i

    df_to_csv.sort_values(by='cluster_id', inplace=True)
    file_name = path_featurevec_csv.split('/')[-1].split('.')[0]
    file_name = save_path+'ground_truth_'+file_name+'.csv'
    df_to_csv.to_csv(file_name, index=False)


if __name__ == '__main__':
    if len(sys.argv) == 3:
        get_ground_truth(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 4:
        get_ground_truth(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        print('Usage: python ground_truth.py <path_to_featurevec_csv> <path_to_keywords_list> [save_path]')
        sys.exit(1)
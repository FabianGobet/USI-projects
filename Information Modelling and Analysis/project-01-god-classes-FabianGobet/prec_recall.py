import pandas as pd
import sys

def get_intrapairs(df: pd.DataFrame) -> list:
    intrapairs = []
    for _, group in df.groupby('cluster_id'):
        if len(group) > 1:
            for i in range(len(group)):
                for j in range(i+1, len(group)):
                    temp_set = set()
                    temp_set.update([group.iloc[i]['method_name'], group.iloc[j]['method_name']])
                    if temp_set not in intrapairs:
                        intrapairs.append(temp_set)
    return intrapairs

def get_intersections(intra1:list, intra2:list ) -> list:
    intersections = []
    for p1 in intra1:
        if p1 in intra2:
            intersections.append(p1)
    return intersections

def get_precision_recall(path_cluster_csv: str, path_ground_truth: str) -> tuple:
    df_d = pd.read_csv(path_cluster_csv)
    df_g = pd.read_csv(path_ground_truth)

    intra_d = get_intrapairs(df_d)
    intra_g = get_intrapairs(df_g)
    inter = get_intersections(intra_d, intra_g)

    p = len(inter) / len(intra_d)
    r = len(inter) / len(intra_g)

    return p,r

def print_utility(p:float,r:float) -> None:
    print(f'p = {p:.4f}')
    print(f'r = {r:.4f}')


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python prec_recall.py <path_to_cluster_csv> <path_to_ground_truth_csv>')
        sys.exit(1)
    else:
        p,r = get_precision_recall(sys.argv[1], sys.argv[2])
        print_utility(p,r)
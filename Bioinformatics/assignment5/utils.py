from archive import Archive, Template
import requests
import os
from typing import List, Tuple
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image, UnidentifiedImageError
import re
import itertools
import numpy as np

def first_pass(hhpred_metrics_path: str, archive: Archive) -> Archive:
    with open(hhpred_metrics_path, 'r') as f:
        lines = f.readlines()
    
    for k,line in enumerate(lines):
        if k == 0:
            continue
        line = line.strip()
        line = line.lower()
        line = re.split(r'\s+', line)
        line[-1] = int(re.sub(r'[()]', '', line[-1]))
        line[-3] = f'"{[int(j) for j in line[-3].split("-")]}"'
        for i in [-4,0]:
            line[i] = int(line[i])
            
        for i in range(-9, -4, 1):
            line[i] = float(line[i]) 
            
        name = ''
        line = line[:2] + [name] + line[-9:]
        line = line + [round(100*line[-4]/archive.get_query_length(), 3)] + 4*[None]
        archive.add_template(Template(line, archive.get_args()))
    return archive


def second_pass(hhpred_alignments_path: str, archive: Archive):
    templates_no_attr = archive.get_attribute('No')
    accounted = []
    
    if len(templates_no_attr) != 0:
        with open(hhpred_alignments_path, 'r') as f:
            line = f.readline()
            while line:
                if line.startswith('No'):
                    no_attr = int(line.split()[1])
                    accounted.append(no_attr)
                    true_idx = templates_no_attr.index(no_attr)
                    line = f.readline().strip()
                    if line[0] == '>':
                        line = line.removeprefix('>')
                        if line[-1] == '}':
                            line = line.removesuffix('}')
                            name, organism = line.split('{')
                        else:
                            name = line
                            organism = f'"unknown"'
                        hit = name[:7].lower().strip()
                        name = name[7:].lower()
                        organism = organism.lower()
                        archive.templates[true_idx].dict['Name'] = f'"{name}"'
                        archive.templates[true_idx].dict['Organism'] = organism
                        archive.templates[true_idx].dict['Hit'] = hit
                        identities, similarity = f.readline().strip().split()[4:6]
                        identities = float(identities.removesuffix('%').split('=')[1])
                        similarity = float(similarity.split('=')[1])
                        archive.templates[true_idx].dict['Identities'] = identities
                        archive.templates[true_idx].dict['Similarity'] = similarity
                line = f.readline()
    else:
        print('No templates with that attribute.')
    return archive, templates_no_attr, accounted


def verify_files_correspondance(templates_no_attr, accounted):
    if len(accounted) != len(templates_no_attr):
        print('Not all templates accounted for.')
        return False
    else:
        print('All templates accounted for.')
        return True


def check_image_exists(image_dir_path: str, hit: str) -> bool:
    image_path = os.path.join(image_dir_path, f'{hit}.jpg')
    return os.path.isfile(image_path)

def get_img_link_from_hit(hit: str) -> str:
    return f'https://cdn.rcsb.org/images/structures/{hit}_assembly-1.jpeg'

def get_image(image_dir_path: str, hit: str) -> None:
    try:
        img_data = requests.get(get_img_link_from_hit(hit)).content
        image_path = os.path.join(image_dir_path, f'{hit}.jpg')
        with open(image_path, 'wb') as handler:
            handler.write(img_data)
        # Verify the image
        with Image.open(image_path) as img:
            img.verify()
    except (requests.RequestException, UnidentifiedImageError) as e:
        print(f"Failed to download or verify image for hit {hit}: {e}")
        if os.path.exists(image_path):
            os.remove(image_path)

def see_all_hits_3d(hit_list: List[str], image_dir_path: str) -> None:
    for hit in hit_list:
        if not check_image_exists(image_dir_path, hit):
            get_image(image_dir_path, hit)
    
    # Plot all images with a maximum of 3 images per row
    num_images = len(hit_list)
    num_cols = 3
    num_rows = (num_images + num_cols - 1) // num_cols  # Calculate the number of rows needed
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 5 * num_rows))
    axes = axes.flatten()  # Flatten the axes array for easy iteration
    
    counter = 0
    for ax, hit in zip(axes, hit_list):
        image_path = os.path.join(image_dir_path, f'{hit}.jpg')
        try:
            img = mpimg.imread(image_path)
            ax.imshow(img)
            ax.set_title(f'{counter}: {hit}')
            ax.axis('off')
            counter += 1
        except UnidentifiedImageError:
            print(f"Cannot identify image file {image_path}")
            ax.set_title(f"{hit} (Error)")
            ax.axis('off')
    
    # Hide any unused subplots
    for ax in axes[num_images:]:
        ax.axis('off')
    
    plt.show()
    
    
def read_csv(csv_path:str) -> List[List[str]]:
    entries = []
    with  open(csv_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        entries.append(line.strip().split(','))
    return entries

def get_hits_from_csv(csv_path:str) -> List[str]:
    entries = read_csv(csv_path)
    idx = entries[0].index('Hit')
    hits = []
    for entry in entries[1:]:
        hits.append(entry[idx])
    return hits


def union_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not intervals:
        return []

    intervals.sort(key=lambda x: x[0])

    merged_intervals = []
    current_start, current_end = intervals[0]

    for start, end in intervals[1:]:
        if start <= current_end: 
            current_end = max(current_end, end)
        else:  
            merged_intervals.append((current_start, current_end))
            current_start, current_end = start, end

    merged_intervals.append((current_start, current_end))

    return merged_intervals

def generate_combinations(objects, min_size, max_size):
    combos = []
    for i in range(min_size, max_size+1):
        for c in itertools.combinations(objects, i):
            combos.append([k for k in c])
    return combos

def freq_score_avg_averaged_by_hits(archive, hits, average_by_hits = True):
    frequencies = archive.get_frequency_coverage_for_hits(hits)
    val = archive.get_average_frequency(frequencies)
    if average_by_hits:
        val = val / len(hits)
    return val

def score_by_params(archive, hits, params, scalings, average_by_hits = True, average_by_num_params = True):
    templates = archive.get_templates_with_trimmed_hits(hits)
    results = []
    for h in hits:
        vec = [templates[h].get_attribute(p) for p in params]
        results.append((np.array(vec) @ np.array(scalings))/(len(params) if average_by_num_params else 1))
    if average_by_hits:
        return sum(results) / len(hits)
    else:
        return sum(results)
    
def avg_log_weighted_freq_score(archive, hits, min_num_templates_scale_to_1 = 3):
    frequencies = archive.get_frequency_coverage_for_hits(hits)
    log_fn = lambda x: np.log10(x+1)/np.log10(min_num_templates_scale_to_1+1)
    toned_freqs = [log_fn(x) for x in frequencies]
    return sum(toned_freqs)/len(toned_freqs)

def generate_pdb_link(hit:str) -> str:
    if '_' in hit:
        hit = hit.split('_')[0]
    return f'https://www.rcsb.org/structure/{hit}'

def get_or_save_pdb_links_from_hit_list(hit_list:str, txt_save_path:str = None) -> list:
    links_list = []

    if txt_save_path is None:
        links_list = [generate_pdb_link(h) for h in hit_list]
    else:
        with open(txt_save_path, 'w') as f:
            for h in hit_list:
                l = generate_pdb_link(h)
                links_list.append(l)
                f.write(l + '\n')
    return links_list
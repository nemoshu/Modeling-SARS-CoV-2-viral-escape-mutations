from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn import preprocessing
import math
import random
import numpy as np
from math import floor
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
import time
from sklearn.neighbors import NearestNeighbors
import random

def label_encode(strains):
    """
    Preprocesses raw gene sequences by encoding into numerical form.

    Args:
        strains(list[string]): strain sequences in AA characters

    Returns:
        encoded_strains(list[list[int]]): list of encoded strains as normalized encodings
    """
    amino_acids = ['A', 'F', 'Q', 'R', 'T', 'Y', 'V', 'I', 'H', 'K', 'P', 'N', 'E', 'G', 'S', 'M', 'D', 'W', 'C', 'L', '-', 'B', 'J', 'Z', 'X']
    le = preprocessing.LabelEncoder()
    le.fit(amino_acids)

    encoded_strains = []
    for strain in strains:
        chars = list(strain)
        encoded_strains.append(le.transform(chars))

    return encoded_strains


def label_decode(encoded_strains):
    """
    Decodes encoded strains into AA characters.

    Args:
        encoded_strains(list[list[int]]): encoded strains in numerical form

    Returns:
        strains(list[list[string]]): 2D list [year] [strain] of decoded strains
    """
    amino_acids = ['A', 'F', 'Q', 'R', 'T', 'Y', 'V', 'I', 'H', 'K', 'P', 'N', 'E', 'G', 'S', 'M', 'D', 'W', 'C', 'L', '-', 'B', 'J', 'Z', 'X']
    le = preprocessing.LabelEncoder()
    le.fit(amino_acids)

    strains = []
    for one_year_encoded_strains in encoded_strains:
        one_year_strains = []
        for encoded_strain in one_year_encoded_strains:
            temp = le.inverse_transform(encoded_strain)
            one_year_strains.append(''.join(temp))
        strains.append(one_year_strains)

    return strains


def strain_cluster(strains,num_clusters=2):
    """
    Clusters similar strains using K-means clustering.

    Args:
        strains(list[string]): unencoded strains in AA characters, typically from df['Sequence']
        num_clusters(int): number of clusters

    Returns:
        result (dict[string, list]): a dictionary containing:
            'data': encoded strains using label_encode()
            'labels': cluster assignments
            'centroids': cluster centers
    """
    encoded_strains = label_encode(strains)
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(encoded_strains)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    result = {'data':encoded_strains, 'labels':labels, 'centroids':centroids}
    return result


def show_cluster(cluster,save_fig_path='none'):
    """
    Visualizes clusters.

    Args:
        cluster (dict): a dictionary containing keys 'data', 'labels' and 'centroids', typically output of strain_cluster.
        save_fig_path(string): path to save the figure. Default is 'none' which no image will be saved.

    Returns:
        None

    Output File (PCA scatter plot) : as specified in save_fig_path parameter, saved only when provided
    """
    encoded_strains = cluster['data']
    pca = PCA(n_components=2) # reduce to 2D using PCA
    reduced_data = pca.fit_transform(encoded_strains)
    fig = plt.figure()
    colors = 10 * ['r.', 'g.', 'y.', 'c.', 'm.', 'b.', 'k.'] # distinct colors for clusters
    labels = cluster['labels']
    for i in range(len(reduced_data)):
        plt.plot(reduced_data[i][0], reduced_data[i][1], colors[labels[i]], markersize=10)
    plt.show()
    if save_fig_path!='none':
        plt.savefig(save_fig_path)


def link_clusters(clusters):
    """
    Links clusters in different years using nearest neighbor analysis.

    Adds 'links' key to each cluster, which are list mappings of current cluster labels to the nearest neighbor cluster indices.

    Args:
        clusters (list[dict]): a list of clusters to link

    Returns:
        None (modifies the passed in cluster directly)
    """
    no_years = len(clusters) # each cluster represents one year
    neigh = NearestNeighbors(n_neighbors=2) # analyze the two nearest neighbors
    for year_idx in range(no_years):
        if year_idx == no_years - 1:  # last year doesn't link
            clusters[year_idx]['links'] = []
            break
        links = []
        current_centroids = clusters[year_idx]['centroids']
        next_year_centroids = clusters[year_idx + 1]['centroids']
        neigh.fit(next_year_centroids)
        idxs_by_centroid = neigh.kneighbors(current_centroids, return_distance=False)

        for label in clusters[year_idx]['labels']:
            links.append(idxs_by_centroid[label])  # centroid idx corresponds to label

        clusters[year_idx]['links'] = links


def sample_from_clusters(clusters_by_years, sample_size):
    """
    Generates time-series samples from clusters.

    Args:
        clusters_by_years(list[dict]): a list of clusters by years, typically output from link_clusters()
        sample_size (int): number of samples to generate

    Returns:
        sampled_strains(list[list]): list of sample_size time-series, each being a list of strains
    """
    sampled_strains = []
    for i in range(sample_size):
        one_sample = []
        start_idx = random.randint(0, len(clusters_by_years[0]['data'])-1)
        start_strain = clusters_by_years[0]['data'][start_idx]
        one_sample.append(start_strain)

        num_years = len(clusters_by_years)
        idx = start_idx
        # select random sample for the remaining years
        for i in range(num_years - 1):
            next_nearest_label = clusters_by_years[i]['links'][idx][0]
            candidate_idx = np.where(clusters_by_years[i + 1]['labels'] == next_nearest_label)[0]
            idx = random.choice(candidate_idx)
            one_sample.append(clusters_by_years[i + 1]['data'][idx])

        sampled_strains.append(one_sample)

    return sampled_strains


def create_dataset(strains, position, window_size=10, output_path='None'):
    """
    Constructs training data for predicting mutation escape.

    9047 is the default index for unknown trigrams.

    Args:
        strains (list[list]): a list of strains across time - n_sample*n_year, each item is a strain(str)
        position (int): the position of the strain
        window_size (int): time window, i.e., number of years to look back
        output_path (string): path to save the dataset (unused)

    Returns:
        dataset (DataFrame): a dataframe containing training data, contraining features and mutation labels

    Input File (CSV File of Dataset): data/raw/H1N1/protVec_100d_3grams.csv
    """
    # create label - 1 if AA at position differs between the last two years and 0 otherwise
    label = []
    for sample in strains:
        if sample[-1][position] == sample[-2][position]:
            label.append(0)
        else:
            label.append(1)
    # read prot embedding
    df = pd.read_csv('D:/TBSI/Masters_Thesis/NLP_papers/TEMPO/Tempo-main/data/raw/H1N1/protVec_100d_3grams.csv', sep='\t')
    triple2idx = {}
    for index, row in df.iterrows():
        triple2idx[row['words']] = index
    # create data
    start_year_idx = len(strains[0]) - window_size - 1
    data = []
    for i in range(len(strains)):
        one_sample_data = []
        for year in range(start_year_idx, len(strains[0]) - 1):
            tritri = []
            if (position == 0):
                tritri.append(9047)
                tritri.append(9047)
                if strains[i][year][position:position + 3] in triple2idx:
                    tritri.append(triple2idx[strains[i][year][position:position + 3]])
                else:
                    tritri.append(9047)
            elif (position == 1):
                tritri.append(9047)
                if strains[i][year][position - 1:position + 2] in triple2idx:
                    tritri.append(triple2idx[strains[i][year][position - 1:position + 2]])
                else:
                    tritri.append(9047)
                if strains[i][year][position:position + 3] in triple2idx:
                    tritri.append(triple2idx[strains[i][year][position:position + 3]])
                else:
                    tritri.append(9047)
            elif (position == 1271):
                if strains[i][year][position - 2:position + 1] in triple2idx:
                    tritri.append(triple2idx[strains[i][year][position - 2:position + 1]])
                else:
                    tritri.append(9047)
                if strains[i][year][position - 1:position + 2] in triple2idx:
                    tritri.append(triple2idx[strains[i][year][position - 1:position + 2]])
                else:
                    tritri.append(9047)
                tritri.append(9047)
            elif (position == 1272):
                if strains[i][year][position - 2:position + 1] in triple2idx:
                    tritri.append(triple2idx[strains[i][year][position - 2:position + 1]])
                else:
                    tritri.append(9047)
                tritri.append(9047)
                tritri.append(9047)
            else:
                if strains[i][year][position - 2:position + 1] in triple2idx:
                    tritri.append(triple2idx[strains[i][year][position - 2:position + 1]])
                else:
                    tritri.append(9047)
                if strains[i][year][position - 1:position + 2] in triple2idx:
                    tritri.append(triple2idx[strains[i][year][position - 1:position + 2]])
                else:
                    tritri.append(9047)
                if strains[i][year][position:position + 3] in triple2idx:
                    tritri.append(triple2idx[strains[i][year][position:position + 3]])
                else:
                    tritri.append(9047)

            one_sample_data.append(str(tritri))

        data.append(one_sample_data)

    dataset = pd.DataFrame(data) # convert to dataframe
    print(dataset.shape)
    print(len(label))
    dataset.insert(0, 'y', label)
    return dataset

def main():
    """
    Executes the processing pipeline.

    Returns:
        None

    Input Files (clusters): /data/zh/sprot_years/
    """
    path = '/data/zh/sprot_years/'
    files = os.listdir(path)
    files.sort()
    cluster_years = []
    start_time = time.time()

    # read cluster by year
    for file in files:
        df = pd.read_csv(path + file)
        strains = df['Sequence'].sample(1000, replace=True)
        cluster = strain_cluster(strains, num_clusters=4)
        cluster_years.append(cluster)
        print("{:.1f}: {} processed.".format(time.time() - start_time, file))

    link_clusters(cluster_years)

    # obtain and decode samples, and create DataFrame
    ss = sample_from_clusters(cluster_years, 10)

    dss = label_decode(ss)

    datasetdf = create_dataset(dss, 12)


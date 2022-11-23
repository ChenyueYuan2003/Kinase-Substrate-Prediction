import argparse
import os
import sys
import h5py
import numpy as np
import pandas as pd
import tarfile
import random
import urllib
import urllib.parse
import urllib.request


def get_run_args():
    parser = argparse.ArgumentParser()

    # samples output path
    parser.add_argument("--data_output", default='./data/samples/', type=str)

    # the num of negative samples
    parser.add_argument("--repeat", default=10, type=int)

    return parser.parse_args()


def download_embbeding(downloading_root, mp, dt=None, out_path='./', uncompress=True):
    """
    Given a metapath and a dataset it downloads the embedding from the Bioteque.
    If no dataset is provided, it dowloands all the datasets available for the metapath.
    """

    mnd = mp[:3]
    if dt is None:
        url = downloading_root.rstrip('/') + '/embeddings>%s>%s/all_datasets_embeddings.tar' % (mnd, mp)
    else:
        url = downloading_root.rstrip('/') + '/embeddings>%s>%s>%s/embeddings.tar.gz' % (mnd, mp, dt)

    # --Testing if exists
    response = urllib.request.urlopen(url)
    if response.getcode() != 200:
        sys.exit('The provided url does not exists:\n"%s"\n' % url)

    # --Creating output file system
    opath = out_path + '/%s/%s' % (mp, dt) if dt is not None else out_path + '/%s' % (mp)
    if not os.path.exists(opath):
        os.makedirs(opath)
    ofile = opath + '/%s' % url.split('/')[-1]

    # --Fetching
    urllib.request.urlretrieve(url, ofile)

    # --Uncompressing
    if uncompress is True:
        with tarfile.open(ofile) as f:
            subfiles = f.getnames()
            f.extractall(opath)
        os.remove(ofile)

        for _ in subfiles:
            if '.tar.gz' in _:
                k = opath + '/%s' % _
                with tarfile.open(k) as f:
                    f.extractall('/'.join(k.rstrip('/').split('/')[:-1]))
                os.remove(k)


# --Reading the embedding
def read_embedding(path, entity):
    # --Reads the ids of the embeddings
    with open(path + '/%s_ids.txt' % entity) as f:
        ids = f.read().splitlines()

    # --Reads the embedding vectors
    with h5py.File(path + '%s_emb.h5' % entity, 'r') as f:
        emb = f['m'][:]

    return ids, emb


def get_unique_kspairs():
    data = pd.read_csv('./data/Kinase_Substrate_Dataset.csv')
    data_human = data[(data['KIN_ORGANISM'] == 'human') & (data['SUB_ORGANISM'] == 'human')]
    data_ks = data_human[['KIN_ACC_ID', 'SUB_ACC_ID']]
    data_ks_unique = data_ks.drop_duplicates()
    data_ks_unique.to_csv('./data/ks_unique.csv', index=False)


def get_random_negative_samples(gen2emb, k_ids, s_ids, output_path, repeat):
    # generate random negative samples ids for each kinase
    for j in range(repeat):
        neg_samples_s_ids = []
        for i in range(len(k_ids)):
            neg_samples_s_ids.append(random.sample(s_ids, 1)[0])

        random_neg_samples = pd.DataFrame({'KIN_ACC_ID': k_ids, 'SUB_ACC_ID': neg_samples_s_ids})

        ran_neg_examples = random_neg_samples.values.tolist()

        # get the embedding of random negative samples
        ran_neg_emb = []
        for i in range(len(ran_neg_examples)):
            ran_neg_examples[i].append(
                gen2emb[ran_neg_examples[i][0]] if ran_neg_examples[i][0] in gen2emb.keys() else np.array([0] * 128))
            ran_neg_examples[i].append(
                gen2emb[ran_neg_examples[i][1]] if ran_neg_examples[i][1] in gen2emb.keys() else np.array([0] * 128))

            ran_neg_emb.append(ran_neg_examples[i][2:])

        ran_neg_emb = np.array(ran_neg_emb)

        ran_neg_emb.tofile(output_path + f'ran_neg_emb_{j}.csv', sep=', ', format='%f')


def main():
    # get arguments
    args = get_run_args()

    downloading_root = 'https://bioteque.irbbarcelona.org/downloads'
    # Setting paths and variables
    mp = 'GEN-_pho-GEN'
    dt = 'omnipath'

    source_entity = mp[:3]
    target_entity = mp[-3:]

    # Path to the embedding data
    data_outpath = './data/embedding_folder/'
    uncompress = True

    # --Downloading
    download_embbeding(downloading_root, mp, dt=dt, out_path=data_outpath, uncompress=uncompress)

    emb_path = data_outpath + '/%s/%s/' % (mp, dt)

    # get the id and embedding of genes
    # since source and target are the same, we can use the  function once
    ids, emb = read_embedding(emb_path, source_entity)
    # map the id to embedding
    gen2emb = {gen: emb[ix] for ix, gen in enumerate(ids)}

    # get the unique kinase-substrate pairs
    get_unique_kspairs()
    ks_ids = pd.read_csv('./data/ks_unique.csv')
    gen = ks_ids.values.tolist()

    # get the ids of kinases and substrates respectively
    k_ids = ks_ids['KIN_ACC_ID'].values.tolist()
    s_ids = ks_ids['SUB_ACC_ID'].values.tolist()

    # get the embedding of kinases-substrates pairs
    gen_emb = []
    for i in range(len(gen)):
        gen[i].append(gen2emb[gen[i][0]] if gen[i][0] in gen2emb.keys() else np.array([0] * 128))
        gen[i].append(gen2emb[gen[i][1]] if gen[i][1] in gen2emb.keys() else np.array([0] * 128))
        gen_emb.append(gen[i][2:])

    gen_emb = np.array(gen_emb)

    if not os.path.exists(args.data_output):
        os.makedirs(args.data_output)
    gen_emb.tofile(args.data_output + 'pos_emb.csv', sep=', ', format='%f')

    # get the embedding of random negative samples
    get_random_negative_samples(gen2emb, k_ids, s_ids, args.data_output, args.repeat)
    print('data preparation finished')


if __name__ == '__main__':
    main()

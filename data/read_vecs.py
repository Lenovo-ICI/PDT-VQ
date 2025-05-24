import numpy as np
import pandas as pd
import os
import faiss
import struct

def sanitize(x):
    return np.ascontiguousarray(x, dtype='float32')

def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()

def mmap_fvecs(fname):
    x = np.memmap(fname, dtype='int32', mode='r')
    d = x[0]
    return x.view('float32').reshape(-1, d + 1)[:, 1:]

def mmap_bvecs(fname):
    x = np.memmap(fname, dtype='uint8', mode='r')
    d = x[:4].view('int32')[0]
    return x.reshape(-1, d + 4)[:, 4:]

def read_fbin(fname):
    x = np.memmap(fname, dtype='int32', mode='r')
    dim = x[1]
    return x[2:].view('float32').reshape(-1, dim)

def parquet_read(path):
    df = pd.read_parquet(path)
    x = np.stack(df['emb'].infer_objects(), axis=0)
    return x

def read_bin(path):
    fq = open(path, 'rb')
    nvecs = struct.unpack('i', fq.read(4))[0]
    dim = struct.unpack('i', fq.read(4))[0]
    vectors = np.frombuffer(fq.read(nvecs * dim * 4), dtype=np.float32).reshape((nvecs, dim))
    return vectors

def parquet_gt(path):
    df = pd.read_parquet(path)
    gt = np.stack(df['neighbors_id'].infer_objects(), axis=0)
    return gt

def generate_gt_nn(xb, xq, k=100):
    index = faiss.IndexFlatL2(xb.shape[-1])
    index.add(xb)
    _, gt = index.search(xq, k=k)
    return gt

def load_sift1m(data_root, train_size=5*10**5, test_size=10**6, mode='train'):
    basedir = os.path.join(data_root, 'sift1m/')
    xt = mmap_fvecs(basedir + 'sift_learn.fvecs')[:train_size]
    xb = mmap_fvecs(basedir + 'sift_base.fvecs')[:test_size]
    xq = mmap_fvecs(basedir + 'sift_query.fvecs')
    gt = generate_gt_nn(xb, xq)
    xt, xb, xq = sanitize(xt), sanitize(xb), sanitize(xq)
    return xt, xb, xq, gt

def load_gist1m(data_root, train_size=5*10**5, test_size=10**6):
    basedir = os.path.join(data_root, 'gist1m/')
    xt = mmap_fvecs(basedir + 'gist_learn.fvecs')[:train_size]
    xb = mmap_fvecs(basedir + 'gist_base.fvecs')[:test_size]
    xq = mmap_fvecs(basedir + 'gist_query.fvecs')
    xt, xb, xq = sanitize(xt), sanitize(xb), sanitize(xq)
    gt = generate_gt_nn(xb, xq)
    return xt, xb, xq, gt

def load_cohere1m(data_root, train_size=10**5, test_size=None):
    basedir = os.path.join(data_root, 'cohere1m/')
    xb = parquet_read(basedir + "shuffle_train.parquet")
    xt = xb[:train_size]
    xb = xb[train_size:]
    xq = parquet_read(basedir + 'test.parquet')
    xt, xb, xq = sanitize(xt), sanitize(xb), sanitize(xq)
    gt = generate_gt_nn(xb, xq)
    return xt, xb, xq, gt

def load_bigann1m(data_root, train_size=5*10**5, test_size=10**6):
    basedir = os.path.join(data_root, 'bigann1b/')
    xt = mmap_bvecs(basedir + 'bigann_learn.bvecs')[:train_size]
    xb = mmap_bvecs(basedir + 'bigann_base.bvecs')[:test_size]
    xq = mmap_bvecs(basedir + 'bigann_query.bvecs')
    xt, xb, xq = sanitize(xt), sanitize(xb), sanitize(xq)
    gt = generate_gt_nn(xb, xq)
    return xt, xb, xq, gt

def load_bigann10m(data_root, train_size=5*10**5, test_size=10**7):
    basedir = os.path.join(data_root, 'bigann1b/')
    xt = mmap_bvecs(basedir + 'bigann_learn.bvecs')[:train_size]
    xb = mmap_bvecs(basedir + 'bigann_base.bvecs')[:test_size]
    xq = mmap_bvecs(basedir + 'bigann_query.bvecs')
    xt, xb, xq = sanitize(xt), sanitize(xb), sanitize(xq)
    gt = generate_gt_nn(xb, xq)
    return xt, xb, xq, gt

def load_deep1m(data_root, train_size=5*10**5, test_size=10**6):
    basedir = os.path.join(data_root, 'deep1b/')
    xt = read_fbin(basedir + 'learn.350M.fbin')[:train_size]
    xb = read_fbin(basedir + 'base.1B.fbin')[:test_size]
    xq = read_fbin(basedir + 'query.public.10K.fbin')
    xt, xb, xq = sanitize(xt), sanitize(xb), sanitize(xq)
    gt = generate_gt_nn(xb, xq)
    return xt, xb, xq, gt

def load_tti1m(data_root, train_size=5*10**5, test_size=10**6):
    basedir = os.path.join(data_root, 'Text2Image-10M/')
    xt = read_fbin(basedir + 'query.learn.50M.fbin')[:train_size]
    xb = read_fbin(basedir + 'base.1M.fbin')[:test_size]
    xq = read_fbin(basedir + 'query.public.100K.fbin')
    xt, xb, xq = sanitize(xt), sanitize(xb), sanitize(xq)
    gt = generate_gt_nn(xb, xq)
    return xt, xb, xq, gt

def load_mmws(data_root, train_size=5*10**5, test_size=10**6):
    basedir = os.path.join(data_root, 'MS-MARCO-Web-Search/')
    vectors = read_bin(basedir + 'vectors.bin')
    xt = vectors[:train_size]
    xb = vectors[train_size:train_size+test_size]
    xq = read_bin(basedir + 'query.bin')
    xt, xb, xq = sanitize(xt), sanitize(xb), sanitize(xq)
    gt = generate_gt_nn(xb, xq)
    return xt, xb, xq, gt

def load_deep10m(data_root, train_size=5*10**5, test_size=10**7):
    basedir = os.path.join(data_root, 'deep1b/')
    xt = read_fbin(basedir + 'learn.350M.fbin')[:train_size]
    xb = read_fbin(basedir + 'base.1B.fbin')[:test_size]
    xq = read_fbin(basedir + 'query.public.10K.fbin')
    xt, xb, xq = sanitize(xt), sanitize(xb), sanitize(xq)
    gt = generate_gt_nn(xb, xq)
    return xt, xb, xq, gt

def load_deep1b(data_root, train_size=5*10**5, test_size=10**9):
    basedir = os.path.join(data_root, 'deep1b/')
    xt = read_fbin(basedir + 'learn.350M.fbin')[:train_size]
    xb = read_fbin(basedir + 'base.1B.fbin')[:test_size]
    xq = read_fbin(basedir + 'query.public.10K.fbin')
    xt, xb, xq = sanitize(xt), sanitize(xb), sanitize(xq)
    gt = generate_gt_nn(xb, xq)
    return xt, xb, xq, gt


READER = {
    'sift1m': load_sift1m,
    'mmws1m': load_mmws,
    'gist1m': load_gist1m,
    'cohere1m': load_cohere1m,
    'bigann1m': load_bigann1m,
    'bigann10m': load_bigann10m,
    'deep1m': load_deep1m,
    'deep10m': load_deep10m,
    'tti1m': load_tti1m
}
    

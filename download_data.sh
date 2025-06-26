#!/bin/bash
set -ex

# download sift1m
mkdir -p data/sift1m
if [ ! -f "data/sift1m/sift.tar.gz" ]; then
  curl -L -o data/sift1m/sift.tar.gz ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
fi
tar xzf data/sift1m/sift.tar.gz -C data/sift1m/ --strip-components=1

echo "Sift1m dataset download finished."

# download bigann
mkdir -p data/bigann
if [ ! -f "data/bigann/bigann_query.bvecs.gz" ]; then
  curl -L -o data/bigann/bigann_query.bvecs.gz ftp://ftp.irisa.fr/local/texmex/corpus/bigann_query.bvecs.gz
fi
gunzip -c data/bigann/bigann_query.bvecs.gz > data/bigann/bigann_query.bvecs

if [ ! -f "data/bigann/bigann_base.bvecs.gz" ]; then
  curl -L -o data/bigann/bigann_base.bvecs.gz ftp://ftp.irisa.fr/local/texmex/corpus/bigann_base.bvecs.gz
fi
gunzip -c data/bigann/bigann_base.bvecs.gz > data/bigann/bigann_base.bvecs

if [ ! -f "data/bigann/bigann_learn.bvecs.gz" ]; then
  curl -L -o data/bigann/bigann_learn.bvecs.gz ftp://ftp.irisa.fr/local/texmex/corpus/bigann_learn.bvecs.gz
fi
gunzip -c data/bigann/bigann_learn.bvecs.gz > data/bigann/bigann_learn.bvecs

echo "Bigann dataset download finished."

# download deep
mkdir -p data/deep

if [ ! -f "data/deep/query.public.10K.fbin" ]; then
  curl -L -o data/deep/query.public.10K.fbin https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/query.public.10K.fbin
fi

if [ ! -f "data/deep/base.1B.fbin" ]; then
  curl -L -o data/deep/base.1B.fbin https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/base.1B.fbin
fi

if [ ! -f "data/deep/learn.350M.fbin" ]; then
  curl -L -o data/deep/learn.350M.fbin https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/learn.350M.fbin
fi

# download gist1m
mkdir -p data/gist1m

if [ ! -f "data/gist1m/gist.tar.gz" ]; then
  curl -L -o data/gist1m/gist.tar.gz ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz
fi
tar xzf data/gist1m/gist.tar.gz -C data/gist1m/ --strip-components=1

echo "Gist1m dataset download finished."

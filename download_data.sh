#!/bin/sh
set -ex

if [ $# -eq 0 ]; then
  set -- sift1m bigann deep gist1m
fi

for dataset in "$@"; do
  case "$dataset" in
    sift1m)
      mkdir -p data/sift1m
      wget -c ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz -O data/sift1m/sift.tar.gz --show-progress
      tar xzf data/sift1m/sift.tar.gz -C data/sift1m --strip-components=1
      echo "Sift1m dataset download finished."
      ;;
    bigann)
      mkdir -p data/bigann
      wget -c ftp://ftp.irisa.fr/local/texmex/corpus/bigann_query.bvecs.gz -O data/bigann/bigann_query.bvecs.gz --show-progress
      gunzip -c data/bigann/bigann_query.bvecs.gz > data/bigann/bigann_query.bvecs

      wget -c ftp://ftp.irisa.fr/local/texmex/corpus/bigann_base.bvecs.gz -O data/bigann/bigann_base.bvecs.gz --show-progress
      gunzip -c data/bigann/bigann_base.bvecs.gz > data/bigann/bigann_base.bvecs

      wget -c ftp://ftp.irisa.fr/local/texmex/corpus/bigann_learn.bvecs.gz -O data/bigann/bigann_learn.bvecs.gz --show-progress
      gunzip -c data/bigann/bigann_learn.bvecs.gz > data/bigann/bigann_learn.bvecs

      echo "Bigann dataset download finished."
      ;;
    deep)
      mkdir -p data/deep

      wget -c https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/query.public.10K.fbin -O data/deep/query.public.10K.fbin --show-progress

      wget -c https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/base.1B.fbin -O data/deep/base.1B.fbin --show-progress

      wget -c https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/learn.350M.fbin -O data/deep/learn.350M.fbin --show-progress

      echo "Deep dataset download finished."
      ;;
    gist1m)
      mkdir -p data/gist1m
      wget -c ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz -O data/gist1m/gist.tar.gz --show-progress
      # curl -C - -o data/gist1m/gist.tar.gz ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz
      tar xzf data/gist1m/gist.tar.gz -C data/gist1m --strip-components=1
      echo "Gist1m dataset download finished."
      ;;
    *)
      echo "Unknown dataset: $dataset" >&2
      exit 1
      ;;
  esac
done
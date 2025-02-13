# hash:sha256:a2c74eb03bb60819c241df99be6c9b356f770624f30a88637085cd1e28bbaf8c
FROM registry.codeocean.com/codeocean/miniconda3:4.9.2-python3.8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3-dev=3.8.2-0ubuntu2 \
    && rm -rf /var/lib/apt/lists/*

RUN conda install -y \
        jupyter==1.0.0 \
        matplotlib==3.3.4 \
        pandas==1.3.3 \
        scipy==1.4.1 \
        seaborn==0.11.2 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    biopandas==0.2.9 \
    biopython==1.79 \
    cmake==3.21.3 \
    cython==0.29.24 \
    networkx==2.6.3 \
    numpy==1.21.2 \
    powerlaw==1.5 \
    prody==2.0 \
    rampy==0.4.9 \
    scikit-learn==1.0 \
    tensorflow==2.3.0 \
    xgboost==1.4.2

RUN pip3 install -U --no-cache-dir \
    graphriccicurvature==0.5.3

COPY postInstall /
RUN /postInstall

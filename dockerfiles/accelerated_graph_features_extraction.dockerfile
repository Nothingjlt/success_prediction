# For more information, please refer to https://aka.ms/vscode-docker-python
FROM kundajelab/cuda-anaconda-base

# set bash as current shell
RUN chsh -s /bin/bash
SHELL ["/bin/bash", "-c"]

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y build-essential python3-dev
RUN conda update -n base -c defaults conda

WORKDIR /graph_measures/features_algorithms/accelerated_graph_features/src
COPY ./lib/graph_measures/features_algorithms/accelerated_graph_features/src /graph_measures/features_algorithms/accelerated_graph_features/src
RUN conda env create -f env.yml

RUN echo "conda activate boost" >> ~/.bashrc
SHELL ["conda", "run", "-n", "boost", "/bin/bash", "-c"]

RUN make -f Makefile-gpu

# WORKDIR /graph_measures
# COPY ./lib/graph_measures /graph_measures

RUN apt-get install -y nano

WORKDIR /app
COPY . /app

RUN rm -rf /app/lib/graph_measures/features_algorithms/accelerated_graph_features/src && \
    ln -s /graph_measures/features_algorithms/accelerated_graph_features/src /app/lib/graph_measures/features_algorithms/accelerated_graph_features

ENV PATH /opt/conda/envs/boost/bin:$PATH
ENV CONDA_DEFAULT_ENV boost

# # Creates a non-root user with an explicit UID and adds permission to access the /app folder
# # For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
# RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
# USER appuser

# # During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
# CMD ["python3", "run_gcn_rnn_trial.py"]

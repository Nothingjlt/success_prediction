# For more information, please refer to https://aka.ms/vscode-docker-python
FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel

# set bash as current shell
RUN chsh -s /bin/bash
SHELL ["/bin/bash", "-c"]

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y build-essential cmake git

RUN conda install -c pytorch magma-cuda110

ENV CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}

RUN git clone --recursive https://github.com/pytorch/pytorch && \
    cd pytorch && \ 
    # if you are updating an existing checkout
    git submodule sync && \
    git submodule update --init --recursive && \
    python setup.py install

RUN python3 -m pip install torch-scatter && \
python3 -m pip install torch-sparse && \
python3 -m pip install torch-cluster && \
python3 -m pip install torch-spline-conv && \
python3 -m pip install torch-geometric

# Install pip requirements
COPY python_env_requirements_docker.txt .
RUN python3 -m pip install -r python_env_requirements_docker.txt

WORKDIR /app
COPY . /app

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD ["/bin/bash"]

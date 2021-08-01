# For more information, please refer to https://aka.ms/vscode-docker-python
FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel

# set bash as current shell
RUN chsh -s /bin/bash
SHELL ["/bin/bash", "-c"]

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y build-essential

RUN python3 -m pip install torch-scatter==2.0.6 -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html && \
python3 -m pip install torch-sparse==0.6.9 -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html && \
python3 -m pip install torch-cluster==1.5.9 -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html && \
python3 -m pip install torch-spline-conv==1.2.1 -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html && \
python3 -m pip install torch-geometric==1.7.0

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
CMD ["python3", "run_gcn_rnn_trial.py"]

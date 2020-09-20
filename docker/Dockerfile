FROM ubuntu:18.04

LABEL maintainer="mike-sino"
LABEL version="1.0"

# Needed for string substitution 
SHELL ["/bin/bash", "-c"]
# Pick up some develop dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gcc make cmake g++ python3-setuptools python3-pip python3.7 python3.7-dev \
        nodejs npm curl unzip git
RUN python3.7 -m pip install pip

# Install Jupyter
RUN pip3 install jupyter
RUN pip3 install ipywidgets
RUN jupyter nbextension enable --py widgetsnbextension

# Install JupyterLab
RUN pip3 install jupyterlab
RUN jupyter serverextension enable --py jupyterlab
RUN mkdir /notebook/

# Install jupyterlab extensions based on nodejs and npm 
# RUN jupyter labextension install jupyterlab-drawio && \
#     jupyter labextension install jupyterlab-chart-editor

# Install Python Packages & Requirements
COPY init/python3/requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

# Install Minimal C kernel for jupyter
RUN pip3 install jupyter-c-kernel && \
    install_c_kernel

# Set environment
ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8

# TensorBoard
EXPOSE 6006

# Jupyter
EXPOSE 8888

# Flask
EXPOSE 8080

CMD [ "/bin/bash" ]

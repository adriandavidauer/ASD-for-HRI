# ASD-for-HRI
A scalable Active Speaker Detection for Human-Robot Interaction

## Install
In your Env run 
```bash
pip install -r requirements.txt
```
# Contributing

## Devcontainer
We provide a Devcontainer for the development of ASD4HRI.
### Prerequisites
You need to [install Docker](https://docs.docker.com/engine/install/) and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/1.14.1/install-guide.html) if you whish to use GPU support.

### Usage of Data in the Container
The Container does not safe any data. If you want to use Data for Experiments, you can use `/home/vscode/host-home/` which is mounted from your local home directory.
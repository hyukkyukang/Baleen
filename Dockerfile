FROM nvidia/cuda:12.0.1-devel-ubuntu22.04

# Install basic packages
RUN apt update
RUN apt install gnupg git curl make g++ wget zip vim sudo -y
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata

# Install postgresql
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install postgresql postgresql-contrib libpq-dev

# Set timezone
RUN ln -fs /usr/share/zoneinfo/Asia/Seoul /etc/localtime && dpkg-reconfigure -f noninteractive tzdata

# python alias to python3
RUN apt install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa -y
RUN apt update
RUN apt install python3.10 -y
RUN apt-get -y install python3-pip python-is-python3
RUN echo "EXPORT PYTHONPATH=./" >> ~/.bashrc
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Install basic packages
RUN apt update
RUN apt install gnupg git curl make g++ wget zip vim sudo -y
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata

# Install postgresql
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install postgresql postgresql-contrib libpq-dev

# Set timezone
RUN ln -fs /usr/share/zoneinfo/Asia/Seoul /etc/localtime && dpkg-reconfigure -f noninteractive tzdata

# Install prerequisites for python3.11
RUN apt install build-essential checkinstall libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev -y 
# Install python3.11
RUN apt install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa -y
RUN apt update
RUN apt install python3.11 python3.11-dev -y
RUN apt-get -y install python3-pip python-is-python3
RUN echo "export PYTHONPATH=./" >> ~/.bashrc
RUN echo "export CONFIGPATH=./config.yml" >> ~/.bashrc
# Set default python version to 3.11
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1    

FROM ubuntu:22.04

# Install basic packages
RUN apt update
RUN apt install gnupg git curl make g++ wget zip vim sudo -y
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata

# Install postgresql
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install postgresql postgresql-contrib libpq-dev

# Set timezone
RUN ln -fs /usr/share/zoneinfo/Asia/Seoul /etc/localtime && dpkg-reconfigure -f noninteractive tzdata

# TODO: Install code and requirements
# RUN git clone https://oauth2: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@github.com/hyukkyukang/table-to-text.git /home/table-to-text
# RUN cd /home/table-to-text && pip install -r requirements.txt
# TODO: Install transformers from source
# RUN python -c "import transformers; transformers.T5ForConditionalGeneration.from_pretrained('t5-small')"

# python alias to python3
RUN apt install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa -y
RUN apt update
RUN apt apt install python3.10 -y
RUN apt-get -y install python3-pip python-is-python3
RUN echo "EXPORT PYTHONPATH=./" >> ~/.bashrc
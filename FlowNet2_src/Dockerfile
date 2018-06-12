FROM nvidia/cuda:8.0-devel-ubuntu16.04

RUN apt-get update -y
RUN apt-get install -y build-essential libpq-dev libssl-dev openssl libffi-dev zlib1g-dev
RUN apt-get install -y wget software-properties-common

RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt update 
RUN apt install -y python3.6 python3.6-dev
RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python3.6 get-pip.py

RUN rm /usr/bin/python3
RUN rm /usr/local/bin/pip3
RUN ln -s /usr/bin/python3.6 /usr/local/bin/python3
RUN ln -s /usr/local/bin/pip /usr/local/bin/pip3
RUN echo 'alias python="/usr/bin/python3.6"' >> ~/.bashrc

RUN pip3 install --upgrade pip
RUN pip3 install torch torchvision cffi tensorboardX 
RUN pip3 install tqdm pypng scipy scikit-image colorama==0.3.7 
RUN pip3 install setproctitle pytz requests

RUN echo 'cd FlowNet2_src/' >> ~/.bashrc
RUN echo 'source install.sh' >> ~/.bashrc
RUN echo 'cd' >> ~/.bashrc


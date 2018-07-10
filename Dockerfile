FROM nvidia/cuda:8.0-devel-ubuntu16.04

RUN apt-get update -y
RUN apt-get install -y build-essential libpq-dev libssl-dev openssl libffi-dev zlib1g-dev
RUN apt-get install -y wget software-properties-common

RUN add-apt-repository -y ppa:jonathonf/ffmpeg-3
RUN apt-get install -y ffmpeg

RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update 
RUN apt-get install -y python3.6 python3.6-dev python3.6-tk
RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python3.6 get-pip.py

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
RUN echo 'cd /app' >> ~/.bashrc


RUN apt-get -y update
RUN apt-get -y install wget unzip \
                       build-essential cmake git pkg-config libatlas-base-dev gfortran \
                       libjasper-dev libgtk2.0-dev libavcodec-dev libavformat-dev \
                       libswscale-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libv4l-dev

RUN pip3 install numpy





RUN wget https://github.com/Itseez/opencv/archive/3.2.0.zip && unzip 3.2.0.zip \
    && mv opencv-3.2.0 /opencv

RUN mkdir /opencv/build
WORKDIR /opencv/build

RUN cmake -D BUILD_TIFF=ON \
		-D BUILD_opencv_java=OFF \
		-D WITH_CUDA=ON \
		-D ENABLE_AVX=ON \
		-D WITH_OPENGL=ON \
		-D WITH_OPENCL=ON \
		-D WITH_IPP=OFF \
		-D WITH_TBB=ON \
		-D WITH_EIGEN=ON \
		-D WITH_V4L=ON \
		-D WITH_VTK=OFF \
		-D BUILD_TESTS=OFF \
		-D BUILD_PERF_TESTS=OFF \
		-D CMAKE_BUILD_TYPE=RELEASE \
		-D BUILD_opencv_python2=OFF \
		-D CMAKE_INSTALL_PREFIX=$(python3.6 -c "import sys; print(sys.prefix)") \
		-D PYTHON3_EXECUTABLE=$(which python3.6) ..


RUN make -j8
RUN make install

CMD ["bash"]

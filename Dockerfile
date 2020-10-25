FROM tensorflow/tensorflow:2.3.1-jupyter

USER root
RUN jupyter nbextension enable --py --sys-prefix widgetsnbextension
ENV DEBIAN_FRONTEND noninteractive

COPY requirements.txt requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 8888

WORKDIR /tf

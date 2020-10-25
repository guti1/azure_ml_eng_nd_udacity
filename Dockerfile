FROM python:3.6-stretch

ENV PYTHONUNBUFFERED=1

USER root
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y apt-utils build-essential

RUN curl -sL https://deb.nodesource.com/setup_14.x | bash -
RUN  apt-get install -y  nodejs npm

COPY . /app

RUN pip install --upgrade pip
RUN pip install poetry==1.1.4
COPY poetry.lock pyproject.toml ./

RUN poetry config virtualenvs.create false \
  && poetry install

RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager
RUN jupyter nbextension enable --py --sys-prefix widgetsnbextension
#COPY requirements.txt requirements.txt

#RUN pip install -r requirements.txt

EXPOSE 8888

WORKDIR /app

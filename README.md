# Machine Learning Engineer with Microsoft Azure nanodegree - Udacity

Resources and projects for Udacity's Azure ML engineer nanodegree

## 1. Setting up your environment.

We use docker, thus the dev-env is set up based on the [Dockerfile](./Dockerfile), the dependencies are managed by 
[Poetry](https://python-poetry.org/). 

- Clone the repo, create a `.env` file in the project root with your azure subscription-id.
- Build the docker image by `docker-compose build --force-rm`.
- Start jupyterlab by `docker-compose up jupyter`
- You can add additional packages with poetry by `poetry add...` and `poetry install..,` etc.

Of course you can also use pyenv + virtualenv with poetry to set up the env. We suggest to use the python version from 
the [Dockerfile](./Dockerfile) (e.g. 3.7.7), and create a virtualenv based on that, afterwards activating it install 
poetry e.g. by executing `pip install poetry`. Once poetry is available just run `poetry install` from the project root.
The dependencies are managed by poetry as in the docker version.

## 2. Project 1 - Optimizing an ML Pipeline in Azure

In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model. 
This model is then compared to an Azure AutoML run. The project files can be found 
[here](./project1_optimizing_ml_pipeline). 
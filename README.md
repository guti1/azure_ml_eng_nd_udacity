# Machine Learning Engineer with Microsoft Azure nanodegree - Udacity

Resources and projects for Udacity's Azure ML engineer nanodegree

## 1. Setting up your environment.

We use docker, thus the dev-env is set up based on the [Dockerfile](./Dockerfile), the dependencies are managed by [Poetry](https://python-poetry.org/). 

- Clone the repo, create a `.env` file in the project root with your azure subscription-id.
- Build the docker image by `docker-compose build --force-rm`.
- Start jupyterlab by `docker-compose up jupyter`
- You can add additional packages with poetry by `poetry add...` and `poetry install..,` etc.

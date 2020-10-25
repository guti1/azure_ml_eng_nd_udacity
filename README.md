# Machine Learning Engineer with Microsoft Azure nanodegree - Udacity

Resources and projects for Udacity's Azure ML engineer nanodegree

## 1. Setting up your environment.

We use docker, thus the dev-env is set up based on the [Dockerfile](./Dockerfile), the requirements are defined in the 
[requirements.txt](./requirements.txt). Please be aware that our env is based on the tf-jupyter Docker image, thus 
it can be considered as quite heavy, but since to solve the project we also would like to experiment with different 
models, approaches I chose this as my primary dev env.  

We store our azure-id in a `.env` file in the project root. Thus before executing all the subsequent notebook, code, 
please make sure it is correctly set up, or just modify the notebooks to your needs accordingly.  
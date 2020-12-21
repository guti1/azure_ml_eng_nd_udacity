## 1. Setting up your environment.

We use docker, thus the dev-env is set up based on the [Dockerfile](../Dockerfile), the dependencies are managed by 
[Poetry](https://python-poetry.org/). 

- Clone the repo, create a `.env` file in the project root with your azure subscription-id.
- Build the docker image by `docker-compose build --force-rm`.
- Start jupyterlab by `docker-compose up jupyter`
- You can add additional packages with poetry by `poetry add...` and `poetry install..,` etc.


## 2. Summary of the problem

The provided dataset is stemming form the 
[UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/bank+marketing). 
It contains data related to a direct marketing campaigns of a financial institution. The problem setting is a binary 
classification where our project goal is to predict whether a client will subscribe a term deposit (that is outcome 
variable y). In the original dataset we have 17 possible predictor variables, both numeric and categorical ones.

During the project we would like to predict whether a given client with a set of previously known attributes (set of 
inputs) would or would not subscribe a term deposit. 

We would construct a default Ml pipeline for that with the basic steps of:
1. Data cleaning and preparation
2. Model building - including optimizing the hyperparameters as well (on a test/valid set).
3. Model performance evaluation step on a held-out-set.

Thus we would use all the 3 part of the dataset namely the train / test / validation sets for different purposes. We 
show how to set up the above pipeline from a jupyter-notebook and execute it through the Azure ML studio. For the 
hyperparameter tuning part we use Hyperdrive functionality of the ML-studio. 

As an alternative to our classical ml-pipeline we also evaluate the achievable performance by the Azure AutoML framework
as well. First we present the steps and the following results in a [notebook](./udacity_project1_solution.ipynb). 


 ## 3. Scikit-learn pipeline
As a benchmark implementation we chose the 
[logistic-regression model](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) 
from the scikit-learn package. As a first iteration we execute the complete pipeline in a single step, namely: 
1. Preprocess and cleanse the data
2. Model-fitting
3. Evaluation on a held-out-set
 
### 3.a. Data
The train dataset originally contains 32950 records, however it contains some duplicate rows, thus we decided to remove
these resulting in 32941 records. Our dataset is highly imbalanced regarding the target variable: it has 29250 `no` 
values. Thus, our 'baseline' accuracy could be 0.887 if we predict `no` as our majority class independently of 
anything. We could approach the issue by e.g. with a sampling based approach (e.g. over/under sampling, or a synthetic 
minority oversampling technique (SMOTE)), or we could use cost sensitive learning as well, by weighting our samples.


First we just use the provided example for preprocessing, maybe as next step we vould evaluate the above mentioned
alternatives. 

### 3.b Algorithms
As baseline solution for the classification problem we choose logistic regression. In the scikit-learn implementation 
we choose to optimize the `C` regularization parameter along with the `max_iter` prameter for controlling the possible 
number of iteration until the algorithm converge. We use the BanditPolicy for the optimization, 
the details about it can be found 
[here](https://docs.microsoft.com/hu-hu/python/api/azureml-train-core/azureml.train.hyperdrive.banditpolicy?view=azure-ml-py).

The optimization policy defines the early termination strategy of the executed runs, namely un that doesn't fall within 
the slack factor or slack amount of the evaluation metric with respect to the best performing run will be terminated. 

As a `challenger` model we use [Lightgbm](https://github.com/microsoft/LightGBM), which is a learning algorithm based on
decision-tree based weak learners which are fit on the training set sequentially (boosting). Since lgbm has a lot of 
tunable hyperparams (see [here](https://lightgbm.readthedocs.io/en/latest/Parameters.html)) we use 100 hyperdrive runs 
to optimize a few from them:
- learning_rate: - controlling the learning - shrinkage rate
- max_depth: limit the max depth for tree model
- num_leaves: max number of leaves in one tree
- min_data_in_leaf: minimal number of data in one leaf
- num_iterations: number of boosting rounds


For the logistic regression we use random parameter sampling for the logistic-regression case and bayesian parameter
sampling for the lgbm run. In random sampling, hyperparameter values are randomly selected from the defined search 
space. The bayesian algorithm  picks samples based on how previous samples performed, so that new samples improve the 
primary metric. In that case it is recommended to set the number of runs greater than or equal to 20 times the number of 
hyperparameters being tuned. 

During the hyperdrive runs we choose accuracy as our primary metric, and we would like to maximize it. 


The models are executed as standalone scripts, see:
- [logistic regression script](./scripts/logit_train.py)
- [lightgbm script](./scripts/logit_train.py)

The scripts are structured as:
1. We are referencing the datasets and executing the predefined cleaning steps.
2. Afterwards the training with the predefined set of hyperparameters as inputs are taking place.
3. The trained model is evaluated on the validation dataset, and the chosen accuracy metrics are reported back.
4. save the serialized model to use it later (e.g. for inferencing on fresh data).
 

## 4. AutoML
We compared our previous models to a more-or-less fully automated autoML run. During that no preprocessing step is 
necessary, thus we just defined the task tpye, set some config parameters and referenced the train and validation 
datasets which were made available to our workspace in a previous step.

After the execution the job, it resulted in 
[VotingEnsemble](https://docs.microsoft.com/en-gb/python/api/azureml-train-automl-runtime/azureml.train.automl.runtime.ensemble.votingensemble?view=azure-ml-py) 
as the best performing model, based on the validation set. As in the previous experiments we downloaded the resulting 
model and evaluated it on the held-out test set.

## 5. Results - pipeline comparison

The full pipeline can be found in the notebook located [here](./udacity_project1_solution.ipynb). 
The final performance on the test set was:
1. Hyperdrive - lgbm: 0.9175
2. AutoML - VotingEnsemble: 0.9163
3. Hyperdrive -  LogisticRegression: 0.9112

## 6. Next steps / TODO / misc:
 - [ ] Present resulting models from the model-dumps in a standalone notebook.
 - [ ] Deprecate SKLearn estimator type function in the favour of `ScriptRunConfig`
 - [ ] Review data-preprocessing steps - onehotencoding, further feature engineering, handle imbalanced classes.
 - [ ] Refactor exec notebook as series of azure-pipeline steps for better reproducibility

Known issues:

Unfortunately executing `poetry add azureml-train-automl-runtime` will result in never ending dependency resolution, 
thus it is not included in the project dependencies. However executing 
`poetry run pip istall azureml-train-automl-runtime` successfully installs the necessary package to execute a model 
resulting from a remote autoML run locally.

polarseg-kitti
==============================

Re-implementation of the PolarNet architecture, developed by Zhang et al.

## Initial steps

Start by autoatcially generating an environment for *conda* or *venv* with the following command in the top directory:

`$ make create_environment` 

After activating this environment, install the requirements:

`$ make requirements`

Finally, initialize the [pre-commit](https://pre-commit.com/) git hooks:

`$ make lint`

## Logging

Logging is done through the [Weights & Biases](https://wandb.ai/) for which, more detailed documentation can be found [here](https://docs.wandb.ai/).

After creating an account, for automatic log-in, copy your API key to the `.env` file in the root folder. If you do NOT want to log the experiment, set `wandb:False` in `config/{your_config}.yaml`.

`WANDB_API_KEY = $YOUR_API_KEY `

> Note: You might need to create the project yourself.

## Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── config             <- configuration files for training and dataloaders
    ├── data
    │   ├── debug          <- Small amount of data separated for debugging purposes.
    |   └── sequences      <- Read more about the data setup in the other README
    |       ├── 00/           
    |       ├── velodyne/
    |       └── labels/
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

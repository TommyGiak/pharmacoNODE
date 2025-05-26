
| **Authors**  | **Project** | **License** |
|:------------:|:-----------:|:-----------:|
| [**T. Giacometti**](https://github.com/TommyGiak) | **pharmacoNODE**<br> | [![license](https://img.shields.io/github/license/mashape/apistatus.svg)](https://github.com/TommyGiak/pharmacoNODE/blob/main/LICENSE) |

<a href="https://github.com/UniboDIFABiophysics">
  <div class="image">
    <img src="https://cdn.rawgit.com/physycom/templates/697b327d/logo_unibo.png" width="90" height="90">
  </div>
</a>

# pharmacoNODE

## Neural Ordinary Differential Equations for pharmacokinetics

Implementation of the NODE algorithm used to train and test NODE in pharmacokinetics.

* [Overview](#overview)
* [Prerequisites](#prerequisites)
* [Usage](#usage)
* [Contribution](#contribution)
* [Authors](#authors)
* [License](#license)
* [Acknowledgment](#acknowledgment)
* [Citation](#citation)

## Overview

This study investigates the use of Neural Ordinary Differential Equations (NODEs) as an alternative to traditional compartmental models and Nonlinear Mixed-Effects (NLME) models for drug concentration prediction in pharmacokinetics. Unlike standard models that rely on strong assumptions and often struggle with high-dimensional covariate relationships, NODEs offer a data-driven approach, learning differential equations directly from data while integrating covariates. To evaluate their performance, NODEs were applied to a real-world Dalbavancin pharmacokinetic dataset comprising 218 patients and compared against a two-compartment model and a NLME within a cross-validation framework, which ensures an evaluation of robustness. Given the challenge of limited data availability, a data augmentation strategy was employed to pre-train NODEs. Their predictive performance was assessed both with and without covariates, while model explainability was analyzed using Shapley additive explanations (SHAP) values. Results show that, in the absence of covariates, NODEs performed comparably to state-of-the-art NLME models. However, when covariates were incorporated, NODEs demonstrated superior predictive accuracy. SHAP analyses further revealed how NODEs leverage covariates in their predictions. These results establish NODEs as a promising alternative for pharmacokinetic modeling, particularly in capturing complex covariate interactions, even when dealing with sparse and small datasets, thus paving the way for improved drug concentration predictions and personalized treatment strategies in precision medicine.

The `pharmacoNODE` package provides a simple _python_ implementation of the algorithm used to train and test Neural ODE for a _dalbavancin_ application in a pharmacokinetic settings. In particular, the NODE is used to model the differential equation that governs the _dalbavancin_ concentration curve along time.

The implementation in this package can take a pharmacokinetic dataset in the same format used in [**Monolix**](https://lixoft.com/products/monolix/) (format details [here](https://monolixsuite.slp-software.com/monolix/2024R1/data-format)), which is very similar also to the [**NONMEM**](https://www.iconplc.com/solutions/technologies/nonmem) format. Up to now, the accepted columns are:

* **ID**: identifier of the individual
* **TIME**: time of the dose or observation record
* **TINF**: duration of the dose administration
* **AMT**: dose amount
* **Y**: oberviation, records the measurement/observation
* **EVID**: event ID, identifier to indicate if the line is a dose-line or a response-line, $0$ for and observation event and $1$ for a dose event
* **covariates**: evevntual continuous or categorical columns for the covariate of each individual (e.g. **Bw** body weight, **Age**, etc...)

This columns are setted in the [`config.ini`](https://github.com/TommyGiak/pharmacoNODE/blob/main/config.ini) file.

## Prerequisites

The _python_ prerequisites are:

* torch
* numpy
* scipy
* pandas
* matplotlib
* torchdiffeq

## Usage

To use the package is sufficient to clone this repository, insert the data file in the `./data/` folder. For stability of the model it is recomendable to standardize all the inputs column in a range around $[0,1]$.

To start a training for of a NODE model can be used the [`main.py`](https://github.com/TommyGiak/pharmacoNODE/blob/main/main.py) file running:

```bash
python main.py
```

The _main_ script trais and save the model using the same pipeline used described in the paper.

To infere patients using the trained model can be used the [`predict.py`](https://github.com/TommyGiak/pharmacoNODE/blob/main/predict.py) file. To insert the informations of a specific patient it is needed a `csv` file with the same format of the data file containing the doses given with the corrisponding time (so `EVID=1`), `TINF` and an ID (eventually also its covariates). Then the predict script can be called with the ID of the patients to be predicted:

```bash
python predict.py <ID>
```

the output is a plot with the concentration curve predicted.

:warning: The data folder is **empty** by default for privacy of the dataset and it is available only **on request**.

The settings of the model and on the training can be changed using the configuration file (see the below section [Configuration](#configuration)).

## Configuration

The configuration file [`config.ini`](https://github.com/TommyGiak/pharmacoNODE/blob/main/config.ini) contains the following contains the following options:

### [NAMES]

* model_name: name of the saved model if present, necessary to run the predict file
* col_names: name of the compulsory columns to insert in the data file
* ID_col: name of the ID column in the data file
* data_file: name if the data file
* data_file_predict: name of the file containing the subjects that must be predicted

### [NN_SETTINGS]

* dim_c: dimension of the FNN architecture of the NODE (dimension of each hidden layer)
* dim_V: dimension of the FNN architecture of the volume estimation (dimension of each hidden layer when covariates are included)
* include_covariates: True if include covariates, False otherwise

### [PATHS]

* path_data: path of the data folder
* path_models: path of the model weights folder

### [TRAIN_SETTINGS]

* epochs_synth: number of epochs of the training on synthetic generated data (following the pipeline of the paper)
* epochs_train: number of epochs of the training of the model with real data
* epochs_fine_tuning: number of epochs of the fine-tuning with real data
* lr: learning rate of the training
* lr_reduced: learning rate of the fine-tuning
* weights_decay: weight decay for the training 

## License

The `pharmacoNODE` package is licensed under the [GPL-3.0 License](https://github.com/TommyGiak/pharmacoNODE/blob/main/LICENSE)

## Contribution

Any contribution is more than welcome :heart:. Just fill an [issue](https://github.com/TommyGiak/pharmacoNODE/issues/new/choose) or a [pull request](https://github.com/TommyGiak/pharmacoNODE/compare) and we will check ASAP!

## Acknowledgment

Thanks goes to all contributors of this project.

## Authors

* <img src="https://avatars.githubusercontent.com/u/127099240?v=4" width="25px"> [<img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="27px">](https://github.com/TommyGiak) [<img src="https://cdn.rawgit.com/physycom/templates/697b327d/logo_unibo.png" width="25px">](https://www.unibo.it/sitoweb/tommaso.giacometti5) **Tommaso Giacometti**

See also the list of [contributors](https://github.com/TommyGiak/TommyGiak/contributors) [![GitHub contributors](https://img.shields.io/github/contributors/TommyGiak/pharmacoNODE?style=plastic)](https://github.com/TommyGiak/pharmacoNODE/graphs/contributors/) who participated in this project.

## Citation

If you have found `pharmacoNODE` helpful in your research, please consider citing the original paper about the wound image segmentation

```BibTex

```

# generalized_contrastive_PCA
Repository for code and scripts related to generalized contrastive PCA

[![Github All Releases](https://img.shields.io/github/downloads/SjulsonLab/generalized_contrastive_PCA/total.svg)]()

--------------------------------------------------------------------------------
### Installation

1. Install [conda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/download.html) in your machine
2. Download this repository
3. On a conda terminal pointed to your copy of this repository, run the following command:
```
  - conda env create -f environment.yml
```

###### Alternative Installation
If you have an environment you want gcPCA to, you can just refer to the class gcPCA in the file `contrastive_methods.py`, at this version you will only need to install the packages: warnings, numpy and scipy.
--------------------------------------------------------------------------------
### Usage

##### Python
To import the class and initialize the model:

```python
from contrastive_methods import gcPCA
gcPCA_model = gcPCA(method='v4.1',normalize_flag=True)
```

Methods can be 'vn.1' or just 'vn', where n can vary from 1 to 4. versions ending
in .1 will return orthogonal dimensions. 'v4.1' corresponds to the (A-B)/(A+B)
objective function, for the other versions please check the table from the SFN
poster in [here](poster_sfn/Eliezyer_Oliveira_SFN_poster)

normalize_flag will signal the function to normalize your data or not, in case
you have a custom normalization you prefer to use, set this variable to False.
Otherwise, the code will z-score and normalize the data by their respective l2-norm.

Fitting the model:
```python
gcPCA_model.fit(Ra,Rb)
```
Ra (ma x p) and Rb (mb x p) are matrices of each experimental conditions (A and B),
with rows as samples (sizes ma and mb, respectively), and p features that are the
same across the experimental conditions (neurons/channel/RNA etc)

The model will have the outputs:
```python
gcPCA_model.loadings_
gcPCA_model.gcPCA_values_
gcPCA_model.Ra_scores_
gcPCA_model.Rb_scores_
```

`gcPCA_model.loadings_`: gcPCs loadings. A matrix with loadings in the rows and
 gcPCs on the columns, ordered by their objective value.

`gcPCA_model.gcPCA_values_`: The objective value of the gcPCA model. It is an
array with the gcPCA objective value for each gcPC.

`gcPCA_model.Ra_scores_`: Ra dataset scores on the gcPCs, A matrix ma x k, with k
being the number of gcPCs, ordered by the values in gcPCA_values_

`gcPCA_model.Rb_scores_`: Rb dataset scores on the gcPCs, A matrix mb x k, with k
being the number of gcPCs, ordered by the values in gcPCA_values_

##### MATLAB

MATLAB does not require conda environment installation.
The file can be found in matlab/gcPCA

You can fit the model by running the following command:
```
 [B, S, X] = gcPCA(Ra, Rb, gcPCAversion)
```

Ra and Rb are the same matrices presented in the python version. The variable
gcPCAversion can take numerical input that varies from 1 to 4, versions that end
in .1 will return orthogonal gcPCs.

More info on the output files can be found in `help gcPCA`

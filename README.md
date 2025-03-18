# Generalized Contrastive PCA

Generalized contrastive PCA is a new dimensionality reduction method. It is a hyperparameter-free method for comparing high-dimensional datasets collected under different experimental conditions to reveal low-dimensional patterns enriched in one condition compared to the other. Unlike traditional dimensionality reduction methods like PCA, which work on a single condition, gcPCA allows for a direct comparison between conditions.

This open-source toolbox includes implementations of gcPCA in both Python and MATLAB, with variants designed for different data types. It provides a straightforward, fast, and reliable way to compare conditions.

##### Key Features
- **Hyperparameter-free**: No manual tuning required.
- **Symmetric comparison**: Both conditions are treated equally.
- **Sparse solutions**: Reduce the complexity of the results for better interpretation.
- **Multiple implementations**: Available in both Python and MATLAB (R implementation on the way!).

You can find more details in our paper at [PLOS Computational Biology](https://doi.org/10.1371/journal.pcbi.1012747)


<p align="center">
  <picture>
    <source media="(prefers-color-scheme: light)" srcset="./docs/images/gcpca_LIGHT.svg"  width="75%" title="gcpca" alt="gcpca" align="center" vspace = "100">
    <source media="(prefers-color-scheme: dark)" srcset="./docs/images/gcpca_DARK.svg" width="75%" title="gcpca" alt="gcpca" align="center" vspace = "100">
    <img alt="Display the image exemplifying the uses of gcPCA" width="75%" title="gcPCA" alt="gcpca" align="center" vspace = "200">
  </picture>
</p>

If you find this project helpful, consider supporting us by clicking the “⭐ Star” button at the top right of the repository.
--------------------------------------------------------------------------------
### Installation

1. Install [conda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/download.html) in your machine
2. Download this repository
3. On a conda terminal pointed to your copy of this repository, run the following command:
```
  - conda env create -f environment.yml
```

### Alternative Installation

If you have an environment you want to use with gcPCA, you can just refer to the class gcPCA in the file `contrastive_methods.py`, at this version the only dependencies are: warnings, numpy, scipy, and time.

--------------------------------------------------------------------------------
### Usage

##### Python
To import the class and initialize the model:

```python
from contrastive_methods import gcPCA
gcPCA_model = gcPCA(method='v4',normalize_flag=True)
```

Methods can be 'vn.1' or just 'vn', where n can vary from 1 to 4. Versions ending
in .1 will return orthogonal dimensions. 'v4.1' corresponds to the (A-B)/(A+B)
objective function, for the other versions please check for more information in the preprint in [bioRxiv](https://doi.org/10.1101/2024.08.08.607264)

normalize_flag will signal the function to normalize your data or not, in case
you have a custom normalization you prefer to use, set this variable to False.
Otherwise, the code will z-score and normalize the data by their respective l2-norm.

Fitting the model:
```python
gcPCA_model.fit(Ra,Rb)
```
Ra (ma x p) and Rb (mb x p) are matrices of each experimental condition (A and B),
with rows as samples (sizes ma and mb, respectively), and p features that are the
same across the experimental conditions (neurons/channel/RNA etc)

The model will have the following outputs:
```python
gcPCA_model.loadings_
gcPCA_model.gcPCA_values_
gcPCA_model.Ra_scores_
gcPCA_model.Ra_values_
gcPCA_model.Rb_scores_
gcPCA_model.Rb_values_
gcPCA_model.objective_values_
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

### Support and citing
If gcPCA is useful in your work, we kindly request that you cite:

>  Eliezyer F. de Oliveira, Pranjal Garg, Jens Hjerling-Leffler, Renata Batista-Brito, and Lucas Sjulson. (2025). Identifying patterns differing between high-dimensional datasets with generalized contrastive PCA. [PLOS Computational Biology]( https://doi.org/10.1371/journal.pcbi.1012747)

### Contact us
If you encounter any issues or have suggestions for improvements, feel free to open a GitHub issue. You can also reach out to the first and last authors of the gcPCA manuscript. If you find this project helpful, consider supporting us by clicking the “⭐ Star” button at the top right of the repository.

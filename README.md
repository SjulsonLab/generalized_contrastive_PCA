# Generalized Contrastive PCA

Generalized contrastive PCA (gcPCA) is a dimensionality reduction method for contrasting datasets. It identifies low-dimensional patterns that are enriched in one experimental condition relative to another. Unlike PCA, which operates on a single dataset, gcPCA enables direct comparisons across conditions.

This repository provides open-source implementations of gcPCA in Python, MATLAB, and R, along with variants for different data types. The toolbox offers a mathematically rigorous and practical approach for comparing high-dimensional datasets.

##### Key Features
- **Hyperparameter-free**: No manual tuning required.
- **Symmetric comparison**: Both conditions are treated equally.
- **Sparse solutions**: Reduce the complexity of the results for better interpretation.
- **Multiple implementations**: Available in both Python, MATLAB and R.

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

##### Python

1. Install [conda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/download.html) in your machine

2. Create an environment to use the package

```
conda create --name gcPCA python=3.12 (or any Python >= 3.9)
```

Activate the conda environment

```
conda activate gcPCA
```


3. Install the package

```
pip install generalized_contrastive_PCA
```

Alternatively, you can just install the environment from this repo environment.yml file

```
conda env create -f environment.yml
```

##### R

Install the R package from GitHub:

```r
install.packages("remotes")
remotes::install_github("SjulsonLab/generalized_contrastive_PCA", subdir = "R_package")
```

Or install it locally from a clone of this repository:

```r
install.packages("devtools")
devtools::install("R_package")
```

--------------------------------------------------------------------------------
### Usage

##### Python
To import the class and initialize the model:

```python
from generalized_contrastive_PCA import gcPCA
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

##### R

Load the package and fit a model:

```r
library(gcpca)

set.seed(1)
Ra <- matrix(rnorm(40 * 5), ncol = 5)
Rb <- matrix(rnorm(35 * 5), ncol = 5)

fit <- gcPCA(Ra, Rb, method = "v4", Ncalc = 3)
pred <- predict(fit, Ra = Ra, Rb = Rb)

pred$Ra_scores
pred$Rb_scores
```

`fit` is an S3 `"gcPCA"` object with the model loadings, scores, objective values,
and fit metadata. You can use `coef(fit)` to extract loadings and `summary(fit)`
for a concise model summary.

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
# gcpca

R package implementing regular and sparse generalized contrastive PCA (`gcPCA`).

## Installation

```r
# from remote repo
remotes::install_github("SjulsonLab/generalized_contrastive_PCA", subdir = "R_package")

# from local source
devtools::install(".")
```

## Minimal usage

```r
library(gcpca)
set.seed(1)
Ra <- matrix(rnorm(40 * 5), ncol = 5)
Rb <- matrix(rnorm(35 * 5), ncol = 5)
fit <- gcPCA(Ra, Rb, method = "v4", Ncalc = 3)
predict(fit, Ra = Ra)$Ra_scores
```

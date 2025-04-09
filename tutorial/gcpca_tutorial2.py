import marimo

__generated_with = "0.11.30"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    mo.md("# How to use sparse gcPCA")
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Introduction

        In this tutorial, I will show you how to use the sparse solution of the gcPCA class with a single-cell RNA sequencing dataset.

        Before diving into the implementation, let’s briefly review what a sparse method is.
        **If you're already familiar with sparse methods, feel free to skip this section.**

        ##### **What does a sparse solution to a method do?**

        Sparse methods are commonly used for **feature selection** and **model interpretability**. The idea is simple: we want to identify which features are most relevant in what the model has learned from the data.

        When performing regression or classification, we often want to understand how the model makes its predictions. One way to do this is by inspecting the model’s weights (also called coefficients or loadings) to see which features have the largest impact.

        However, this becomes difficult when dealing with datasets that have a **large number of features**, which is the case in biomedical data. To address this, a common solution is to add a penalty to the model that discourages large weights unless they are essential. This is the principle behind lasso and ridge regression.

        A regular linear regression minimizes the residual sum of squares:

        $\min_\beta ||Y - X\beta||^2$

        It finds the $\beta$ coefficients that minimize the difference between the predictions $X\beta$ and the observed data $Y$.

        In contrast, lasso regression adds a penalty on the sum of the absolute values of the coefficients (L1 norm), encouraging some coefficients to be exactly zero:

        $\min_\beta ||Y - X\beta||^2 + \lambda \sum |\beta_j|$


        Here, $\lambda$ is a hyperparameter that controls the strength of the penalty:

            - Higher $\lambda$ values increase the penalty, forcing more coefficients to zero.

            - Lower $\lambda$ values allow more features to be retained.

        This results in a sparse solution—only a subset of features have non-zero weights, making the model more interpretable.


        ##### **Sparsity in gcPCA**

        The sparse version of gcPCA builds on existing sparse PCA methods by incorporating an elastic net penalty, which combines both L1 (lasso) and L2 (ridge) regularization:

            L1 (lasso) promotes sparsity by driving some coefficients to zero.

            L2 (ridge) helps stabilize the solution when the data is ill-conditioned (e.g., when features are highly correlated).

        In our implementation, you only need to control the L1 penalty (lambda), while the L2 penalty (kappa) is set automatically. However, we leave it accessible in case you need to adjust it for your specific use case.

        The objective function minimized in sparse gcPCA is:

        $\hat{\boldsymbol\beta}_j = \arg\min_{\boldsymbol\beta_j} \left\| \mathbf{\Theta}^{1/2} \mathbf{y}_j - \mathbf{\Theta}^{1/2} \boldsymbol\beta_j \right\|^2 + \kappa \left\| \mathbf{J} \mathbf{M}^{-1} \boldsymbol\beta_j \right\|^2 + \lambda \left\| \mathbf{J} \mathbf{M}^{-1} \boldsymbol\beta_j \right\|_1$

        Where:

            - $\lambda$ controls the L1 penalty

            - $\kappa$ controls the L2 penalty

            - $\Theta$, $J$, and $M$ are matrices defined in the gcPCA framework (see full method documentation for details).

        To obtain different sparse solutions, simply vary the value of $\lambda$. A higher $\lambda$ will lead to more sparsity (fewer features retained).

        Important: It is important to verify the penalty is not distorting your results, which could impair interpretability <<< maybe write this in a box like



        A sparse method is tradionally used for feature selection and interpretation. The idea is very simple and help us understand what the models *learned from the data*.

        When performing a regression and/or classification in a dataset we care about the performance of the model and what it learned from the data. The first attempt of this analysis is to look at the magnitude of the weights/loadings of the model and identify the most important features. 

        **I NEED TO MAKE SURE EVERYTHING HERE IS RIGHT, REVIEW THE STATS BOOK - an introduction to statitical learning**
        This become highly intractable with large number of features, which is very common in today's biomedical datasets. A very elegant way to solve this is to add a penalty to the model for large weights, unless the feature is important. This is the basis of what lasso regression and ridge regression do. If a linear regression would usually look like this:

        $ minimize ||Y - X \beta ||$ <<< verify this

        I.e., finding the betas that minimize the Y, then a lasso regression would be: 

        $||Y- X \beta || - \sum \lambda \beta$ <<< verify this

        Here, $\lambda$ is the variable the controls the amount of punishment to the weights in $\beta$. The higher the number in $\lambda$ the higher the punishgment in the weights.

        ##### **Variable in the sparse gcPCA**

        The sparse gcPCA borrows from other sparse PCA approaches in which uses a combination of lasso and ridge regression, called elastic net regression. This means you have two variables to control, but for our case you only have to control for one (the lasso part). The ridge penalty is used to handle ill-conditioned data, and our script takes care of that automatically. However, we leave the ridge part in there if you find necessary for your needs.

        The objective function of sparse gcPCA is:

        $\hat{\boldsymbol\beta}_j = \argmin_{\boldsymbol\beta_j} ||\mathbf{\Theta}^{\frac{1}{2}} \mathbf{y}_j - \mathbf{\Theta}^{\frac{1}{2}} \boldsymbol\beta_j||^2 + \kappa||\mathbf{J} \mathbf{M}^{-1} \boldsymbol\beta_j||^2 + \lambda||\mathbf{J} \mathbf{M}^{-1} \boldsymbol\beta_j||_1$

        The parameter $\kappa$ controls the amount of L2 penalty (ridge), and the parameter $\lambda$ controls the amount of L1 penalty (lasso). To achieve different sparse solution you want to change the latter parameter ($\lambda$)
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Importing required libraries""")
    return


@app.cell
def _():
    # for analysis
    import os
    import numpy as np
    import pandas as pd
    from generalized_contrastive_PCA import sparse_gcPCA
    # for plots
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.colors import ListedColormap
    return ListedColormap, cm, np, os, pd, plt, sparse_gcPCA


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Ancillary functions""")
    return


@app.cell
def _(cm, np, pd, plt):
    def load_and_prepare_data(data_path, metadata_path):
        """
        Load gene expression and metadata, and prepare data matrices for sparse gcPCA.

        Returns:
            gaba_X (np.ndarray): Gene expression matrix for GABAergic cells (samples x genes)
            glut_X (np.ndarray): Gene expression matrix for Glutamatergic cells (samples x genes)
            gaba_subclasses (List[str]): Subclass annotations for GABAergic cells
            glut_subclasses (List[str]): Subclass annotations for Glutamatergic cells
        """
        # Load data and metadata
        data_df = pd.read_csv(data_path, index_col=0)  # genes x samples
        data_df = data_df.transpose()  # samples x genes

        metadata_df = pd.read_csv(metadata_path)

        # Filter metadata to only GABAergic and Glutamatergic
        metadata_df = metadata_df[metadata_df['class'].isin(['GABAergic', 'Glutamatergic'])]

        # Filter metadata to samples present in data
        metadata_df = metadata_df[metadata_df['sample_name'].isin(data_df.index)]

        # Set sample_name as index to align with data_df
        metadata_df = metadata_df.set_index('sample_name')

        # Subset GABAergic and Glutamatergic samples
        gaba_samples = metadata_df[metadata_df['class'] == 'GABAergic'].index
        glut_samples = metadata_df[metadata_df['class'] == 'Glutamatergic'].index

        # Get matrices (samples x genes)
        gaba_X = data_df.loc[gaba_samples].values
        glut_X = data_df.loc[glut_samples].values

        # Get subclass labels in same order
        gaba_subclasses = metadata_df.loc[gaba_samples]['subclass'].values.tolist()
        glut_subclasses = metadata_df.loc[glut_samples]['subclass'].values.tolist()

        return gaba_X, glut_X, gaba_subclasses, glut_subclasses

    def plot_gcPCA_fits(model, subclasses, title_prefix="gcPCA", figsize=(15, 10)):
        """
        Plot 2D gcPCA embeddings for each lambda value.

        Args:
            model: A fitted sparse_gcPCA object.
            subclasses: List of subclass labels corresponding to samples in `gaba_X`.
            title_prefix: Title prefix for each subplot.
            figsize: Size of the whole figure.
        """
        embeddings = model.Zsparse  # list of [samples x components] arrays
        lambdas = model.lasso_penalty
        num_plots = len(embeddings)

        # Get unique subclasses and assign colors
        unique_subclasses = sorted(set(subclasses))
        colors = cm.tab20(np.linspace(0, 1, len(unique_subclasses)))
        subclass_to_color = dict(zip(unique_subclasses, colors))
        subclass_colors = [subclass_to_color[sub] for sub in subclasses]

        # Plot
        fig, axes = plt.subplots(1, num_plots, figsize=figsize, squeeze=False)
        for i, (Z, lam) in enumerate(zip(embeddings, lambdas)):
            ax = axes[0, i]
            ax.scatter(Z[:, 0], Z[:, 1], c=subclass_colors, s=20, alpha=0.7)
            ax.set_title(f"{title_prefix} λ={lam}")
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")
            ax.grid(True)

        # Create one legend for all plots
        handles = [plt.Line2D([0], [0], marker='o', color='w',
                              label=sub, markerfacecolor=color, markersize=6)
                   for sub, color in subclass_to_color.items()]
        fig.legend(handles=handles, title="Subclass", loc='center right')
        plt.tight_layout(rect=[0, 0, 0.95, 1])  # leave space for legend
        plt.show()
    return load_and_prepare_data, plot_gcPCA_fits


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Running sparse gcPCA with multiple $\lambda$ paramaters""")
    return


@app.cell
def _():
    # path of data - Update to your own if running this notebook locally!
    data_path = '/Users/eliezyerdeoliveira/Documents/mouse_VISp_gene_expression_matrices_2018-06-14/mouse_VISp_2018-06-14_exon-matrix.csv'
    metadata_path = '/Users/eliezyerdeoliveira/Documents/mouse_VISp_gene_expression_matrices_2018-06-14/mouse_VISp_2018-06-14_samples-columns.csv'

    # gaba_X, glut_X, gaba_subclasses, glut_subclasses = load_and_prepare_data(data_path=data_path,metadata_path = metadata_path)
    return data_path, metadata_path


@app.cell
def _(sparse_gcPCA):
    # fitting gcPCA

    #parameters
    lambdas_array = [1e-7,1e-5,1e-3,1] # array of different lambdas to fit
    number_of_dimensions = 2

    #initialize the model
    sparse_gcPCA_model = sparse_gcPCA(method='v4',lasso_penalty=lambdas_array, Nsparse=number_of_dimensions)

    # fit
    #sparse_gcPCA_model.fit(gaba_X,glut_X)
    return lambdas_array, number_of_dimensions, sparse_gcPCA_model


@app.cell
def _(gaba_subclasses, plot_gcPCA_fits, sparse_gcPCA_model):
    # plotting the results
    plot_gcPCA_fits(sparse_gcPCA_model, gaba_subclasses)
    return


if __name__ == "__main__":
    app.run()

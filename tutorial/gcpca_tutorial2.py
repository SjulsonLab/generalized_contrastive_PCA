import marimo

__generated_with = "0.12.7"
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

        First, let’s briefly review what a sparse solution is.
        **If you're already familiar with methods with sparse solutions, feel free to skip this section.**

        ##### **What does a sparse solution to a method do?**

        Sparse solutions are commonly used for **feature selection** and **model interpretability**. The idea is simple: we want to identify which features are most relevant to what the model has learned from the data.

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

        After you select an optimal lambda, only a subset of features have non-zero weights, making the model more interpretable.


        ##### **Sparsity in gcPCA**

        The sparse version of gcPCA builds on existing sparse PCA methods by incorporating an elastic net penalty, which combines both L1 (lasso) and L2 (ridge) regularization:

            - L1 (lasso) promotes sparsity by driving some coefficients to zero.

            - L2 (ridge) helps stabilize the solution when the data is ill-conditioned (e.g., when features are highly correlated).

        In our implementation, you only need to control the L1 penalty (lambda), while the L2 penalty (kappa) is set automatically. However, we leave it accessible in case you need to adjust it for your specific use case.

        The objective function minimized in sparse gcPCA is:

        $\hat{\boldsymbol\beta}_j = \arg\min_{\boldsymbol\beta_j} \left\| \mathbf{\Theta}^{1/2} \mathbf{y}_j - \mathbf{\Theta}^{1/2} \boldsymbol\beta_j \right\|^2 + \kappa \left\| \mathbf{J} \mathbf{M}^{-1} \boldsymbol\beta_j \right\|^2 + \lambda \left\| \mathbf{J} \mathbf{M}^{-1} \boldsymbol\beta_j \right\|_1$

        Where:

            - $\lambda$ controls the L1 penalty

            - $\kappa$ controls the L2 penalty

            - $\Theta$, $J$, and $M$ are matrices defined in the gcPCA framework (see full method documentation for details).

        To obtain different sparse solutions, simply vary the value of $\lambda$. A higher $\lambda$ will lead to more sparsity (fewer features retained).

        > **_IMPORTANT:_** You need to verify the penalty is not distorting your original results, which could impair interpretability

        Let's start the tutorial!
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


@app.cell(hide_code=True)
def _(np, pd):
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

        # Remove genes expressed in less than 100 cells
        data_df = filter_genes_by_min_cells(data_df,min_cells=100)
    
        # Select the top 1000 most variable genes
        data_df = select_highly_variable_genes(data_df,top_n=1000)

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

        # Log transforming the data
        gaba_X = np.log(gaba_X+1)
        glut_X = np.log(glut_X+1)

        # Centering the data
        gaba_X = gaba_X - gaba_X.mean(axis=0)
        glut_X = glut_X - glut_X.mean(axis=0)

        return gaba_X, glut_X, gaba_subclasses, glut_subclasses



    def plot_gcPCA_fits(model, subclasses, title_prefix="gcPCA", figsize=(15, 5)):
        """
        Plot 2D gcPCA embeddings and stem plots of top 100 gene loadings for each lambda,
        including the original projection. The first embedding plot includes a legend for the subclasses.
        The y-axis limits for the stem plots are fixed based on the original gcPCA loadings,
        highlighting the shrinking effect of lasso.

        Args:
            model: A fitted sparse_gcPCA object.
            subclasses: List of subclass labels corresponding to the samples.
            title_prefix: Title prefix for each subplot.
            figsize: Size of each row (width, height).
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        # Combine original and penalized embeddings and loadings
        embeddings = [model.original_gcPCA.Ra_scores_] + model.Ra_scores_
        sparse_loadings = [model.original_gcPCA.loadings_] + model.sparse_loadings_
        lambda_labels = ["original"] + [f"λ={lam}" for lam in model.lasso_penalty]
        num_plots = len(embeddings)

        # Get unique subclasses and assign colors
        unique_subclasses = sorted(set(subclasses))
        colors = cm.tab20(np.linspace(0, 1, len(unique_subclasses)))
        subclass_to_color = dict(zip(unique_subclasses, colors))
        subclass_colors = [subclass_to_color[sub] for sub in subclasses]

        # Using the original gcPCA loadings, identify the top 200 genes for both PCs.
        orig_loadings = model.original_gcPCA.loadings_
        top1_idx = np.argsort(-(orig_loadings[:, 0]))[:200]
        top2_idx = np.argsort(-(orig_loadings[:, 1]))[:200]
    
        # Determine fixed y-axis limits based on original loadings for the top genes.
        max_pc1 = np.max(np.abs(orig_loadings[top1_idx, 0]))
        max_pc2 = np.max(np.abs(orig_loadings[top2_idx, 2]))
        y_min_pc1 = -1*max_pc1
        y_max_pc1 = max_pc1
        y_min_pc2 = -1*max_pc2
        y_max_pc2 = max_pc2

        # Create subplots: each row corresponds to one version (original or lasso-penalized)
        fig, axes = plt.subplots(num_plots, 3, figsize=(figsize[0], figsize[1] * num_plots), squeeze=False)

        # Pre-create legend handles so we can add the legend on the first plot
        legend_handles = [plt.Line2D([0], [0], marker='o', color='w',
                                      label=sub, markerfacecolor=color, markersize=6)
                          for sub, color in subclass_to_color.items()]

        for i, (Z, label, L) in enumerate(zip(embeddings, lambda_labels, sparse_loadings)):
            # Embedding plot
            ax0 = axes[i, 0]
            ax0.scatter(Z[:, 0], Z[:, 1], c=subclass_colors, s=20, alpha=0.7)
            ax0.set_title(f"{title_prefix} {label}")
            ax0.set_xlabel("Component 1")
            ax0.set_ylabel("Component 2")
            ax0.grid(True)
            if i == 0:
                # Highlight the first plot and add the subclass legend.
                for spine in ax0.spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(2)
                ax0.legend(handles=legend_handles, title="Subclass", loc='best')

            # Stem plot for PC1 loadings
            ax1 = axes[i, 1]
            pc1_vals = L[top1_idx, 0]
            markerline, stemlines, baseline = ax1.stem(pc1_vals)
            plt.setp(markerline, markersize=4)
            ax1.set_title("Top 200 gcPC1 loadings")
            ax1.set_ylabel("Loading")
            ax1.set_xticks(range(200))
            # Remove gene names for clarity.
            ax1.set_xticklabels([])
            # Set the fixed y-axis limits based on the original loadings.
            ax1.set_ylim(y_min_pc1, y_max_pc1)

            # Stem plot for PC2 loadings
            ax2 = axes[i, 2]
            pc2_vals = L[top2_idx, 1]
            markerline, stemlines, baseline = ax2.stem(pc2_vals)
            plt.setp(markerline, markersize=4)
            ax2.set_title("Top 200 gcPC2 loadings")
            ax2.set_ylabel("Loading")
            ax2.set_xticks(range(200))
            ax2.set_xticklabels([])
            # Set the fixed y-axis limits based on the original loadings.
            ax2.set_ylim(y_min_pc2, y_max_pc2)

        plt.tight_layout() 
        plt.show()



    def filter_genes_by_min_cells(data_df, min_cells=100):
        expressed_counts = (data_df > 0).sum(axis=0)
        selected = expressed_counts >= min_cells
        filtered_df = data_df.loc[:, selected]
        return filtered_df


    def select_highly_variable_genes(data_df, top_n=1000):
        means = data_df.mean(axis=0)
        variances = data_df.var(axis=0)
        dispersion = variances / means
        top_genes = dispersion.nlargest(top_n).index
        filtered_df = data_df[top_genes]
        return filtered_df
    return (
        filter_genes_by_min_cells,
        load_and_prepare_data,
        plot_gcPCA_fits,
        select_highly_variable_genes,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Loading and preparing data

        The data used here is a single-cell RNA sequencing of VISp made available by the Allen Institute for Brain Science, you can download it [**here**](https://portal.brain-map.org/atlases-and-data/rnaseq/mouse-v1-and-alm-smart-seq).

        I am going to contrast the GABAergic cells (condition A) and Glutamatergic cells (condition B) transcriptomics profiles using gcPCA.

        > NOTE:
        If you are running this notebook locally, change the data_path and metadata_path variables to the path where the data is stored on your computer.


        """
    )
    return


@app.cell
def _(load_and_prepare_data):
    # path of data - Update to your own if running this notebook locally!
    data_path = '/home/eliezyer/Documents/mouse_VISp_gene_expression_matrices_2018-06-14/mouse_VISp_2018-06-14_exon-matrix.csv'
    metadata_path = '/home/eliezyer/Documents/mouse_VISp_gene_expression_matrices_2018-06-14/mouse_VISp_2018-06-14_samples-columns.csv'

    gaba_X, glut_X, gaba_subclasses, glut_subclasses = load_and_prepare_data(data_path = data_path, metadata_path = metadata_path)
    return (
        data_path,
        gaba_X,
        gaba_subclasses,
        glut_X,
        glut_subclasses,
        metadata_path,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Fitting the sparse gcPCA model

        The sparse gcPCA model can be run on a _subset of dimensions_ rather than the whole space of all gcPCs. This is recommended because it can speed up the processing time. In this tutorial I picked `number_of_dimensions = 2` to fit the model on only two dimensions, make sure you select the dimensionality appropriate for your case.

        I'm going to use 4 different $\lambda$ values here. The parameter to control the lasso penalty parameter in the model (controlling the $\lambda$ parameter discussed in the introduction) is `lasso_penalty`. For the ridge penalty parameter ($\kappa$), you have to declare `ridge_penalty` in the model initialization.
        """
    )
    return


@app.cell
def _(gaba_X, glut_X, sparse_gcPCA):
    # fitting gcPCA

    #parameters
    lambdas_array = [1e-2,1e-1,1,3] # array of different lambdas to fit
    number_of_dimensions = 2

    #initialize the model
    sparse_gcPCA_model = sparse_gcPCA(method='v4',lasso_penalty=lambdas_array, Nsparse=number_of_dimensions,normalize_flag=True)

    # fit
    sparse_gcPCA_model.fit(gaba_X,glut_X)
    return lambdas_array, number_of_dimensions, sparse_gcPCA_model


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Visualizing the results

        Below you are going to find a five-row and three-column figure.

        In the first row, I show you the plot of the original gcPCA. The first column presents the scores of the gCPCA. You can see the subclasses of interneurons are separated in the top gcPCs, with the first gcPC separating neurons that are Medial Ganglionic Ence (MGE) or Caudal Ganglionic Ence (CGE) derived. The second gcPC further separates the subclasses. The second and third columns show the gcPC1-2 top 200 loadings. 

        In the following rows, I plotted the same results but for the sparse gcPCA of varying lasso penalty ($\lambda$). The more I increase the lambda, the more loadings are shrunk to zero, making the model easier to interpret. On the scores plot in the first column, you can see that $\lambda=1$ value used started distorting the clusters, making them closer together. However, their original structure and separability are still maintained, and the model might be used for further analysis/investigation. Increasing the $\lambda$ further can distort the values in a way that the clusters are overlapping, making the model useless.
        """
    )
    return


@app.cell
def _(gaba_subclasses, plot_gcPCA_fits, sparse_gcPCA_model):
    # plotting the results
    plot_gcPCA_fits(sparse_gcPCA_model, gaba_subclasses)

    return


if __name__ == "__main__":
    app.run()

import marimo

__generated_with = "0.11.30"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    mo.md("# How to use sparse gcPCA")
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Introduction

        In this tutorial, I will show how to use the sparse gcPCA class.
        """
    )
    return


if __name__ == "__main__":
    app.run()

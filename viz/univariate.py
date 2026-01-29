import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class UnivariatePlots:
    @staticmethod
    def plot_cat_vars(df: pd.DataFrame) -> None:
        cols = df.select_dtypes(["object", "int64"]).columns
        n_cols = len(cols)
        nrows = int(np.ceil(n_cols / 2))

        # Create subplots
        fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(12, 5 * nrows))
        axes = axes.flatten()

        for i, col in enumerate(cols):
            sns.countplot(data=df, x=col, ax=axes[i], palette="viridis")
            axes[i].set_title(f"Distribution of {col}", fontsize=14)
            axes[i].set_xlabel("")
            axes[i].set_ylabel("Count")
            axes[i].tick_params(axis="x", rotation=45)

        for j in range(n_cols, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_cont_vars(df: pd.DataFrame) -> None:
        cols = df.select_dtypes(exclude=["object", "category", "int64"]).columns
        nrows = int(np.ceil(len(cols) / 2))
        ncols = 2

        # Create subplots
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 5 * nrows))
        ax = ax.flatten()

        for i, col in enumerate(cols):
            sns.histplot(data=df, x=col, ax=ax[i], palette="viridis", kde=True)
            ax[i].set_title(f"Distribution of {col}", fontsize=14)
            ax[i].set_ylabel("Count")
            ax[i].tick_params(axis="x", rotation=45)

        for j in range(len(cols), len(ax)):
            ax[j].axis("off")

        plt.tight_layout()
        plt.show()

        pass

    @staticmethod
    def plot_outliers(df: pd.DataFrame) -> None:
        cols = df.select_dtypes(exclude=["object", "category", "int64"]).columns
        nrows = int(np.ceil(len(cols) / 2))
        ncols = 2

        # Create subplots
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 5 * nrows))
        ax = ax.flatten()

        for i, col in enumerate(cols):
            sns.boxplot(data=df[col], ax=ax[i], palette="viridis")  # type: ignore
            ax[i].set_title(f"Distribution of {col}", fontsize=14)
            ax[i].set_ylabel("Count")
            ax[i].tick_params(axis="x", rotation=45)

        for j in range(len(cols), len(ax)):
            ax[j].axis("off")

        plt.tight_layout()
        plt.show()

        pass

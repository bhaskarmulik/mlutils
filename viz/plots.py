from typing import Optional
from matplotlib.axes import Axes
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class UnivariatePlots:
    @staticmethod
    def plot_cat_vars(
        df: pd.DataFrame
        ) -> Axes:
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

        return axes

    @staticmethod
    def plot_cont_vars(
        df: pd.DataFrame
        ) -> Axes:
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

        return ax

    @staticmethod
    def plot_outliers(
        df: pd.DataFrame
        ) -> Axes:
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

        return ax


class MultiVariatePlots:

    @staticmethod
    def plot_corr(
        df: pd.DataFrame, 
        target: Optional[str] = None, 
        cmap: str = "viridis", 
        target_cmap: str = "rocket"
    ) -> Axes:

        '''
        Plots a correlation heatmap for numerical columns in a DataFrame.

        Parameters:

        df (pd.DataFrame): The input DataFrame containing numerical columns.
        target (str, optional): The target column to highlight in the heatmap. Defaults to
        None.
        cmap (str, optional): The colormap for the heatmap. Defaults to "viridis".
        target_cmap (str, optional): The colormap for highlighting the target column.
        Defaults to "rocket".

        Returns:
        None: The function displays a heatmap plot.

        '''

        sns.set_theme(style="white", font_scale=0.9)

        corr = df.corr(numeric_only=True)
        mask = np.triu(np.ones_like(corr, dtype=bool))

        plt.figure(figsize=(12, 8))
        ax = sns.heatmap(
            corr,
            mask=mask,
            cmap=cmap,
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            square=True,
            cbar_kws={"shrink": 0.8},
        )

        if target is not None:
            if target not in corr.columns:
                raise ValueError(f"target '{target}' not in columns")

            idx = corr.columns.get_loc(target)
            mask_target = np.ones_like(corr, dtype=bool)
            mask_target[idx, :] = False
            mask_target[:, idx] = False
            mask_target |= np.triu(np.ones_like(corr, dtype=bool), 1)

            sns.heatmap(
                corr,
                mask=mask_target,
                cmap=target_cmap,
                annot=False,
                cbar=False,
                linewidths=0.5,
                square=True,
                ax=ax,
            )

            # Emphasize target labels
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                if label.get_text() == target:
                    label.set_weight("bold")    #type: ignore
                    label.set_color("crimson")

        plt.title("Feature Correlation (upper triangle)")
        plt.tight_layout()
        plt.show()
        return ax


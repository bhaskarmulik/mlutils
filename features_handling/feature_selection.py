from typing import Optional
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
import numpy as np
from scipy.stats import chi2_contingency, kruskal
from itertools import combinations



class TargetCorrelation:
    """
    A class to compute feature interactions and importance.
    """
    
    def compute_correlation(
            self, 
            df: pd.DataFrame, 
            target: str
            ) -> pd.Series:
        
        '''
        Computes correlation of all features with the target variable.
        Returns a Series sorted by absolute correlation values in descending order.
        '''
        corr = df.corr()[target].drop(target)
        return corr.abs().sort_values(ascending=False)

    def compute_mutual_info(
            self, 
            df: pd.DataFrame, 
            target: str, 
            categorical_cols: list[str]
            ) -> pd.Series:
        
        '''
        Computes mutual information between features and the target variable.
        Handles both classification and regression tasks based on the target variable type.
        Returns a Series sorted by mutual information values in descending order.
        
        Most useful for categorical target variables and in non-linear relationships.
        '''
        X = df.drop(columns=[target])
        y = df[target]

        discrete_features = X.columns.isin(categorical_cols)

        if y.dtype.kind in {'O', 'b'} or y.name in categorical_cols:
            mi = mutual_info_classif(
                X, y, discrete_features=discrete_features
            )
        else:
            mi = mutual_info_regression(
                X, y, discrete_features=discrete_features
            )

        return pd.Series(mi, index=X.columns).sort_values(ascending=False)
    
    def cat_target_cramers_v(
            self,
            df: pd.DataFrame,
            target: str,
            categorical_cols: Optional[list[str]] = None,
            return_filtered: bool = False,
            redundancy_threshold: float = 0.3,
            p_value_threshold: float = 0.05,
            bias_correction: bool = True
        ) -> pd.DataFrame:
        """
        Computes Cramér's V between categorical features and a categorical target.

        Returns:
            DataFrame with columns:
            ['cramers_v', 'p_value', 'chi2']
        """

        if categorical_cols is None:
            categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        results = {}

        for col in categorical_cols:
            if col == target:
                continue

            contingency = pd.crosstab(df[col], df[target])

            # Skip degenerate cases
            if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                results[col] = {
                    "cramers_v": np.nan,
                    "p_value": np.nan,
                    "chi2": np.nan
                }
                continue

            chi2, p, _, _ = chi2_contingency(contingency)
            n = contingency.values.sum()
            r, k = contingency.shape

            # Bias-corrected Cramér's V
            if bias_correction:
                phi2 = max(0, chi2 / n - ((k - 1) * (r - 1)) / (n - 1))
                r_corr = r - ((r - 1) ** 2) / (n - 1)
                k_corr = k - ((k - 1) ** 2) / (n - 1)
                v = np.sqrt(phi2 / max(1e-12, min(r_corr - 1, k_corr - 1)))
            else:
                v = np.sqrt((chi2 / n) / min(r - 1, k - 1))

            results[col] = {
                "cramers_v": v,
                "p_value": p,
                "chi2": chi2
            }

        result_df = pd.DataFrame.from_dict(results, orient="index") \
                                .sort_values("cramers_v", ascending=False)

        if return_filtered:
            return result_df.loc[
                (result_df["cramers_v"] > redundancy_threshold) &
                (result_df["p_value"] < p_value_threshold)
            ]

        return result_df
    

    def cat_num_kruskal_wallis(
            self,
            df: pd.DataFrame,
            target: str,
            categorical_cols: Optional[list[str]] = None,
            return_filtered: bool = False,
            effect_size_threshold: float = 0.05,
            p_value_threshold: float = 0.05
        ) -> pd.DataFrame:
        """
        Kruskal-Wallis test for categorical features vs numeric target.
        
        Returns:
            DataFrame with columns:
            ['H_statistic', 'p_value', 'epsilon_squared']
        """

        if categorical_cols is None:
            categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        results = {}

        for col in categorical_cols:
            if col == target:
                continue

            groups = [
                df.loc[df[col] == level, target]
                for level in df[col].dropna().unique()
            ]

            # Need at least 2 non-empty groups
            if len(groups) < 2 or any(len(g) == 0 for g in groups): #type: ignore
                results[col] = {
                    "H_statistic": np.nan,
                    "p_value": np.nan,
                    "epsilon_squared": np.nan
                }
                continue

            H, p = kruskal(*groups)

            n = sum(len(g) for g in groups) #type: ignore
            k = len(groups)

            # Epsilon-squared effect size (recommended for KW)
            epsilon_sq = (H - k + 1) / (n - k) if n > k else np.nan

            results[col] = {
                "H_statistic": float(H),
                "p_value": float(p),
                "epsilon_squared": float(max(0, epsilon_sq))
            }

        result_df = pd.DataFrame.from_dict(results, orient="index") \
                                .sort_values("epsilon_squared", ascending=False)

        if return_filtered:
            return result_df.loc[
                (result_df["epsilon_squared"] >= effect_size_threshold) &
                (result_df["p_value"] < p_value_threshold)
            ]

        return result_df
    
class InFeatureInteraction:
    """
    A class to compute feature interactions using mutual information.
    """

    def _mutual_info_matrix(
            self,
            df : pd.DataFrame
        ) -> pd.DataFrame:

        '''
        Calculates the mutual information matrix for all pairs of features in a DataFrame.
        MI matrices are rarely worth it and often misleading.
        '''
    
        return_df = pd.DataFrame(index=df.columns, columns=df.columns)

        for col in df.columns:

            y = df[col]
            X = df.drop(columns=[col])

            if y.dtype == 'object' or y.dtype.name == 'category':
                mi = mutual_info_classif(X.select_dtypes(exclude=['object', 'category']), y)
            else:
                mi = mutual_info_regression(X.select_dtypes(exclude=['object', 'category']), y)
            mi_series = pd.Series(mi, index=X.select_dtypes(exclude=['object', 'category']).columns)
            return_df[col] = mi_series.sort_values(ascending=False)

        return return_df
    
    def cat_cat_association(
        self,
        df: pd.DataFrame,
        categorical_cols: list[str] | None = None,
        bias_correction: bool = True,
        redundancy_matrix: bool = False,
        redundancy_threshold: Optional[float] = 0.3,
        p_value_threshold: Optional[float] = 0.05
        ) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Computes pairwise Chi-square p-values and Cramér's V for all categorical columns.

        Returns:
            cramers_v_df: symmetric DataFrame of Cramér's V
            p_value_df: symmetric DataFrame of p-values
            redundant (optional): boolean DataFrame indicating redundant feature pairs
        """

        categorical_cols = categorical_cols if categorical_cols is not None else df.select_dtypes(include=["object", "category"]).columns.tolist()

        cramers_v_df = pd.DataFrame(
            0.0, index=categorical_cols, columns=categorical_cols
        )
        p_value_df = pd.DataFrame(
            1.0, index=categorical_cols, columns=categorical_cols
        )

        # 2. Pairwise computation
        for col_x, col_y in combinations(categorical_cols, 2):

            contingency = pd.crosstab(df[col_x], df[col_y])

            # Skip degenerate cases
            if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                continue

            chi2, p, _, _ = chi2_contingency(contingency)
            n = contingency.values.sum()
            r, k = contingency.shape

            # Cramér's V
            if bias_correction:
                phi2 = max(0, chi2 / n - ((k - 1) * (r - 1)) / (n - 1))
                r_corr = r - ((r - 1) ** 2) / (n - 1)
                k_corr = k - ((k - 1) ** 2) / (n - 1)
                v = np.sqrt(phi2 / max(1e-12, min(k_corr - 1, r_corr - 1)))
            else:
                v = np.sqrt((chi2 / n) / min(k - 1, r - 1))

            cramers_v_df.loc[col_x, col_y] = float(v)
            cramers_v_df.loc[col_y, col_x] = float(v)

            p_value_df.loc[col_x, col_y] = float(p) #type: ignore
            p_value_df.loc[col_y, col_x] = float(p) #type: ignore

        for col in categorical_cols:
            cramers_v_df.loc[col, col] = 1.0
            p_value_df.loc[col, col] = 0.0

        if redundancy_matrix:
            redundant = (cramers_v_df > redundancy_threshold) & (p_value_df < p_value_threshold)
            return cramers_v_df, p_value_df, redundant
        else:
            return cramers_v_df, p_value_df
        

    
    
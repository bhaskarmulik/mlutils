import copy
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from typing import Tuple


class target_encoding:

    @staticmethod
    def mean_target_encoding(
            train : pd.DataFrame,
            predict : pd.DataFrame,
            target : str,
            cols : list,
            n_splits=10
            ) -> Tuple[pd.DataFrame, pd.DataFrame]:

        # Work with minimal copies
        train_result = train.copy()
        predict_result = predict.copy()
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        overall_mean = train[target].mean()

        for col in cols:
            # Convert to categorical codes to save memory during processing
            train_codes = train[col].astype('category').cat.codes
            predict_codes = predict[col].astype('category').cat.codes
            
            # Calculate global means using codes (more memory efficient)
            code_to_global_mean = train.groupby(train_codes)[target].mean()
            
            # K-Fold encoding
            mean_encoded = np.full(len(train), overall_mean, dtype=np.float32)
            
            for tr_idx, val_idx in kf.split(train):
                tr_codes = train_codes.iloc[tr_idx]
                val_codes = train_codes.iloc[val_idx]
                tr_target = train[target].iloc[tr_idx]
                
                # Calculate fold means
                fold_means = tr_target.groupby(tr_codes).mean()
                
                # Map validation codes to fold means
                val_means = val_codes.map(fold_means)
                
                # Fill missing with global means
                nan_mask = val_means.isna()
                if nan_mask.any():
                    val_means[nan_mask] = val_codes.map(code_to_global_mean)[nan_mask]
                
                mean_encoded[val_idx] = val_means.fillna(overall_mean).values
            
            train_result[f'mean_{col}'] = mean_encoded
            
            # Test data encoding
            test_encoded = predict_codes.map(code_to_global_mean).fillna(overall_mean)
            predict_result[f'mean_{col}'] = test_encoded.astype(np.float32).values
        
        return train_result, predict_result
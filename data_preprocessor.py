import pandas as pd
import numpy as np

class DataPreprocessor:
    def __init__(self, df):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        self.data = df.copy()
        self.history = {
            'dropped_columns': [],
            'filled_values': {},
            'normalization_params': {},
            'one_hot_columns': []
        }

    def remove_missing(self, threshold=0.5):
        if not (0 <= threshold <= 1):
            raise ValueError("Threshold must be between 0 and 1")

        missing_ratio = self.data.isnull().mean()
        cols_to_drop = missing_ratio[missing_ratio > threshold].index.tolist()
        
        self.data.drop(columns=cols_to_drop, inplace=True)
        self.history['dropped_columns'] = cols_to_drop

        for col in self.data.columns:
            if self.data[col].isnull().sum() > 0:
                if pd.api.types.is_numeric_dtype(self.data[col]):
                    fill_val = self.data[col].mean()
                else:
                    mode_val = self.data[col].mode()
                    fill_val = mode_val[0] if not mode_val.empty else "Unknown"
                
                self.history['filled_values'][col] = fill_val
                self.data[col] = self.data[col].fillna(fill_val)
        
        return self.data

    def encode_categorical(self):
        cat_cols = self.data.select_dtypes(include=['object', 'category']).columns.tolist()
        if not cat_cols:
            return self.data

        self.data = pd.get_dummies(self.data, columns=cat_cols, dummy_na=False, dtype=int)
        
        self.history['one_hot_columns'] = [c for c in self.data.columns if c not in cat_cols]
        return self.data

    def normalize_numeric(self, method='minmax'):
        num_cols = self.data.select_dtypes(include=['number']).columns.tolist()
        
        for col in num_cols:
            
            if method == 'minmax':
                min_val = self.data[col].min()
                max_val = self.data[col].max()
                denom = max_val - min_val
                
                self.history['normalization_params'][col] = {'min': min_val, 'max': max_val}
                
                if denom != 0:
                    self.data[col] = (self.data[col] - min_val) / denom
                else:
                    self.data[col] = 0.0
                    
            elif method == 'std':
                mean_val = self.data[col].mean()
                std_val = self.data[col].std()
                
                self.history['normalization_params'][col] = {'mean': mean_val, 'std': std_val}
                
                if std_val != 0:
                    self.data[col] = (self.data[col] - mean_val) / std_val
                else:
                    self.data[col] = 0.0
            else:
                raise ValueError("Method must be 'minmax' or 'std'")
                
        return self.data

    def fit_transform(self, threshold=0.5, method='minmax'):
        self.remove_missing(threshold)
        self.encode_categorical()
        self.normalize_numeric(method)
        return self.data
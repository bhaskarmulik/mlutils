from typing import Annotated, Literal
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder
import copy
import numpy as np
import pandas as pd

class _PyTorchEmbeddingModel(nn.Module):
    def __init__(self, embedding_configs, n_numerical, hidden_units=100):
        super(_PyTorchEmbeddingModel, self).__init__()
        self.embeddings = nn.ModuleList()
        total_embed_dim = 0
        
        for n_cat, embed_dim in embedding_configs:
            # n_cat already includes +1 for unknown padding from the Generator
            self.embeddings.append(nn.Embedding(n_cat, embed_dim))
            total_embed_dim += embed_dim
            
        self.num_bn = nn.BatchNorm1d(n_numerical) if n_numerical > 0 else None
        
        # Dense Layers
        input_dim = total_embed_dim + n_numerical
        self.fc1 = nn.Linear(input_dim, hidden_units)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.relu2 = nn.ReLU()
        self.output = nn.Linear(hidden_units, 1) 
        
    def forward(self, x_num, x_cat):
        embedded = []
        for i, emb_layer in enumerate(self.embeddings):
            col_data = x_cat[:, i].long()
            embed_out = emb_layer(col_data)
            embedded.append(embed_out)
            
        if embedded:
            cat_out = torch.cat(embedded, dim=1)
        else:
            cat_out = torch.tensor([]).to(x_num.device)

        if self.num_bn is not None and x_num.shape[1] > 0:
            num_out = self.num_bn(x_num)
        else:
            num_out = x_num
            
        if embedded and x_num.shape[1] > 0:
            x = torch.cat([num_out, cat_out], dim=1)
        elif embedded:
            x = cat_out
        else:
            x = num_out
            
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.output(x)

class EmbeddingGenerator(BaseEstimator, TransformerMixin):
    def __init__(self, epochs=5, batch_size=128, embedding_size=None, 
                 learning_rate=0.001, loss_fn="mse", target_obj="regression"):
        self.epochs = epochs
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate
        self.loss_fn = loss_fn
        self.target_obj = target_obj
        
        self.model = None
        self.categorical_columns = []
        self.numerical_columns = []
        self.category_maps = {} # Stores {col: {category: index}}
        self.embedding_dims = {}
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # We handle encoding manually to strictly control the Unknown Index
        self.oe = OrdinalEncoder(
            handle_unknown='use_encoded_value', 
            unknown_value=-1,
            dtype=np.int32
        )

    def fit(self, X, y):
        # 1. Identify Columns
        self.categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
        self.numerical_columns = X.select_dtypes(exclude=['object', 'category']).columns.tolist()
        
        # 2. Fit Encoder & Prepare Categorical Data
        if self.categorical_columns:
            self.oe.fit(X[self.categorical_columns])
            
            # Extract categories to create a reliable map
            for i, col in enumerate(self.categorical_columns):
                cats = self.oe.categories_[i]
                # Map category -> int. 
                # We shift everything by +1, leaving 0 for "Unknown"
                self.category_maps[col] = {cat: idx + 1 for idx, cat in enumerate(cats)} #type: ignore
        
        # 3. Process Data for Training
        X_cat_tensor, X_num_tensor = self._prepare_tensors(X)
        y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1).to(self.device)
        
        dataset = TensorDataset(X_num_tensor, X_cat_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # 4. Create Model
        self._create_model()
        if self.model is not None:
            self.model.to(self.device)
        else:
            raise AttributeError("Model assigned to None. Please check model initialization")
        
        # 5. Train
        criterion = nn.L1Loss() if self.loss_fn == "mae" else nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate) #type: ignore
        
        self.model.train() #type: ignore
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_num, batch_cat, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_num, batch_cat) #type: ignore
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            # Optional: print(f"Epoch {epoch+1} Loss: {epoch_loss/len(dataloader)}")
            
        return self

    def transform(self, X):
        
        if self.model:
            self.model.eval() 
        X_copy = X.copy()
        
        # We iterate over stored columns to ensure consistency
        for idx, col in enumerate(self.categorical_columns):
            if col not in X_copy.columns:
                raise ValueError(f"Column {col} missing from input data")

            # Get Learned Embeddings (CPU numpy)
            # The weight matrix is size (N+1, dim)
            emb_matrix = self.model.embeddings[idx].weight.data.cpu().numpy() #type: ignore
            
            # Create a dictionary map: {Category_Name: Embedding_Vector}
            # 1. Start with the Known categories
            cat_to_vec = {}
            for cat, vocab_idx in self.category_maps[col].items():
                cat_to_vec[cat] = emb_matrix[vocab_idx] # vocab_idx is already shifted by +1
            
            # 2. Handle the "Unknown" case (Index 0)
            unknown_vec = emb_matrix[0] 
            
            # 3. Apply map to the column
            # This is much faster and safer than merge
            def map_val(val):
                return cat_to_vec.get(val, unknown_vec)
            
            # Apply transformation
            # Result is a Series of numpy arrays
            vectors = X_copy[col].map(map_val)
            
            # Expand into columns
            # This handles the creation of new columns efficiently
            col_names = [f"{col}_embed_{i}" for i in range(emb_matrix.shape[1])]
            embed_df = pd.DataFrame(vectors.tolist(), index=X_copy.index, columns=col_names)
            
            # Concat and Drop original
            X_copy = pd.concat([X_copy, embed_df], axis=1)
            # Optional: Drop original categorical column
            # X_copy.drop(columns=[col], inplace=True) 

        return X_copy

    def _prepare_tensors(self, X):
        # Handle Categorical
        if self.categorical_columns:
            # Transform returns 0..N-1, with -1 for unknowns
            encoded = self.oe.transform(X[self.categorical_columns])
            # Shift everything by +1. Now: Unknown=0, Cats=1..N
            encoded = encoded + 1 
            X_cat_tensor = torch.tensor(encoded, dtype=torch.long).to(self.device)
        else:
            X_cat_tensor = torch.tensor([], device=self.device)
            
        # Handle Numerical
        if self.numerical_columns:
            X_num_tensor = torch.tensor(X[self.numerical_columns].values, dtype=torch.float32).to(self.device)
        else:
            X_num_tensor = torch.tensor([], device=self.device)
            
        return X_cat_tensor, X_num_tensor

    def _create_model(self):
        embedding_configs = []
        for col in self.categorical_columns:
            # number of unique categories + 1 for Unknown
            n_cat = len(self.category_maps[col]) + 1 
            
            if self.embedding_size:
                dim = self.embedding_size
            else:
                dim = min(50, int(np.ceil((n_cat)/2)))
            
            embedding_configs.append((n_cat, dim))
            
        self.model = _PyTorchEmbeddingModel(
            embedding_configs=embedding_configs,
            n_numerical=len(self.numerical_columns)
        )
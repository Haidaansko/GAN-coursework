from itertools import product
import numpy as np
import pandas as pd


class Model:
    def fit(self, X, Y, n_bins=5):
        self.means = {}
        self.stds = {}
        for col in Y.columns:
            self.means[col] = np.zeros((n_bins, n_bins, n_bins))
            self.stds[col] = np.zeros((n_bins, n_bins, n_bins))
        self.bins = {}
        self.masks = self.make_masks(X, n_bins)
        
        for i, j, k in product(range(n_bins), range(n_bins), range(n_bins)):
            mask = self.masks['TrackP'][:, i] & \
                self.masks['TrackEta'][:, j] & \
                self.masks['NumLongTracks'][:, k]
            for col in Y.columns:
                self.means[col][i, j, k] = np.mean(Y[col][mask])
                self.stds [col][i, j, k] = np.std (Y[col][mask])
        

    def predict(self, X):
        prediction = pd.DataFrame()
        count = np.zeros((self.means['RichDLLk'].shape), dtype=int)
        pred_masks = {}
        n_bins = count.shape[0]
        
        pred_masks = self.make_masks()
        
        for i, j, k in product(range(n_bins), range(n_bins), range(n_bins)):
            count[i,j,k] = np.count_nonzero(
                pred_masks['TrackP'][:, i] & \
                pred_masks['TrackEta'][:, j] & \
                pred_masks['NumLongTracks'][:, k]
            )
        
        
        for col in self.means.keys():
            gaussian = np.array([])
            for i, j, k in product(
                range(n_bins), range(n_bins), range(n_bins)):
                gaussian = np.append(
                    gaussian, 
                    np.random.normal(
                        loc=self.means[col][i, j, k], 
                        scale=self.stds[col][i, j, k], 
                        size=count[i, j, k]))
                        
            prediction[col] = gaussian
        return prediction
    
    
    def make_masks(self, X, n_bins):
        masks = {}
        for col in X.columns:
            self.bins[col] = np.quantile(X[col], np.linspace(
                1 / n_bins, 1 - 1 / n_bins, n_bins - 1))
            left_masks = X[col][:, None] <= self.bins[col]
            right_masks = ~left_masks
            masks[col] = np.zeros((X[col].size, n_bins), dtype=bool)
            masks[col][:, 0] = left_masks[:, 0]
            for i in range(1, n_bins - 1):
                masks[col][:, i] = np.logical_and(
                    left_masks[:, i], right_masks[:, i - 1])
            masks[col][:, -1] = right_masks[:, -1]
        return masks

import warnings
from scipy.sparse.linalg import eigsh
from scipy.sparse import eye
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.metrics import pairwise_kernels
from sklearn.neighbors import NearestCentroid
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.preprocessing import OneHotEncoder
import numpy as np


class MDSFDA(BaseEstimator, ClassifierMixin, TransformerMixin):
    """
    MDS-FDA algorithm for non-euclidean data embedding. 
    This implementation is utilized the implementation of kernel fisher discriminant analysis
    (https://github.com/concavegit/kfda).
    
    Parameters
    ----------
    n_components : int the dimensionality for embedding.
        This is limited by the amount of classes minus one (i.e., c-1).
    robustness_offset : float
        The small value to add along the diagonal of N to gurantee
        valid fisher directions (i.e., regularization).
        Set this to 0 to disable the feature. Default: 1e-8.       
    **kwds : parameters to pass to the kernel function.
    
    Attributes
    ----------
    classes_ : array of shape (n_classes,)
        The unique class labels
    embedding_ : array of shape (n_samples, n_components) 
        Fisher discriminant embedding
    eigvec_ : array of shape (n_components, n_samples) that
        represent the fisher components.
    """
    
    
    def __init__(self, n_components=2, robustness_offset=1e-6, **kwds):        
        self.n_components = n_components
        self.kwds = kwds
        self.robustness_offset = robustness_offset

        
    
    def fit(self, D, y):
        """
        Fit the NearestCentroid model according to the given training data.
        
        Parameters
        ----------
        D : {array-like} of shape (n_samples, n_samples)
            Precomputed pairwise distance matrix.
        y : array, shape = [n_samples]
            Target values (integers)
        """
        self.classes_ = unique_labels(y)
        if self.n_components > self.classes_.size - 1:
            warnings.warn(
                "n_components > classes_.size - 1."
                "Only the first classes_.size - 1 components will be valid."
            )
        
        # Obtain the gram matrix from pairwise distances
        n = D.shape[0]
        D = D @ D
    
        C = np.eye(n) - np.ones((n,n)) / n 
        G = -0.5 * (C @ D) @ C
        
        self.K_ = G
        self.y_ = y

        y_onehot = OneHotEncoder().fit_transform(
            self.y_[:, np.newaxis])

        
        m_classes = y_onehot.T @ self.K_ / y_onehot.T.sum(1)
        indices = (y_onehot @ np.arange(self.classes_.size)).astype('i')
        
        
        # Set denominator of the objective function J
        N = self.K_  @ self.K_ 
        N += eye(self.y_.size) * self.robustness_offset
        
        # Set numerator
        m_classes_centered = m_classes - self.K_.mean(1)
        M = m_classes_centered.T @ m_classes_centered

        # Find the eigenvectors 
        w, self.eigvec_ = eigsh(M, self.n_components, N, which='LM')
        
        # Find the embedding
        self.embedding_ = self.K_ @ self.eigvec_
        
        return self
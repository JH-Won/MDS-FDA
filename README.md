# FDA-MDS: Extending Fisher-Discriminant Analysis for Non-vector Data

This repository provides the code of MDS-FDA, an algorithm for obtaining low-dimensional embedding of non-vector, labled data. The algorithm is a non-vector version of Fisher-Discriminant Analaysis (FDA), exploiting the information from Multidimensional scaling (MDS). For a detailed description, take a look at [the short article of MDS-FDA]().  

## Problem

We consider the set $\mathcal{Q} = {(s_i, y_i)}_{i=1}^{N}$, a set of non-euclidean data $s_i$ and their classes $y_i = 1,2,\cdots, c$. 

We also assume that we know the pairwise distance matrix $D \in\mathbb{R}_+^{N \times N}$ of the data, where

$$D_{ij} = d(s_i, s_j)$$ 

with some metric $d(\cdot, \cdot) \in \mathbb{R}_+$. 

Using this information, the goal is to find the linear embedding $U \in\mathbb{R}^{N \times k} (k \le c-1) $, maximizing between-class covariance and minimizing within-class covariance of $s_i$ (as what FDA does).


## Algorithm and the solution
The objective function we consider is:

$$\mathcal{J}(V) = \max_{V} \frac{tr(V^\top M V)}{tr(V^\top G^2 V)},$$

where $G\in\mathbb{R}^{N \times N}$ is the gram matrix obtained from MDS of $\mathcal{Q}$ (using $D$):

$$G = -\frac{1}{2} C D^2 C,$$ 

with a centering matrix $C,$ and $M \in\mathbb{R}^{N \times N}$ is a partial matrix of $G$ defined as:

$$\sum_{i=1}^c N_i  K^{(i)} K^{{(i)}\top},$$ 

where $K^{(i)} \in\mathbb{R}^{N \times 1}$ with $$K^{(i)}_l = \sum_k^{N_j} \mathbf{x}_l^\top \mathbf{x}_k.$$ $x_l$ is a vector representation of $s_l$, so the inner product $\mathbf{x}_l^\top \mathbf{x}_k$ represents a similarity between $s_l$ and $s_k$.


### The closed-form solution:
The MDS-FDA has the closed-form solution obtained by solving the following eigenvalue problem (for the derivation, see [the article of MDS-FDA]()):

$$ (G^2)^{-1} M V = V \Lambda,$$

with an diagonal-eigenvalue matrix $\Lambda \in \mathbb{R}^{k \times k}$. We obtain the optimal $V^*\in\mathbb{R}^{N \times k}$ by picking the $k$ eigenvectors corresponding to the largest $k$ eigenvalues in $\Lambda$.


### Obtaining linear embedding: 
With the obtained optimal projection matrix $V^*\in\mathbb{R}^{N \times k}$, we can find the linear embedding of $\mathcal{Q}$ as:

$$U = GV^* = -\frac{1}{2} C D^2 C V^*.$$


## Code

The MDS-FDA algorithm is implemented as the class `MDSFDA` in the `MDSFDA.py`. The class `MDSFDA` is instantiated with the dimensionality of embedding $U$: `n_components` and the regularization `robustness_offset` for the gram matrix $G$ (that is, $G \leftarrow G + \gamma I$):
```python
mdsfda = MDSFDA(n_components=2, robustness_offset=1e-6)
```
Then, we fit the instance using $N \times N$ parwise distance matrix $D$ and the label vector $\mathbf{y}$, and obtain the embedding using the following codes:
```python
mdsfda.fit(D, y)
U = mdsfda.embedding_
```
The shape of the embedding $U$ will be $N \times k, k \le c-1$ (in the above example, $k=2$).


#### Notebook tutorial

You can run the code in the `Example_protein_sequences.ipynb`. This example use the protein sequence to show the MDS-FDA embedding. Contact jonghyun1@hanyang.ac.kr if you have any inquiries, quetions, or suggestions.

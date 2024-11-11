import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold._utils import _binary_search_perplexity
from sklearn.manifold import _barnes_hut_tsne

# Optimized processes (not discussed)
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads #Number of threads

class TSNE():
    
    def __init__(
        self,
        n_components = 2,
        perplexity = 30.0
    ):
        # Control the number of exploration iterations with early_exaggeration on
        self._EXPLORATION_N_ITER = 250
        # Control the number of iterations between progress checks
        self._N_ITER_CHECK = 50

        self.n_components = n_components
        self.perplexity = perplexity
        self.early_exaggeration = 12.0
        self.n_iter = 1000 # number of iteration
        self.it = 0 # Current iteration number

        self.n_samples = None
        self.learning_rate_ = None
        self.n_neighbors = None
        self.degrees_of_freedom = None
        self.num_threads = None

    
    def initialize_params(self, X):

        # Define number of samples
        self.n_samples = X.shape[0]
        # Always learning_rate_ = 'auto'
        self.learning_rate_ = np.maximum(self.n_samples / self.early_exaggeration / 4, 50)
        # Compute the number of nearest neighbors to find.
        self.n_neighbors = min(self.n_samples - 1, int(3.0 * self.perplexity + 1))
        # Degree of freedom in gradient descent
        self.degrees_of_freedom = max(self.n_components - 1, 1)
        # Define the number of available threads
        self.num_threads = _openmp_effective_n_threads()
        # Numerical precision
        self.MACHINE_EPSILON = np.finfo(np.double).eps


    def random_init(self):
        
        # Random initialization of low-dimensional representation Y
        random_state = np.random.RandomState(42)
        X_embedded = 1e-4 * random_state.standard_normal( 
                    size = (self.n_samples, self.n_components)
                ).astype(np.float32)
        params = X_embedded.ravel()

        return params


    def distances_knn(self, X : np.array):

        # Find the nearest neighbors for every point
        knn = NearestNeighbors(n_neighbors = self.n_neighbors, metric = "euclidean")
        knn.fit(X)
        distances_nn = knn.kneighbors_graph(mode = "distance")
        del knn # Free memory
        distances_nn.data **= 2

        distances_nn.sort_indices()
        n_samples = distances_nn.shape[0]
        distances_data = distances_nn.data.reshape(n_samples, -1)
        distances_data = distances_data.astype(np.float32, copy=False)

        return distances_data, distances_nn.indices, distances_nn.indptr
    

    def _joint_probabilities_nn(
            self,
            distances : np.array, 
            indices : np.array, 
            indptr : np.array
        ) -> np.array:

        # Binary search and conditional probability evaluation (in Cython)
        conditional_P = _binary_search_perplexity(distances, self.perplexity, verbose = 0)

        # Symmetrize the joint probability distribution using sparse operations
        P = csr_matrix(
            (conditional_P.ravel(), indices, indptr),
            shape = (self.n_samples, self.n_samples),
        )
        P = P + P.T

        # Normalize the joint probability distribution
        sum_P = np.maximum(P.sum(), self.MACHINE_EPSILON)
        P /= sum_P

        return P



    def _kl_divergence_bh(
        self,
        Y : np.array,
        P # compressed sparse matrix
    ):

        Y = Y.astype(np.float32, copy=False)
        Y_embedded = Y.reshape(self.n_samples, self.n_components)

        val_P = P.data.astype(np.float32, copy=False)
        neighbors = P.indices.astype(np.int64, copy=False)
        indptr = P.indptr.astype(np.int64, copy=False)

        grad = np.zeros(Y_embedded.shape, dtype=np.float32)
        error = _barnes_hut_tsne.gradient(
            val_P = val_P,
            pos_output = Y_embedded,
            neighbors = neighbors,
            indptr = indptr,
            forces = grad,
            theta = 0.5,
            n_dimensions = self.n_components,
            verbose = 0,
            dof = self.degrees_of_freedom,
            num_threads = self.num_threads,
        )

        c = 2.0 * (self.degrees_of_freedom + 1.0) / self.degrees_of_freedom
        grad = grad.ravel()
        grad *= c

        return error, grad



    def _gradient_descent(
        self,
        Y0 : np.array,
        P , # compressed sparse matrix
        momentum : float = 0.8
    ):

        Y = Y0.copy().ravel()
        update = np.zeros_like(Y)
        gains = np.ones_like(Y)
        error = np.finfo(float).max
        best_error = np.finfo(float).max
        best_iter = i = self.it

        for i in range(self.it, self.n_iter):

            error, grad = self._kl_divergence_bh(Y = Y, P = P)

            inc = update * grad < 0.0
            dec = np.invert(inc)
            gains[inc] += 0.2
            gains[dec] *= 0.8
            np.clip(gains, 0.01, np.inf, out=gains) # min_gain = 0.01
            grad *= gains
            update = momentum * update - self.learning_rate_ * grad
            Y += update

            grad_norm = np.linalg.norm(grad)

            # Check progress of gradient descent
            if error < best_error:
                best_error = error
                best_iter = i
            elif i - best_iter > 300: # n_iter_without_progress
                break
            if grad_norm <= 1e-7: # min_grad_norm
                break

        return Y, error, i


    def fit_transform(self, X : np.array) -> np.array:

        # Initialize optimization parameters
        self.initialize_params(X)

        # Compute the distances
        distances_nn, indices, indptr = self.distances_knn(X = X)
        
        # compute the joint probability distribution for the input space
        P = self._joint_probabilities_nn(
                                    distances = distances_nn, 
                                    indices = indices,
                                    indptr = indptr
                                )

        # Random initialization of low-dimensional representation Y
        Y = self.random_init()

        # Learning schedule (part 1): do 250 iteration with lower momentum but
        # higher learning rate controlled via the early exaggeration parameter
        P *= self.early_exaggeration
        Y, self.kl_divergence_, it = self._gradient_descent(
                                            Y0 = Y, 
                                            P = P,
                                            momentum = 0.5
                                    )

        # Learning schedule (part 2): disable early exaggeration and finish
        # optimization with a higher momentum at 0.8
        P /= self.early_exaggeration
        remaining = self.n_iter - self._EXPLORATION_N_ITER
        if self.it < self._EXPLORATION_N_ITER or remaining > 0:
            self.it = self.it + 1
            Y, self.kl_divergence_, self.it = self._gradient_descent(
                                            Y0 = Y, 
                                            P = P, 
                                            momentum = 0.8
                                        )

        return Y.reshape(self.n_samples, self.n_components)
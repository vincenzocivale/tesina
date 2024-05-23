from sklearn.decomposition import PCA, TruncatedSVD, NMF
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.decomposition import FactorAnalysis, FastICA
from keras.layers import Input, Dense
from keras.models import Model


def benchmark_dimensionality_reduction(X, metric='recostruction_error', n_components=2):
    """
    Perform dimensionality reduction techniques and evaluate the performance of a given model.

    Parameters:
    - X: Input features
    - y: Target variable
    - model: Model to evaluate
    - n_components: Number of components for dimensionality reduction

    Returns:
    - results: Dictionary containing the evaluation results for each dimensionality reduction technique
    """

    results = {}

    if metric == 'recostruction_error':
        # PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        X_pca_reconstructed = pca.inverse_transform(X_pca)
        results['PCA'] = calculate_reconstruction_error(X, X_pca_reconstructed)

        # SVD
        svd = TruncatedSVD(n_components=n_components)
        X_svd = svd.fit_transform(X)
        X_svd_reconstructed = svd.inverse_transform(X_svd)
        results['SVD'] = calculate_reconstruction_error(X, X_svd_reconstructed)

        # NMF
        nmf = NMF(n_components=n_components, init='random', random_state=42)
        X_nmf = nmf.fit_transform(X)
        X_nmf_reconstructed = np.dot(X_nmf, nmf.components_)
        results['NMF'] = calculate_reconstruction_error(X, X_nmf_reconstructed)

        # Autoencoder
        input_layer = Input(shape=(X.shape[1],))
        encoded = Dense(n_components, activation='relu')(input_layer)
        decoded = Dense(X.shape[1], activation='sigmoid')(encoded)
        autoencoder = Model(input_layer, decoded)
        autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
        autoencoder.fit(X, X, epochs=50, batch_size=256, shuffle=True, verbose=0)
        X_encoded = autoencoder.predict(X)
        results['Autoencoder'] = calculate_reconstruction_error(X, X_encoded)

        # Factor Analysis
        fa = FactorAnalysis(n_components=n_components)
        X_fa = fa.fit_transform(X)
        X_fa_reconstructed = fa.inverse_transform(X_fa)
        results['FA'] = calculate_reconstruction_error(X, X_fa_reconstructed)

        # Independent Component Analysis
        ica = FastICA(n_components=n_components)
        X_ica = ica.fit_transform(X)
        X_ica_reconstructed = ica.inverse_transform(X_ica)
        results['ICA'] = calculate_reconstruction_error(X, X_ica_reconstructed)

    return results


def reduce_dimensionality(X_train, X_test, reduction_technique, n_components=2):
    """
    Perform dimensionality reduction on the input data using the specified technique.

    Args:
    - X_train: Training input features
    - X_test: Test input features
    - reduction_technique: The dimensionality reduction technique to use ('PCA', 'Isomap', 't-SNE', 'Autoencoder')
    - n_components: Number of components for dimensionality reduction

    Returns:
    - Tuple containing the reduced training and test input features
    """

    if reduction_technique == 'PCA':
        reducer = PCA(n_components=n_components)
    elif reduction_technique == 'SVD':
        reducer = TruncatedSVD(n_components=n_components)
    elif reduction_technique == 'NMF':
        reducer = NMF(n_components=n_components, init='random', random_state=42)
    elif reduction_technique == 'Autoencoder':
        input_layer = Input(shape=(X_train.shape[1],))
        encoded = Dense(n_components, activation='relu')(input_layer)
        decoded = Dense(X_train.shape[1], activation='sigmoid')(encoded)
        autoencoder = Model(input_layer, decoded)
        autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
        autoencoder.fit(X_train, X_train, epochs=50, batch_size=256, shuffle=True, verbose=0)
        reducer = Model(input_layer, encoded)
    elif reduction_technique == 'FA':
        reducer = FactorAnalysis(n_components=n_components)
    elif reduction_technique == 'ICA':
        reducer = FastICA(n_components=n_components)
    reducer.fit(X_train)
    X_train_reduced = reducer.transform(X_train)
    X_test_reduced = reducer.transform(X_test)

    return X_train_reduced, X_test_reduced


def calculate_reconstruction_error(X, X_reconstructed):
    return mean_squared_error(X, X_reconstructed)

dimensionality_red_dict = {
    'pca': PCA(n_components=2)
}

import matplotlib.pyplot as plt
from statistics import mean
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_curve, auc


def plot_pvalues(df, title, p_value_threshold=0.05, save=False):
    """
    Crea un grafico a barre orizzontale per rappresentare i valori p-value corrispondenti a ciascuna feature.
    
    Args:
    - df: DataFrame pandas contenente le colonne 'Feature' e 'p-value'.
    """

    # Creazione del grafico
    plt.figure(figsize=(10, 6))
    plt.barh(df['Feature'], df['P-value'], color='skyblue')
    
    # Aggiunta della retta tratteggiata per il p-value
    plt.axvline(x=p_value_threshold, color='red', linestyle='--')
    
    # Titoli e etichette
    plt.xlabel('P-value')
    plt.ylabel('Feature')
    plt.title(title)
    
    # Visualizzazione della griglia
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    if save:
        plt.savefig(f'../reports/figures/{title}.png')
    
    # Mostra il grafico
    plt.show()

def plot_isto_features(df, column_list):
    """
    Crea un grafico per ciascuna colonna nella lista e le ordina in verticale.
    
    Args:
    - df: DataFrame pandas contenente i dati.
    - column_list: Lista di nomi di colonne da visualizzare.
    """
    num_plots = len(column_list)
    fig, axes = plt.subplots(nrows=num_plots, ncols=1, figsize=(8, 6*num_plots), sharex=True)

    for i, column in enumerate(column_list):
        ax = axes[i] if num_plots > 1 else axes
        ax.bar(df.index, df[column], color='skyblue')
        ax.set_title(column)
        ax.set_ylabel('Value')
        ax.set_xlabel('Index')
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

def cross_validation_scores(results, maximize=True):
    # check for no results
    if len(results) == 0:
        print('no results')
        return
    # create a list of (name, mean(scores)) tuples
    mean_scores = [(k, mean(v)) for k, v in results.items()]
    # sort tuples by mean score
    mean_scores = sorted(mean_scores, key=lambda x: x[1])
    # reverse for descending order (e.g., for accuracy)
    if maximize:
        mean_scores = list(reversed(mean_scores))
    # retrieve the top n for summarization
    names = [x[0] for x in mean_scores]
    scores = [mean(results[name]) for name in names]

    bars = plt.bar(names, scores)

    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.grid(False)



def plot_model_performance(model, X_test, y_test, metric='accuracy', title=None):
    """
    Generate a plot showing the performance of the trained model.

    Args:
        model: Trained machine learning model.
        X_test (array-like): Test features.
        y_test (array-like): True labels of the test data.
        metric (str): Evaluation metric to be used (e.g., 'accuracy', 'precision', 'recall', 'f1-score').
        title (str or None): Title for the plot.

    Returns:
        None
    """
    if metric not in ['accuracy', 'precision', 'recall', 'f1-score']:
        print("Invalid metric. Please choose one of: 'accuracy', 'precision', 'recall', 'f1-score'")
        return

    # Predict labels for the test data
    y_pred = model.predict(X_test)

    # Calcola il punteggio della metrica
    if metric == 'accuracy':
       score = model.score(X_test, y_test)
    else:
        score = classification_report(y_test, y_pred, output_dict=True)['macro avg'][metric]
    
    # Plot confusion matrix
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(ax=axes[0])
    axes[0].set_title('Confusion Matrix')
    axes[0].text(0.5, -0.15, f'{metric.capitalize()}: {score:.2f}', ha='center', transform=axes[0].transAxes, fontsize=12)

    # ROC curve
    y_test_bin = (y_test == y_test.iloc[0]).astype(int)  # Converti le etichette in binarie
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test_bin, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    axes[1].plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('Receiver Operating Characteristic')
    axes[1].legend(loc="lower right")

    if title:
        plt.suptitle(title)
        
    plt.tight_layout()
    plt.show()




def plot_class_distribution(y, title='Class Distribution'):
    """
    Plot the class distribution of a target variable.

    Args:
        y (Series or array-like): Target variable.

    Returns:
        None
    """
    class_counts = y.value_counts()
    class_labels = class_counts.index
    class_values = class_counts.values

    plt.figure(figsize=(8, 6))
    plt.bar(class_labels, class_values, color='skyblue')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title(title)
    plt.show()

def boxplot_non_categorical_features(df, title):
    """
    Create a single boxplot for all non-categorical features in the DataFrame.

    Args:
        df (DataFrame): The DataFrame containing the features.
        title (str): The title for the boxplot.

    Returns:
        None
    """
    # Seleziona solo le colonne non categoriche
    non_categorical_columns = df.select_dtypes(exclude=['object']).columns.tolist()

    # Disegna il boxplot per le colonne non categoriche
    plt.figure(figsize=(12, 6))
    df[non_categorical_columns].boxplot()
    plt.title(title)
    plt.xticks(rotation=45, ha='right')  # Ruota le etichette sull'asse x per una migliore leggibilità
    plt.tight_layout()  # Imposta il layout in modo che i titoli degli assi non si sovrappongano
    plt.show()


def benchmark_dim_red_result_plot(results):
    """
    Create a bar chart to compare the reconstruction errors of different dimensionality reduction techniques.

    Parameters:
    - results: (return di benchmark_dimensionality_reduction) Dictionary containing the evaluation results for each dimensionality reduction technique.
    """
    
    techniques = list(results.keys())
    errors = list(results.values())
    
    plt.figure(figsize=(10, 6))
    plt.bar(techniques, errors, color='skyblue')
    plt.xlabel('Dimensionality Reduction Technique')
    plt.ylabel('Reconstruction Error')
    plt.title('Comparison of Reconstruction Errors for Dimensionality Reduction Techniques')
    plt.xticks(rotation=45)
    plt.ylim(0, max(errors) * 1.1)  # Add some space above the highest bar for better visualization
    
    for i, v in enumerate(errors):
        plt.text(i, v + max(errors) * 0.02, f"{v:.4f}", ha='center', va='bottom')

    plt.tight_layout()
    plt.show()


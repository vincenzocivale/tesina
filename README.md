# Tesina_modelli

## Descrizione
Il progetto si concentra sull'analisi di un dataset contenente dati di pazienti appartenenti a quattro gruppi distinti: persone sane, individui affetti da sepsi, pazienti con COVID-19 e soggetti con disturbi mentali. L'obiettivo è addestrare un classificatore automatico tramite tecniche di Machine Learning (ML) per identificare correttamente i gruppi di appartenenza.

## Organizzazione del Progetto

├── LICENSE
├── Makefile           <- Makefile con comandi come `make data` o `make train`
├── README.md          <- Il README principale per sviluppatori che utilizzano questo progetto.
├── data
│   ├── interim        <- Dati intermedi che sono stati trasformati.
│   ├── processed      <- I set di dati finali e canonici per la modellazione.
│   └── raw            <- Il dump originale e immutabile dei dati.
│
├── models             <- Modelli addestrati e serializzati, previsioni del modello o riepiloghi del modello
│
├── notebooks          <- Notebook Jupyter in cui si mostra il workflow dell'analisi (main del progetto)
├── references         <- Dizionari dei dati, manuali e tutti gli altri materiali esplicativi.
│
├── reports            <- Analisi generate in HTML, PDF, LaTeX, ecc.
│   └── figures        <- Grafici e figure generate da utilizzare nella relazione
│
├── requirements.txt   <- Il file dei requisiti per riprodurre l'ambiente di analisi, 
│                          generato con `pip freeze > requirements.txt`
│
├── setup.py           <- rende il progetto installabile tramite pip, in modo che src possa essere importato
├── src                <- Codice sorgente da utilizzare in questo progetto.
  ├── __init__.py      <- Rende src un modulo Python
    │
    ├── data    
    │   ├── organize_data.py         <- Funzioni per l'organizzazione e la gestione dei dati in un formato pronto per l'analisi
    │   ├── data_selection.py        <- Funzioni per selezione dei campioni in base a determinati valori target e filtraggio outliers 
    │   └── data_preprocessing.py    <- Funzioni per il preprocessing dei dati prima dell'addestramento con i modelli
    │
    ├── features
    │   ├── dimensionality_reduction.py   <- Funzioni per ridurre dimensionalità dei dati               
    │   └── build_features.py             <- Funzioni per estrarre features dai dati
    │
    ├── models               
    │   ├── param_grids.py  <- Contiene i dizionari dei parametri per il fine tuning dei modelli 
    │   ├── models.py       <- Definizioni dei modelli e funzioni correlate
    │   └── evaluation.py   <- Contiene le funzioni per valutare i modelli con la cross-validazione
    │
    ├── statistics                 
    │   ├── tests_normality.py   <- Funzioni per i test di normalità 
    │   ├── tests_difference.py  <- Funzioni per i test di differenza tra gruppi 
    │   └── tests_correlation.py <- Funzioni per i test di correlazione 
    │
    └── visualization  
        └── visualize.py


## Ringraziamenti

Il progetto si basa sul [template cookiecutter data science project](https://drivendata.github.io/cookiecutter-data-science/).

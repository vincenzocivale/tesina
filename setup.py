from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description="""Il progetto che si concentra sull'analisi di un dataset contenente dati di pazienti appartenenti a quattro gruppi distinti: persone sane, individui affetti da sepsi, pazienti con COVID-19 e soggetti con disturbi mentali. L'obiettivo è addestrare un classificatore automatico tramite tecniche di Machine Learning (ML) per identificare correttamente i gruppi di appartenenza""",
    author='Vincenzo Y. Civale',
    license='MIT',
)

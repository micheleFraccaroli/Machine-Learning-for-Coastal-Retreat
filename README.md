# Machine-Learning-for-Coastal-Retreat
usage: coastal_analysis.py [-h] [d] [out] [aug] [tsize] [hpo] [t] [m]

Options: 
    -h, --help 
                    Show this help message and exit 
    d
                    Path of the dataset 
    out 
                    Output path 
    aug 
                    Oversample the dataset: True | False 
    tsize 
                    Size of the test set in percentage 
    hpo 
                    Type of hyper-parameter optimization. Choice between: 'grid_search' | 'random_search' | 'bayes_search' 
    t 
                    Target feature 
    m 
                    Model: RF for random forest or XG for XGBoost 

# Esempio
python coastal_analysis.py dataset_3.xlsx test1/ False 0.1 grid_search cliff_top RF            
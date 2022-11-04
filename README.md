# Machine-Learning-for-Coastal-Retreat
usage: coastal_analysis.py [-h] [d] [out] [aug] [tsize] [hpo] [t] [m] \

Options: 
&emsp;-h, --help \
&emsp&emsp&emsp&emsp&emspShow this help message and exit \
&emspd \
&emsp&emsp&emsp&emsp&emspPath of the dataset \
&emspout \
&emsp&emsp&emsp&emsp&emspOutput path \
&emspaug \
&emsp&emsp&emsp&emsp&emspOversample the dataset: True | False \
&emsptsize \
&emsp&emsp&emsp&emsp&emspSize of the test set in percentage \
&emsphpo \
&emsp&emsp&emsp&emsp&emspType of hyper-parameter optimization. Choice between: 'grid_search' | 'random_search' | 'bayes_search' \
&emspt \
&emsp&emsp&emsp&emsp&emspTarget feature \
&emspm \
&emsp&emsp&emsp&emsp&emspModel: RF for random forest or XG for XGBoost \

# Esempio
python coastal_analysis.py dataset_3.xlsx test1/ False 0.1 grid_search cliff_top RF            

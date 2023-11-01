# Machine-Learning-for-Coastal-Retreat
usage: coastal_analysis.py [-h] [d] [out] [aug] [tsize] [hpo] [t] [m] \

Options: 
&emsp;-h, --help \
&emsp;&emsp;&emsp;&emsp;&emsp;Show this help message and exit \
&emsp;d \
&emsp;&emsp;&emsp;&emsp;&emsp;Path of the dataset \
&emsp;out \
&emsp;&emsp;&emsp;&emsp;&emsp;Output path \
&emsp;aug \
&emsp;&emsp;&emsp;&emsp;&emsp;Oversample the dataset: True | False \
&emsp;tsize \
&emsp;&emsp;&emsp;&emsp;&emsp;Size of the test set in percentage \
&emsp;hpo \
&emsp;&emsp;&emsp;&emsp;&emsp;Type of hyper-parameter optimization. Choice between: 'grid_search' | 'random_search' | 'bayes_search' \
&emsp;t \
&emsp;&emsp;&emsp;&emsp;&emsp;Target feature \
&emsp;m \
&emsp;&emsp;&emsp;&emsp;&emsp;Model: RF for random forest or XG for XGBoost \

# Esempio
python coastal_analysis.py dataset_3.xlsx test1/ False 0.1 grid_search cliff_top RF            

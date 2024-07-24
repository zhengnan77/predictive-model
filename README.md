# predictive-model
####### Requirements #######
1. Make sure you have installed Python 3.11 
2. Install python modules with "pip install -r requirements.txt"
####### Introduction ######
The datas are in "data" folder
There are 4 .py files in "src" folder

calculateVIP.py:
    Used to calculate the Variable Importance in Projection(VIP)
    Change the "dataSetName" to calculate the VIP of different dataSets

outOfBox.py:
    Used to Out Of Box test
    Change the "dataSetName" and "testpath" to evaluate the model with different datas

main.py:
    Used to choose and evaluate models
    Change the "dataSetName" to train different dataSets
    Will print the spearman correlation

test.py:
    Used to evaluate model
    Change the "dataSetName" to load or train different models
    Change the "testpath" to evaluate models with different ways and datas

Only one .py file can be executed at a time.
Models will be saved in the "models" folder
The evaluate results will be print.

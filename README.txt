Aditya Dutta & Ege Gürsel

Project Structure:
    data_to_csv.py:
        Loads raw Excel file and cleans data
        Saves as csv file
    apriori.py:
        Implements Apriori using Numpy
        Generates Association Rules
    clustering.py:
        Performs RMF analysis & K-Means clustering
    requirements.txt:
        list of Python Dependencies


Installation:
    Clone the repository
    Set Up Virtual Environment:
        python3 -m venv venv
    Activate Virtual Environment:
        source venv/bin/activate
    Install Dependancies:
        pip3 install -r requirements.txt


Usage:
    Download raw data from https://archive.ics.uci.edu/dataset/502/online+retail+ii
    Run the following:
        python3 data_to_csv.py
        python3 apriori.py
        python3 clustering.py
This folder containa sample-data.csv with the raw datas used for this project. Using this dataset, inside the notebook you'll find a unsupervised machine learning model (DBSCan)
which sorted all items of the dataset into 15 clusters, data processed and labeled with their respective clusters are then stored in processed_data.csv.
That second dataset is used by the fin_similar_items.py python program that take one item's id in entry and returns a list of 5 random items from the same cluster.
The notebook also contains a tentative of topic modeling of the datas using TruncatedSVD.

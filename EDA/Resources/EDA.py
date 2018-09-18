import numpy as np
import pandas as pd
import random
import seaborn as sns
import matplotlib.pyplot as plt
import os
path = os.getcwd()

def split_dataset(df, size=0.7):
    """Randomly split dataset into train and test datasets
    70% for training and 30% for test by default"""
    df['split'] = np.random.randn(df.shape[0], 1)

    msk = np.random.rand(len(df)) <= size

    train = df[msk]
    test = df[~msk]
    print('Training set dimensions: {}'.format(train.shape))
    print('Test set dimensions: {}'.format(test.shape))
    train = train.drop(['split'], axis='columns')
    train.to_csv('mice_train.csv', index=False)
    test = test.drop(['split'], axis='columns')
    test.to_csv('mice_test.csv', index=False)
    print('Train and Test were datasets generated!')


def correlation_matrix(df, output_name='pearson_corr.png', size1=15, size2=15, cmap='YlGnBu', save=True, display=True):
	"""return correlation matrix among available features
	for a given dataframe. Plot will be done by default with size 15x15 and using YlGnBu as default color map"""

	fig, ax = plt.subplots(figsize=(size1, size2))
	matrix = sns.heatmap(df.corr(method='pearson'), annot=True,
	cmap=cmap)
	if save and not display:
		fig.savefig(output_name)		
		plt.close()
		print('Plot saved at: {}'.format(path))	
	else:
		print('Generating Plot...')	
		return matrix		
	

def box_plot(df, output_name='box-plot.png', size1=10, size2=10, save=False, legend='class', display=True):
	"""Plot box plot for a given dataframe"""
	fig = plt.figure(figsize=(size1,size2))
	box_plot = sns.boxplot(data=df, orient='h')
	#plt.xticks(rotation=90)
	if save and not display:
		fig.savefig(output_name)
		plt.close()
		print('Plot saved at: {}'.format(path))	
	else:
		print('Generating Plot...')	
		return box_plot	


import logging
import pandas as pd
import numpy as np
import category_encoders as ce
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


# TODO 
# - Was sind mögliche Schritte zur Datenvorverarbeitung?`
# - Normalization / Standardisation
# - Null-Werte filtern
# - Outlier (Ausreißer) bestimmen
# - Statistiken (Verteilung, Mittelwert, Varianz, etc.)
# - Bayesian Modelling
# - Taking care of Categorical Features



def ts_auto(data):

	# 1. Nan entfernen
    # 2. Standardisieren
    # 3. Umwandeln kategorischer Daten in Numerische
    # 4. Korrelierende Daten filtern und überflüssige Spalten entfernen




	
	pass


def rm_nan(data, columns = None):
	"""
	Remove nan values from the data.
	"""

	# TODO 
	# - Doku

	rm_df = None

	if columns == []:
		# remove nan on all data
		rm_df= data.dropna()
	else:
		# remove rows if a specific column has nan values
		rm_df = data.dropna(subset=columns)

	return rm_df


def norm(df, columns = None):
	"""
	"""
	# TODO 
	# - Doku

	# Normalize values to range (0,1)
	scaler = MinMaxScaler()

	for col in columns:
		df[col] = scaler.fit_transform(df[col].values.reshape(-1,1))

	return df    


def std(df, columns = None):
	"""
	"""
	# TODO 
	# - Doku

	scaler = StandardScaler()

	for col in columns:
		df[col] = scaler.fit_transform(df[col].values.reshape(-1,1))	
	
	return df    


def encode(df, columns = None, metric = "OneHot"):

	# TODO
	# - Doku
	# - hinzufügen verschiedener Encoding Verfahren
	# - 

	pass



def corr(df, columns = None, corr_limit = 0.8):
	"""
	Find strongly correlating features and remove them.
	"""

	# TODO 
	# - Doku

	corr_matrix = df.corr()
		
	liste = []
	marked = []
	
	# Iterate through all columns and compute the correlation between features
	for i in range (0, np.shape(corr_matrix)[0]):
		for j in range (i+1, np.shape(corr_matrix)[0]):           
			if i != j:                  
				if np.abs(corr_matrix.iloc[i,j]) > corr_limit:                      
					if [i,j] not in liste:                            
						marked.append(i)                                                                    
						liste.append([j, i])
	
	# Remove features with high correlation value
	df.drop(df.columns[marked], axis = 1, inplace = True)  

	return df


def stat(df, percentile = None, include = None):
	# Compute basic data statistics 

	descriptive_stat = df.describe()
	
	
	
	
	
	return descriptive_stat


def _reduce_mem_usage(df):
	""" 
	iterate through all the columns of a dataframe and modify the data type to reduce memory usage.        
	"""

	start_mem = df.memory_usage().sum() / 1024**2
	print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
	
	for col in df.columns:
		col_type = df[col].dtype
		
		if col_type != object:
			c_min = df[col].min()
			c_max = df[col].max()
			if str(col_type)[:3] == 'int':
				if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
					df[col] = df[col].astype(np.int8)
				elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
					df[col] = df[col].astype(np.int16)
				elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
					df[col] = df[col].astype(np.int32)
				elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
					df[col] = df[col].astype(np.int64)  
			else:
				if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
					df[col] = df[col].astype(np.float16)
				elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
					df[col] = df[col].astype(np.float32)
				else:
					df[col] = df[col].astype(np.float64)
		else:
			df[col] = df[col].astype('category')

	end_mem = df.memory_usage().sum() / 1024**2
	print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
	print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

	return df
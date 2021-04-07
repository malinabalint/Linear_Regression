import numpy as np 
import pandas as pd
from IPython.display import display
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn import tree
import matplotlib.pyplot as plt

#1.The following block of code loads the data
try:
    data = pd.read_csv('vgsales.csv')
    print ('Dataset loaded...')
except:
    print ('Unable to load dataset...')

#display(data[:10])
data = data[np.isfinite(data['Year'])]													#"year" feature missing have been deleted 							
naSales = data['NA_Sales']																#setting Y-value(to be predicted)
features = data.drop(['Name', 'Global_Sales', 'NA_Sales'], axis = 1)					#setting X-value(features) dropping Name, Global_Sales, NA_Sales
#display(naSales[:5])
#display(features[:5])

#2.Principal Component Analysis to obtain underlying latent feature
salesFeatures = features.drop(['Rank', 'Platform', 'Year', 'Genre', 'Publisher'], 		#dividing fatures data set into two 
                              axis = 1)
otherFeatures = features.drop(['EU_Sales', 'JP_Sales', 'Other_Sales', 'Rank'], 			 
                              axis = 1)													

pca = PCA(n_components = 1)																#obtain PCA transformed features
pca.fit(salesFeatures)
salesFeaturesTransformed = pca.transform(salesFeatures)

salesFeaturesTransformed = pd.DataFrame(data = salesFeaturesTransformed, 				#merge new transformed salesFeatures	
                                        index = salesFeatures.index, 
                                        columns = ['Sales'])
rebuiltFeatures = pd.concat([otherFeatures, salesFeaturesTransformed], 
                            axis = 1)

#display(rebuiltFeatures[:5])															#displaying first 5 rebuilt features

#3.Processing data
temp = pd.DataFrame(index = rebuiltFeatures.index)

for col, col_data in rebuiltFeatures.iteritems():
    
    if col_data.dtype == object:
        col_data = pd.get_dummies(col_data, prefix = col)
        
    temp = temp.join(col_data)
    
rebuiltFeatures = temp
#display(rebuiltFeatures[:5])
# Dividing the data into training and testing sets...

#4. Divide data into Training and Testing sets.
X_train, X_test, y_train, y_test = train_test_split(rebuiltFeatures, 
                                                    naSales, 
                                                    test_size = 0.2, 
                                                    random_state = 2)

#5. Model Selection
													
# Creating & fitting a Decision Tree Regressor
regDTR = DecisionTreeRegressor(random_state = 4)
regDTR.fit(X_train, y_train)
y_regDTR = regDTR.predict(X_test)

#plt.figure()
#plt.scatter(X_train.iloc[:,2].values, y_train, s=20, edgecolor="black", c="darkorange", label="data")
#plt.plot(X_test, y_regDTR, color="cornflowerblue", label="max_depth=2", linewidth=2)
#plt.xlabel("data")
#plt.ylabel("target")
#plt.title("Decision Tree Regression")
#plt.legend()
#plt.show()

print ('The following is the r2_score on the DTR model...')
print (r2_score(y_test, y_regDTR))														#first param is correct values, second is estimated target

from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
plt.figure()
plt.scatter(X_test.index.values.tolist(), y_regDTR, s=20, edgecolor="black", c="darkorange", label="data")
#plt.plot(X_test, y_regDTR, color="cornflowerblue", label="max_depth=2", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()

# Creating a K Neighbors Regressor##############################################
#regKNR = KNeighborsRegressor()
#regKNR.fit(X_train, y_train)
#y_regKNR = regKNR.predict(X_test)

#print ('The following is the r2_score on the KNR model...')
#print (r2_score(y_test, y_regKNR))

################################################################################

#Creating a Linear Regression 
#regLin = LinearRegression()
#regLin.fit(X_train, y_train)
#y_regLin = regLin.predict(X_test)

#print ('The following is the r2_score on the Linear model...')
#print (r2_score(y_test, y_regLin))

#import matplotlib.pyplot as plt
#plt.plot(y_test, y_regDTR)
#plt.show()
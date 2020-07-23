import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import KBinsDiscretizer

from sklearn import model_selection

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.model_selection import cross_val_score, cross_val_predict, cross_validate

from sklearn import metrics

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

import warnings
warnings.filterwarnings("ignore")

listings = pd.read_csv("D:\Trent Notes\Data Minining\Assignments\Assignment 2\seattle\listings.csv")
print(listings.head())

print(listings.info())

print(listings.describe())

plt.rcParams['figure.figsize']= (20,7)

listings.price = listings.price.str.strip("$").str.replace(",", "")
listings.monthly_price = listings.monthly_price.str.strip("$").str.replace(",", "")
listings.weekly_price = listings.weekly_price.str.strip("$").str.replace(",", "")
print(listings.price.head(3))
print(listings.monthly_price.head(3))
print(listings.weekly_price.head(5))

columns = ['price','bathrooms','bedrooms','beds','last_review']
null_columns =listings[columns]

# mean is calculated for price and then replacd the mean in place of nulls
listings.price = listings.price.astype(np.float64)
price_mean = listings.price.mean()
print(price_mean)
listings["price"]=listings["price"].fillna(listings["price"].mean())

#Mode is calculated for last review  and then replacd with the mode in place of nulls
last_review_mode = listings.last_review.mode()[0]
print(last_review_mode)
listings["last_review"]=listings["last_review"].fillna(listings["last_review"].mode()[0])


#mode is calculated for bathrooms and all the nulls are replaced by nulls 
bathrooms_mode = listings.bathrooms.mode()
print(bathrooms_mode)
listings["bathrooms"]=listings["bathrooms"].fillna(listings["bathrooms"].mode()[0])


#mode is calculated for bedrooms and all the nulls are replaced by nulls 
bedrooms_mode = listings.bedrooms.mode()
print(bedrooms_mode)
listings["bedrooms"]=listings["bedrooms"].fillna(listings["bedrooms"].mode()[0])


#mode is calculated for bedrooms and all the nulls are replaced by nulls 
beds_mode = listings.beds.mode()
print(beds_mode)
listings['beds'] =listings['beds'].fillna(listings['beds'].mode()[0])

null_columns.isnull().sum()

listings.monthly_price = listings.monthly_price.astype(np.float64)

listings.weekly_price = listings.weekly_price.astype(np.float64)

listings.bedrooms = listings.bedrooms.astype(int)

listings.beds = listings.beds.astype(int)

listings.accommodates = listings.accommodates.astype(int)

Room =listings[['room_type','review_scores_cleanliness','review_scores_communication']]
print(Room)

#grouping the dataset by room to perform analysis
room =Room.groupby('room_type').mean()
room

#removing the index of the column
room = room.reset_index()

plt.plot( 'room_type', 'review_scores_cleanliness', data=room, marker='o', markerfacecolor='orange', color='darkblue', linewidth=4)
plt.plot( 'room_type', 'review_scores_communication', data=room, marker='o',  markerfacecolor='orange',color='olive', linestyle='dashed',linewidth=4)
plt.legend()
plt.xlabel("Room_Type")
plt.ylabel("Average Rating out of 10")
plt.title("Average Rating of Cleanleness & Communication By Room Type")
plt.show()

fig,axes= plt.subplots(nrows=2, ncols= 2,figsize=(10,10))
# graph to analyse the count of feedbacks recieved for teh below mentioned categories rated among the scale of 1 to 10
sb.countplot(listings['review_scores_cleanliness'],ax=axes[0][0])
sb.countplot(listings['review_scores_checkin'],ax=axes[0][1])
sb.countplot(listings['review_scores_communication'],ax=axes[1][0])
#sb.countplot(listings['review_scores_location'],ax=axes[1][1])
sb.countplot(listings['review_scores_value'],ax=axes[1][1])
plt.title("Analysis of Rating provided for various factors")
plt.show()


property= listings[['beds', 'price', 'property_type']]
price= property.groupby('property_type', as_index= False).agg({'beds':'mean','price':'mean'})

plt.figure(figsize=(13,8))
sb.scatterplot(x='beds',y='price',hue='property_type',data= price,s =150 ,palette="rainbow")
plt.title("Analysis of no of beds vs Avg Price by property type")
plt.xlabel("No of beds")
plt.ylabel("Average price")
plt.show()

columns = ['accommodates','bathrooms','bedrooms','guests_included','beds','price']
sb.heatmap(listings[columns].corr(),annot=True)
plt.show()

#Analysis of Rating for various Room types based on is loaction exact
sb.violinplot("room_type", "review_scores_rating", hue="is_location_exact", data=listings,palette='rainbow')
plt.show()

# binning the price column 

bins = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')

listings['price']=bins.fit_transform(listings[['price']])

print(listings['price'].unique())

# printing the number of values by count
print(listings["price"].value_counts())

plt.figure(figsize=(8,6))
sb.countplot(listings["price"])
plt.show()

#first converting the last review column to date farm and then subtracting it from todays date to calculate no of days
listings['last_review']= pd.to_datetime(listings['last_review']).dt.date

listings['no_days']= (pd.datetime.now().date()-listings['last_review']).dt.days

print(listings[['last_review', 'no_days']].head())




#binning the availability_30 into 10 equal bins
bins = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')

listings['availability_30']=bins.fit_transform(listings[['availability_30']])

print(listings['availability_30'].unique())

#binning the availability_60 into 10 equal bins
bins = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')

listings['availability_60']=bins.fit_transform(listings[['availability_60']])

print(listings['availability_60'].unique())

#binning the availability_90 into 10 equal bins
bins = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')

listings['availability_90']=bins.fit_transform(listings[['availability_90']])

print(listings['availability_90'].unique())

#binning the availability_365 into 10 equal bins
bins = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')

listings['availability_365']=bins.fit_transform(listings[['availability_365']])

print(listings['availability_365'].unique())

X= listings[['accommodates', 'bathrooms', 'bedrooms','beds']]

predictor= listings[['price']]

X_train, X_test, y_train, y_test = train_test_split(X, predictor, test_size=0.1, random_state=101)

print ("number of training samples {}".format(X_train.shape))
print ("number of tesing {}".format( y_train.shape))

# 1. Applying Decision Tree algorithm to predict the price based on the given training variables

decision= DecisionTreeClassifier()
decission_model= decision.fit(X_train, y_train)
predict= decision.predict(X_test)
print(accuracy_score(y_test, predict))
# creating confusion matrix
confuson_matrix =confusion_matrix(y_test, predict)
print(confuson_matrix)
print(classification_report(y_test, predict))

#performing 10 fold cross validation 
cross_val_scores = cross_val_score(decission_model, X, predictor, cv=10)
print(cross_val_scores)

#creation of confusion matrix plot
sb.set(font_scale=1.0)
sb.set(rc={'figure.figsize':(4,3)})
sb.heatmap(confuson_matrix, annot=True, cmap="YlGnBu")
plt.ylabel('PredictedValueValue')
plt.xlabel('ActualValues')
plt.tight_layout()
plt.show()


#Random forest classifier

random = RandomForestClassifier(random_state = 0)
# training the model with training data
rand_forest =random.fit(X_train,y_train)
# prediction form test data
randfor_pred = random.predict(X_test)
print(accuracy_score(y_test, randfor_pred))
# 10 fold validation
cv_randfor  = cross_validate(rand_forest, X, predictor, cv=10)
print(cv_randfor)

confusion_matrix_randfor = confusion_matrix(y_test, randfor_pred)
print(confusion_matrix_randfor)
# Plotting of the confusion matrix
sb.set(font_scale=1.0)
sb.heatmap(confusion_matrix_randfor, annot=True, cmap="YlGnBu")
plt.ylabel('TrueValue')
plt.xlabel('PredictedValue')
plt.tight_layout()
plt.show()

# 1. Applying Knn to predict the price based on the given training variables

Knn= KNeighborsClassifier(n_neighbors = 11)
knn_model= decision.fit(X_train, y_train)
predict= decision.predict(X_test)
print(accuracy_score(y_test, predict))
# creating confusion matrix
confuson_matrix =confusion_matrix(y_test, predict)
print(confuson_matrix)
print(classification_report(y_test, predict))

#performing 10 fold cross validation 
cross_val_scores = cross_val_score(knn_model, X, predictor, cv=10)
print(cross_val_scores)

#creation of confusion matrix plot
sb.set(font_scale=1.0)
sb.set(rc={'figure.figsize':(4,3)})
sb.heatmap(confuson_matrix, annot=True, cmap="YlGnBu")
plt.ylabel('PredictedValueValue')
plt.xlabel('ActualValues')
plt.tight_layout()
plt.show()

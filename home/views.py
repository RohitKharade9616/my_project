from multiprocessing import context
from django.shortcuts import render,HttpResponse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns 
# Create your views here.

def validate_for_crop(request):
    u=request.GET["username"]
    p=request.GET["pass"]

    if u=="rohit" and p=="1234":
        return render(request,'about.html')
    else:
        return "wrong"


def Login(request):
    return render(request,'login.html')

def contact(request):
    return render(request,'contact.html')

def index(request):
    # return HttpResponse("this is homepage")
    # pass a variable
    context={
        "var":"google"
    }
    return render(request,'index.html',context)


def about(request):
    # return HttpResponse("this is aboutpage")
    return render(request,'about.html')


def fertilizer(request):
    # return HttpResponse("this is contactpage")    
    return render(request,'fertilizer.html')


def predict_crop(request):
    # NITROGEN,PHOSPHORUS,POTASSIUM,TEMPERATURE,HUMIDITY,PH,RAINFALL,CROP
    nitrogen=request.GET["nitrogen"]
    phosphorus=request.GET["phosphorous"]
    potassium=request.GET["pottasium"]
    temp=request.GET["temp"]
    humidity=request.GET["humidity"]
    ph=request.GET["ph"]
    rainfall=request.GET["rainfall"]
    
    # importing required libraries
    


    df=pd.read_csv(r"C:\Users\ROHIT\Desktop\projects\telusko\calc\crop (1).csv")
    # basic information of dataset
    # print(df.head())
    # print(df.info())
    # print(df.describe())
    # print(df.columns)
    # print(df.dtypes)
    # print(df.shape)

    # checking missing values in dataset
    # print(df.isnull().sum())


    # print(df['CROP'].unique())

    # crop_summary=pd.pivot_table(df,index=["CROP"],aggfunc='mean')
    # print(crop_summary)

    x=df.iloc[:,0:7]
    y=df.iloc[:,7]
    X_train,X_test,y_train,y_test=train_test_split(x,y)

    # from sklearn.preprocessing import MinMaxScaler
    # scaler = MinMaxScaler()
    # transform data
    # x = scaler.fit_transform(x)


    from sklearn.linear_model import LogisticRegression
    cls1=LogisticRegression()
    cls1.fit(X_train,y_train)
    y_pred=cls1.predict([[nitrogen,phosphorus,potassium,temp,humidity,ph,rainfall]])
    return render(request,'about.html',{'result':y_pred})


def predict_fertilizer(request):
    te=request.GET["t"]
    hu=request.GET["h"]
    mo=request.GET["m"]
    ni=request.GET["n"]
    po=request.GET["po"]
    pho=request.GET["pho"]

    df=pd.read_csv("Fertilizer_Prediction.csv")
    # basic information of dataset
    # print(df.head())
    # print(df.info())
    # print(df.describe())
    # print(df.columns)
    # print(df.dtypes)
    # print(df.shape)


    # # checking missing values in dataset
    # # print(df.isnull().sum())


    # # print(df['CROP'].unique())

    # # crop_summary=pd.pivot_table(df,index=["CROP"],aggfunc='mean')
    # # print(crop_summary)

    # # correlation matric
    # cor=df.corr()
    # dataplot = sns.heatmap(cor, cmap="YlGnBu", annot=True)
    # # plt.show()
    df=df.drop(columns=["Soil Type","Crop Type"])
    print(df.head())

    x=df.iloc[:,0:6]
    # print(x.head())
    y=df.iloc[:,6]
    # print(y.head)

    # splitting the dataset
    X_train,X_test,y_train,y_test=train_test_split(x,y)

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    # transform data
    x = scaler.fit_transform(x)
    
    #model using  logistic regression
    from sklearn.linear_model import LogisticRegression
    cls1=LogisticRegression()
    # cls1.fit(X_train,y_train)
    # y_pred=cls1.predict(X_test)
    # print(y_pred)


    # model using knn
    from sklearn.neighbors import KNeighborsClassifier  
    cls2= KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2 )  
    # cls.fit(X_train, y_train)  
    # y_pred=cls.predict(X_test)
    # from sklearn.metrics import accuracy_score
    # accuracy=accuracy_score(y_pred , y_test)
    # print(accuracy)




    # model using suppport vector machine
    from sklearn.svm import SVC
    cls3=SVC()
    # cls.fit(X_train,y_train)
    # y_pred=cls.predict(X_test)
    # from sklearn.metrics import accuracy_score
    # accuracy=accuracy_score(y_pred , y_test)
    # print(accuracy)

    from sklearn.tree import DecisionTreeClassifier
    cls4=DecisionTreeClassifier(random_state=1)


    # using ensemble learning
    from sklearn.ensemble import VotingClassifier
    model = VotingClassifier(estimators=[('lr', cls1), ('knn',cls2),('svm',cls3),('dt',cls4)], voting='hard')
    model.fit(X_train,y_train)
    y_pred=model.predict([[te,hu,mo,ni,po,pho]])
    return render(request,'fertilizer.html',{'result':y_pred})
    # print(model.score(X_test,y_test))
    # y_pred=model.predict(X_test)
    # print(y_pred)

    # estimate performance of model
    # from sklearn.metrics import confusion_matrix,accuracy_score
    # cm=confusion_matrix(y_test,y_pred)
    # plt.figure(figsize=(15,15))
    # sns.heatmap(cm,annot=True,fmt=".0f",linewidths=.5,square=True,cmap="Blues")
    # plt.ylabel("actual label")
    # plt.xlabel("predicted label")
    # all_sample_title="Confusion Matrix - score :"+str(accuracy_score(y_test,y_pred))
    # plt.title(all_sample_title,size=15)
    # plt.show()


    # classificatio report

    # from sklearn.metrics import classification_report
    # print(classification_report(y_test,y_pred))
    
        
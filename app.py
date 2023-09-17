import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve,confusion_matrix,precision_recall_curve
from sklearn.metrics import precision_score, recall_score,roc_curve
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
def main():
    st.title("Binary Classification Web App")
    st.sidebar.title("Binary Classification Web App")
    st.markdown("Are your mushroom edible or Piosonous ? 🍄")
    st.sidebar.markdown("Are your mushroom edible or Piosonous ? 🍄")

    @st.cache_data(persist=True)
    def load_data():
        data=pd.read_csv('mushrooms.csv')
        label = LabelEncoder()
        for col in data.columns:
            data[col] = label.fit_transform(data[col])
        return data
        
    @st.cache_data(persist=True)
    def split(df):
        y = df.type
        x = df.drop(columns = ['type'])
        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3, random_state = 0)
        return x_train,x_test,y_train,y_test
    def plot_metrics(matrics_list,y_score,clf):
        if 'Confusion Matrix' in matrics_list:
            st.subheader('Confusion Matrix')
            clf = make_pipeline(StandardScaler(), model)
            clf.fit(x_train, y_train)
            y_score = clf.decision_function(x_test)

            fpr, tpr, _ = confusion_matrix(y_test, y_score)

            fig, ax = plt.subplots()  # Create a figure containing a single axes.
            ax.plot(fpr, tpr)
            #print(roc_display)
            st.pyplot(fig)
        if 'ROC Curve' in matrics_list:
            st.subheader("ROC Curve")


            fpr, tpr, _ = roc_curve(y_test, y_score, pos_label=clf.classes_[1])

            fig, ax = plt.subplots()  # Create a figure containing a single axes.
            ax.plot(fpr, tpr)
            #print(roc_display)
            st.pyplot(fig)
        if 'Precision-Recall Curve' in matrics_list:
            st.subheader("Precision Recall")
            


            fpr, tpr, _ = precision_recall_curve(y_test, y_pred, pos_label=clf.classes_[1])

            fig, ax = plt.subplots()  # Create a figure containing a single axes.
            ax.plot(fpr, tpr)
            #print(roc_display)
            st.pyplot(fig)

    df = load_data()
    x_train,x_test,y_train,y_test = split(df)
    class_names = ['edible','poisonous']
    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox("Classifier",("Support Vector Machine (SVM)","Logistic regression"))

    if classifier == 'Support Vector Machine (SVM)':
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01,10.0,step=0.01,key='C')
        kernel = st.sidebar.radio("Kernel",("rbf","linear"),key = 'kernel')
        gamma = st.sidebar.radio("Gamma (Kernel Cofficient)",("scale","auto"),key='gamma')

        metrics = st.sidebar.multiselect("What metrices to plot?", ('ROC Curve','Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Support Vector Machine (SVM) Results")
            model = SVC(C=C,kernel=kernel,gamma=gamma)
            model.fit(x_train,y_train)
            accuracy = model.score(x_test,y_test)
            y_pred = model.predict(x_test)

            clf = make_pipeline(StandardScaler(), model)
            clf.fit(x_train, y_train)
            y_score = clf.decision_function(x_test)

            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precission: ",precision_score(y_test,y_pred,labels=class_names).round(2))
            st.write("Recall: ",recall_score(y_test,y_pred,labels=class_names).round(2))
            plot_metrics(metrics,y_score,clf)

    if classifier == 'Logistic regression':
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01,10.0,step=0.01,key='C_LR')
        max_iter = st.sidebar.slider("Maximum number of iterations", 100,500,key='max_iter')

        metrics = st.sidebar.multiselect("What metrices to plot?", ('ROC Curve','Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Logistic regression Results")
            model = LogisticRegression(C=C,max_iter=max_iter)
            model.fit(x_train,y_train)
            accuracy = model.score(x_test,y_test)
            y_pred = model.predict(x_test)

            clf = make_pipeline(StandardScaler(), model)
            clf.fit(x_train, y_train)
            y_score = clf.decision_function(x_test)


            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precission: ",precision_score(y_test,y_pred,labels=class_names).round(2))
            st.write("Recall: ",recall_score(y_test,y_pred,labels=class_names).round(2))
            plot_metrics(metrics,y_score,clf)

    if classifier == 'Random Forest':
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators=st.sidebar.number_input("The number of trees in the forest",100,5000,step = 10,key='n_estimator')
        max_depth = st.sidebar.number_input("The maximum depth",1,20,step=1,key="max_depth")
        bootstrap = st.sidebar.radio("Bootstrap samples ",('True','False'),key='bootstrap')

        metrics = st.sidebar.multiselect("What metrices to plot?", ('ROC Curve','Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Random Forest Results")
            model = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,bootstrap=bootstrap,n_jobs=-1)
            model.fit(x_train,y_train)
            accuracy = model.score(x_test,y_test)
            y_pred = model.predict(x_test)

            clf = make_pipeline(StandardScaler(), model)
            clf.fit(x_train, y_train)
            y_score = clf.decision_function(x_test)

            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precission: ",precision_score(y_test,y_pred,labels=class_names).round(2))
            st.write("Recall: ",recall_score(y_test,y_pred,labels=class_names).round(2))
            plot_metrics(metrics,y_score,clf)

    if st.sidebar.checkbox("Show raw data ",False):
        st.subheader("Mushroom Data Set (Classification)")
        st.write(df)

        
if __name__ == '__main__':
    main()

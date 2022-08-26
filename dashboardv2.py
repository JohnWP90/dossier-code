import streamlit as st
from streamlit_lottie import st_lottie
from st_aggrid import AgGrid
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.metrics import auc, confusion_matrix
import seaborn as sns
from PIL import Image
import requests


# Title
st.set_page_config(page_title="Home Credit Default Risk")
title= st.title("Home Credit Default Risk")
logo = Image.open("logo_pret_a_depense.png")
st.sidebar.image(logo,width=280)

# Function to import dataframe
@st.cache
def get_dataset(csv_file ):
    data = pd.read_csv(csv_file, index_col=0).reset_index(drop=True)
    return data

# Tables
data_clients = get_dataset("processed_test_data.csv") # modification
data = get_dataset("dashboard_test_data.csv")
data_index = data.set_index("SK_ID_CURR")
descriptions_id = get_dataset("descriptions_applicants.csv")
descriptions_id_index = descriptions_id.set_index("SK_ID_CURR")
roc_curve = get_dataset("fpr_tpr_table.csv")
precision_recall = get_dataset("precision_recall_table.csv")
y_table = get_dataset("y_table.csv")

# Selectbox for selection of applicant through SK_ID_CURR
applicant_id = st.sidebar.selectbox(label="Select or write applicant's ID :", options=data_clients.SK_ID_CURR, index=0)

if applicant_id:
    st.subheader("Applicant's ID :")

    # Multiselect options for the applicant's personal information.
    id_options = st.multiselect("Select the columns you want to see :", descriptions_id.columns, default=["NAME_CONTRACT_TYPE","CODE_GENDER","CNT_CHILDREN","AMT_INCOME_TOTAL"])
    id_table = st.dataframe(descriptions_id.loc[descriptions_id.SK_ID_CURR==applicant_id,id_options])
    
    # Features for scoring
    scoring_features = st.checkbox(label="Show important features used for scoring:", value=False)
    if scoring_features:
        st.write("Important features in scoring:")
        # Multiselect options for features used in calculating the probabilities.
        features_options = st.multiselect(label="Features and probabilities:",
        options=data.columns,
        default=["PAYMENT_RATE","EXT_SOURCE_1","EXT_SOURCE_2","EXT_SOURCE_3","DAYS_BIRTH"]
        )
        important_features_table = st.dataframe(data.loc[data.SK_ID_CURR==applicant_id,features_options])


    # Buttons
    get_score = st.sidebar.radio("Applicant's score", options=["Show score","Hide score"], index=0)
    # Figure showing probabilities
    url_api = "https://api-flask-version6.herokuapp.com/"
    response = requests.post(url=url_api, json={'applicant_id':applicant_id})
    
    if response.status_code == 200:
        r = response.json()
        proba_paying = r["Score "]
        proba_not_paying = 1-proba_paying
    else:
        print(response.status_code)

    # proba_paying = np.round(float(data.loc[data.SK_ID_CURR==applicant_id, "PROBABILITY_PAYING"].values),2)
    # proba_not_paying = np.round(float(data.loc[data.SK_ID_CURR==applicant_id, "PROBABILITY_NOT_PAYING"].values),2)

    if get_score == "Show score":
        if proba_paying>proba_not_paying:
            st.sidebar.markdown(f"<h2 style='text_align:right; color: lightgreen;'>Score={proba_paying}, Credit approved !</h2>", unsafe_allow_html=True)
        else:
            st.sidebar.markdown(f"<h2 style='text_align:right; color: red;'>Score={proba_paying}, Credit refused !</h2>", unsafe_allow_html=True)        


    graphs = st.sidebar.radio(label="Graphs", options=["Model performance","Similar applicants"], index=0)
    if graphs == "Model performance":
        # Checboxes
        check_radio = st.sidebar.radio("Check to visualise:",
        options=["Probabilities","ROC Curve","Confusion matrix","Precision-Recall","Thresholds"]
        )

        if check_radio == "Probabilities":
            fig1 = px.bar(data_frame=data,
                        x = ['Paying back','Not paying'], 
                        y=[proba_paying, proba_not_paying], 
                        color=["Repaying" ,"Not repaying"], 
                        opacity=0.7,
                        title='Probabilities for the applicant to repay or not',
                        text=[proba_paying ,proba_not_paying]
                        )
            st.plotly_chart(fig1, use_container_width=False)    

        if check_radio == "ROC Curve":
            roc_score = auc(roc_curve.fpr,roc_curve.tpr)
            fig2 = px.area(data_frame = roc_curve, x='fpr', y='tpr', labels={"x":"False Positive Rate","y":"True Positive Rate"},title="ROC Curve")
            fig2.add_shape(
                    type='line', line=dict(dash='dash'),
                    x0=0, x1=1, y0=0, y1=1)
            st.plotly_chart(fig2, use_container_width=False)    
            st.info(f"AUC score = {round(roc_score,2)}")

        if check_radio == "Confusion matrix":
            fig3 = px.imshow(
                confusion_matrix(y_table.ytrue, y_table.ypred),
                text_auto=True, # Write the numbers.
                labels=dict(x="Predictions", y="Actual"),
                x=[0,1],
                y=[0,1],
                title="Confusion matrix",
                )
            st.plotly_chart(fig3, use_container_width=False) 

        if check_radio == "Precision-Recall":
            fig4 = auc(precision_recall.recall, precision_recall.precision)
            fig4 = px.area(data_frame = precision_recall, x='recall', y='precision', labels={"x":"Recall","y":"Precision"},title="Precision-Recall curve")

            st.plotly_chart(fig4, use_container_width=False)    
            st.info(f"Average precision score = {0.24}")

        

        if check_radio == "Thresholds":
            fig6 = px.line(data_frame=roc_curve, x='thresholds', y=['fpr','tpr'])
            st.plotly_chart(fig6, use_container_width=False)
            st.info("Max(TPR-FPR) is at Thresholds = 0.41")



    else :
        # Selecting similar applicants with similar probabilities.
        similar_ids = data.loc[data.PROBABILITY_PAYING.between(proba_paying-0.01,proba_paying+0.01),"SK_ID_CURR"].tolist()
        container_width = True

        def show_info():
            st.info(f"- Number of similar applicants = {len(similar_ids)}/{len(descriptions_id)}\n - The **\" \- \"** sign means unknown")

        def plot_histogram(data_column, radio_name, nbins=None):
            if nbins!=None:
                bins_number = st.slider("Number of bins:", min_value=10, max_value=100, value=nbins)
            else:
                bins_number=nbins

            fig = px.histogram(x=descriptions_id_index.loc[similar_ids, data_column],
                color=descriptions_id_index.loc[similar_ids, data_column],
                labels=dict(color=radio_name, x=radio_name),
                nbins=bins_number
                )
            st.plotly_chart(fig, use_container_width=container_width)
            show_info()

        # Similar Clients
        compare_with_others = st.sidebar.radio("Characteristics of other similar applicants :",
            options=[
                "Distribution of probability",
                "Gender",
                "Family status",
                #"Number of Children",
                "Income",
                "Occupation",
                "Number of children"
                ],
            index=1
            )

        # First histogram
        if compare_with_others=="Distribution of probability":
            bin_number = st.slider("Number of bins",min_value=10, max_value=100, value=42)
            fig5a = px.histogram(
                x=data_index.loc[similar_ids,"PROBABILITY_PAYING"],
                color=data_index.loc[similar_ids,"PREDICTION"],
                nbins=bin_number,
                labels=dict(color='Labels', x='probability of repaying'),
                )
            st.plotly_chart(fig5a, use_container_width=container_width)
            show_info()
            st.info(f"- {0} : repaying \n - {1} : not repaying")

        # Second histogram
        if compare_with_others=="Gender":
            plot_histogram("CODE_GENDER","Gender")

        # Third histogram
        if compare_with_others=="Number of children":
            plot_histogram("CNT_CHILDREN", "Number of children")#,"Number of children", nbins=10)

        # Fourth histogram
        if compare_with_others=="Income":
            plot_histogram("AMT_INCOME_TOTAL","Income", nbins=50)

        # Fifth histogram
        if compare_with_others=="Occupation":
            plot_histogram("OCCUPATION_TYPE","Occupation")

        #Sixth histogram
        if compare_with_others=="Family status":
            plot_histogram("NAME_FAMILY_STATUS", "Family status")
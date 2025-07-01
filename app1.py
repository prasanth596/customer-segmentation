import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import matplotlib.pyplot as plt

st.title("🛍️ Customer Segmentation & Market Basket Analysis")

# Upload dataset
uploaded_file = st.file_uploader("Upload Customer Data CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:", df.head())

    option = st.selectbox("Choose Analysis", ["Customer Segmentation", "Market Basket Analysis"])

    if option == "Customer Segmentation":
        st.subheader("KMeans Clustering")
        income_col = st.selectbox("Select Income Column", df.columns)
        spend_col = st.selectbox("Select Spending Column", df.columns)

        n_clusters = st.slider("Choose number of clusters", 2, 10, 3)
        model = KMeans(n_clusters=n_clusters)
        df['Cluster'] = model.fit_predict(df[[income_col, spend_col]])
        st.write(df.head())

        # Plot
        fig, ax = plt.subplots()
        scatter = ax.scatter(df[income_col], df[spend_col], c=df['Cluster'])
        plt.xlabel(income_col)
        plt.ylabel(spend_col)
        st.pyplot(fig)

    elif option == "Market Basket Analysis":
        st.subheader("Apriori Association Rules")

        # Assume data is list of lists
        transactions = df.values.tolist()
        te = TransactionEncoder()
        te_array = te.fit(transactions).transform(transactions)
        df_encoded = pd.DataFrame(te_array, columns=te.columns_)

        frequent = apriori(df_encoded, min_support=0.3, use_colnames=True)
        rules = association_rules(frequent, metric="lift", min_threshold=1)
        st.write("Association Rules", rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

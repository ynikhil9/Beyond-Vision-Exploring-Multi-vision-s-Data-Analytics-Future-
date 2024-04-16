import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import missingno as msno
import stemgraphic
from scipy import stats
from streamlit_lottie import st_lottie
import json
import requests


# Set page title and description
st.set_page_config(page_title="Data Analysis Dashboard", layout="wide")


# Define a function to create a linear regression model
def train_linear_regression(df, selected_feature, target_feature):
    # Extract selected features and target feature
    X = df[selected_feature].values.reshape(-1, 1)
    y = df[target_feature].values.reshape(-1, 1)
    
    # Train-test split (you may need to customize this)
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Train the linear regression model
    linear_regressor = LinearRegression()
    linear_regressor.fit(X_train, y_train)
    
    # Return the trained model and test data for evaluation
    return linear_regressor, X_test, y_test

# Simple Linear Regression

def SimpleLinearRegression():
    def get_url(url:str):
        r=requests.get(url)
        if r.status_code !=200:
            return None
        return r.json()

    url=get_url("https://assets6.lottiefiles.com/packages/lf20_vvjhceqy.json")
    url2=get_url(("https://assets6.lottiefiles.com/packages/lf20_qpsnmykx.json"))
    url3=get_url("https://assets1.lottiefiles.com/packages/lf20_uxndffhr.json")


    colx,coly=st.columns(2)
    with colx:
        st.title("Beyond Vision: Exploring Multivision's Data Analytics Future")
        st.write("Upload a CSV or Excel file for Dashboard")
    with coly:
        st_lottie(url)


    # Upload file
    uploaded_file = st.sidebar.file_uploader("Upload file", type=["csv", "xlsx"])

    try:
        if uploaded_file is not None:
            # Determine file type
            file_type = "csv" if uploaded_file.type == "text/csv" else "excel"

            # Read file into DataFrame
            if file_type == "csv":
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            # File Information
            
            st.sidebar.subheader("File Information")
            st.sidebar.write("File Name:", uploaded_file.name)
            st.sidebar.write("Number of Rows:", df.shape[0])
            st.sidebar.write("Number of Columns:", df.shape[1])
            

            # Missing Values
            st.sidebar.subheader("Missing Values")
            missing_values = pd.DataFrame(df.isnull().sum(), columns=["Missing Values"]).reset_index()
            missing_values.columns = ["Column", "Missing Values"]
            st.sidebar.dataframe(missing_values)

            if df.isnull().values.any():
                # Drop rows with missing values
                df.dropna(inplace=True)
                # Display the updated DataFrame in Streamlit
                st.sidebar.subheader("Handling missing values")
                missing_values = pd.DataFrame(df.isnull().sum(), columns=["Missing Values"]).reset_index()
                missing_values.columns = ["Column", "Missing Values"]
                st.sidebar.dataframe(missing_values)


            # Data Types
            st.sidebar.subheader("Data Types")
            data_types = pd.DataFrame(df.dtypes, columns=["Data Type"]).reset_index()
            data_types.columns = ["Column", "Data Type"]
            st.sidebar.dataframe(data_types)
            

            # Display the first few rows of the DataFrame
            st.subheader("Preview")
            st.dataframe(df.head())

            

            # Create columns for layout
            col1, col2, col3 = st.columns(3)

            # Summary Statistics
            with col1:
                st.subheader("Summary Statistics")
                st.write(df.describe())

            # Correlation Heatmap
            with col2:
                st.subheader("Correlation Heatmap")
                corr = df.corr()

                # Display the correlation matrix as a DataFrame
                st.write(corr)
                
                

            # Histogram
            with col3:
                st.subheader("Histogram")
                selected_column_hist = st.selectbox("Select a column for the histogram", df.columns)
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.histplot(data=df, x=selected_column_hist, kde=True, ax=ax)
                st.pyplot(fig)

            col4,col5, colz=st.columns(3)
            with col4:
            # Density Plot
                st.subheader("Density Plot")
                selected_column_density = st.selectbox("Select a column for the density plot", df.columns)
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.kdeplot(data=df, x=selected_column_density, fill=True, ax=ax)
                st.pyplot(fig)
            with colz:
                st_lottie(url2)

            with col5 :
                

            # Area Plot
                st.subheader("Area Plot")
                selected_column_area = st.selectbox("Select a column for the area plot", df.columns)
                area_data = df[selected_column_area].value_counts().reset_index()
                area_data.columns = ["Value", "Count"]
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.lineplot(data=area_data, x="Value", y="Count", ax=ax)
                st.pyplot(fig)

            
            # Select features for linear regression
            st.subheader("Select Features for Linear Regression")
            selected_feature = st.selectbox("Select a feature  (independent variable)", df.columns)
            default_target_feature = df.columns[-1]  # Get the name of the last column as default
            target_feature = st.selectbox("Select a target column (dependent variable)", df.columns, index=len(df.columns)-1, format_func=lambda x: default_target_feature if x == "" else x)

            # Train linear regression model
            linear_regressor, X_test, y_test = train_linear_regression(df, selected_feature, target_feature)

            # Display linear regression coefficients and intercept
            st.subheader("Linear Regression Model")
            st.write("Coefficient (slope):", linear_regressor.coef_[0][0])
            st.write("Intercept:", linear_regressor.intercept_[0])

            

            # Display prediction results
            st.subheader("Prediction Results")
            y_pred = linear_regressor.predict(X_test)
            results = pd.DataFrame({"Actual": y_test.flatten(), "Predicted": y_pred.flatten()})
            st.write(results)

            # Calculate mean squared error (cost)
            cost = mean_squared_error(y_test, y_pred)
            st.subheader("Mean Squared Error (Cost)")
            st.write(cost)

            # Plot actual vs. predicted values
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(X_test, y_test, color="blue", label="Actual")
            ax.plot(X_test, y_pred, color="red", label="Predicted")
            ax.set_xlabel(selected_feature)
            ax.set_ylabel(target_feature)
            ax.legend()
            st.pyplot(fig)
            

            
                
    except Exception as e:
        st.error("An error occurred: {} : Try another column for continue".format(str(e)))


# Multiple Linear Regression Method 

def MultipleLinearRegression():
    def get_url(url:str):
        r=requests.get(url)
        if r.status_code !=200:
            return None
        return r.json()

    url=get_url("https://assets6.lottiefiles.com/packages/lf20_vvjhceqy.json")
    url2=get_url(("https://assets6.lottiefiles.com/packages/lf20_qpsnmykx.json"))
    url3=get_url("https://assets1.lottiefiles.com/packages/lf20_uxndffhr.json")


    colx,coly=st.columns(2)
    with colx:
        st.title("Beyond Vision: Exploring Multivision's Data Analytics Future")
        st.write("Upload a CSV or Excel file for Dashboard")
    with coly:
        st_lottie(url)


    # Upload file
    uploaded_file = st.sidebar.file_uploader("Upload file", type=["csv", "xlsx"])

    try:
        if uploaded_file is not None:
            # Determine file type
            file_type = "csv" if uploaded_file.type == "text/csv" else "excel"

            # Read file into DataFrame
            if file_type == "csv":
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            # File Information
            
            st.sidebar.subheader("File Information")
            st.sidebar.write("File Name:", uploaded_file.name)
            st.sidebar.write("Number of Rows:", df.shape[0])
            st.sidebar.write("Number of Columns:", df.shape[1])
            

            # Missing Values
            st.sidebar.subheader("Missing Values")
            missing_values = pd.DataFrame(df.isnull().sum(), columns=["Missing Values"]).reset_index()
            missing_values.columns = ["Column", "Missing Values"]
            st.sidebar.dataframe(missing_values)

            if df.isnull().values.any():
                # Drop rows with missing values
                df.dropna(inplace=True)
                # Display the updated DataFrame in Streamlit
                st.sidebar.subheader("Handling missing values")
                missing_values = pd.DataFrame(df.isnull().sum(), columns=["Missing Values"]).reset_index()
                missing_values.columns = ["Column", "Missing Values"]
                st.sidebar.dataframe(missing_values)


            # Data Types
            st.sidebar.subheader("Data Types")
            data_types = pd.DataFrame(df.dtypes, columns=["Data Type"]).reset_index()
            data_types.columns = ["Column", "Data Type"]
            st.sidebar.dataframe(data_types)
            

            # Display the first few rows of the DataFrame
            st.subheader("Preview")
            st.dataframe(df.head())

            

            # Create columns for layout
            col1, col2, col3 = st.columns(3)

            # Summary Statistics
            with col1:
                st.subheader("Summary Statistics")
                st.write(df.describe())

            # Correlation Heatmap
            with col2:
                st.subheader("Correlation Heatmap")
                corr = df.corr()

                # Display the correlation matrix as a DataFrame
                st.write(corr)
                
                

            # Histogram
            with col3:
                st.subheader("Histogram")
                selected_column_hist = st.selectbox("Select a column for the histogram", df.columns)
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.histplot(data=df, x=selected_column_hist, kde=True, ax=ax)
                st.pyplot(fig)

            col4,col5, colz=st.columns(3)
            with col4:
            # Density Plot
                st.subheader("Density Plot")
                selected_column_density = st.selectbox("Select a column for the density plot", df.columns)
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.kdeplot(data=df, x=selected_column_density, fill=True, ax=ax)
                st.pyplot(fig)
            with colz:
                st_lottie(url2)

            with col5 :
                

            # Area Plot
                st.subheader("Area Plot")
                selected_column_area = st.selectbox("Select a column for the area plot", df.columns)
                area_data = df[selected_column_area].value_counts().reset_index()
                area_data.columns = ["Value", "Count"]
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.lineplot(data=area_data, x="Value", y="Count", ax=ax)
                st.pyplot(fig)

            # Implement multiple linear regression
            st.subheader("Multiple Linear Regression")

            # Selecting features and target
            output_col = st.selectbox("Select the target column", df.columns, index=len(df.columns)-1)

            feature_cols = [col for col in df.columns if col != output_col]

            # Splitting data into features and target
            X = df[feature_cols]
            y = df[output_col]

            # Splitting data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # Train the model
            lr = LinearRegression()
            lr.fit(X_train, y_train)

            # Make predictions
            y_pred = lr.predict(X_test)

            # Calculate MSE
            mse = mean_squared_error(y_test, y_pred)

            st.write("Intercept:", lr.intercept_)
            st.write("Coefficients:", lr.coef_)
            st.write("Mean Squared Error:", mse)

            # Plotting actual vs predicted values
            fig, ax = plt.subplots()
            ax.plot(X_test, y_test, "*", color="green", label="Actual Values")
            ax.plot(X_test, y_pred, "+", color="red", label="Predicted Values")
            ax.set_title("Performance Testing")
            ax.set_xlabel("Input")
            ax.set_ylabel("Output")
            ax.legend()
            st.pyplot(fig)


            

            
                
    except Exception as e:
        st.error("An error occurred: {} : Try another column for continue".format(str(e)))




    


#st.sidebar.header("Created by Tanu Maurya ✨")

choice = st.sidebar.selectbox('Menu', ['Simple Linear Regression', 'Multiple Linear Regression'])

if choice == 'Simple Linear Regression':
    SimpleLinearRegression()
elif choice == 'Multiple Linear Regression':
    MultipleLinearRegression()

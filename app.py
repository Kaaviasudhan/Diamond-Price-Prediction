import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import seaborn as sns
import pickle
import joblib

from sklearn.tree import DecisionTreeRegressor


# Load the pre-trained model and encoder from pickle files
model = pickle.load(open('rf.pkl', 'rb'))

# Define the app layout
st.title('Diamond Price Predictor')
st.image("""https://www.thestreet.com/.image/ar_4:3%2Cc_fill%2Ccs_srgb%2Cq_auto:good%2Cw_1200/MTY4NjUwNDYyNTYzNDExNTkx/why-dominion-diamonds-second-trip-to-the-block-may-be-different.png""")


# Predefined Function:
# Define the prediction function
def predict(carat, cut, color, clarity, depth, table, vol):
    #Predicting the price of the carat
    if cut == 'Fair':
        cut = 0
    elif cut == 'Good':
        cut = 1
    elif cut == 'Very Good':
        cut = 2
    elif cut == 'Premium':
        cut = 3
    elif cut == 'Ideal':
        cut = 4

    if color == 'J':
        color = 0
    elif color == 'I':
        color = 1
    elif color == 'H':
        color = 2
    elif color == 'G':
        color = 3
    elif color == 'F':
        color = 4
    elif color == 'E':
        color = 5
    elif color == 'D':
        color = 6
    
    if clarity == 'I1':
        clarity = 0
    elif clarity == 'SI2':
        clarity = 1
    elif clarity == 'SI1':
        clarity = 2
    elif clarity == 'VS2':
        clarity = 3
    elif clarity == 'VS1':
        clarity = 4
    elif clarity == 'VVS2':
        clarity = 5
    elif clarity == 'VVS1':
        clarity = 6
    elif clarity == 'IF':
        clarity = 7
    
    prediction = model.predict(pd.DataFrame([[carat, cut, color, clarity, depth, table, vol]], columns=['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'vol']))
    return prediction

app_mode = st.sidebar.selectbox('Select Page', ['Home', 'Analysis', 'Predict the Price'])

if app_mode == 'Home':
    st.markdown('Dataset:')
    df = pd.read_csv('./dataset/diamonds.csv')
    df = df.drop('Unnamed: 0', axis=1)
    st.write(df.head())
    
elif app_mode == 'Analysis':
    st.title('Analysis:')
    st.markdown('Dataset:')
    df = pd.read_csv('./dataset/diamonds.csv')
    df = df.drop('Unnamed: 0', axis=1)
    st.write(df.head())
    
    df.drop(['x','y','z'], axis=1, inplace=True)
    cut_mapping = {'Fair': 0, 'Good': 1, 'Very Good': 2, 'Premium': 3, 'Ideal': 4}
    df.cut = df.cut.map(cut_mapping)
    color_mapping = {'J': 0, 'I': 1, 'H': 2, 'G': 3, 'F': 4, 'E': 5, 'D': 6}
    df.color = df.color.map(color_mapping)
    clarity_mapping = {'I1': 0, 'SI2': 1, 'SI1': 2, 'VS2': 3, 'VS1': 4, 'VVS2': 5, 'VVS1': 6, 'IF': 7}
    df.clarity = df.clarity.map(clarity_mapping)
    
#     # The scatter matrix plot visualizes data distribution and also shows correlations that exist between attributes
#     diamonds_scattergram = scatter_matrix(df, alpha=0.5, diagonal="kde", figsize=(20,17))
#     plt.show()
    
#     # Correlation analysis using Pearson correlation (linear correlation analysis)
#     # Pearson correlation coefficient values will be represented on a heatmap
#     correlation_matrix = df.corr()

#     plt.figure(figsize=(10,6))
#     sns.heatmap(correlation_matrix, cmap="coolwarm", annot=True)
#     plt.title("Correlation heatmap", fontsize=14, pad=12)
#     plt.show()
    
    st.markdown('Model Fit and Evalute:')
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor, export_text
    import sklearn.ensemble as se
    
    # Assign X and Y Variable value:
    X = df.drop(['price'], axis=1)
    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)    
    
    mms=MinMaxScaler(feature_range=(0,1))
    X_train=mms.fit_transform(X_train)
    X_test=mms.fit_transform(X_test)
    X_train=pd.DataFrame(X_train)
    X_test=pd.DataFrame(X_test)

    lr = LinearRegression()
    
    # Create an instance of DecisionTreeRegressor
    tree = DecisionTreeRegressor(random_state=42)
    
    rf = se.RandomForestRegressor(n_estimators=100, random_state=42)

    model=[lr,tree,rf]

    for models in model:
        models.fit(X_train,y_train)

        ypred=models.predict(X_test)
        
        st.text('Model :')
        st.write(models)
        st.text('-----------------------------------------------------------------------------------------------------------------------')
        st.write('R squared of this model on training set: {:.2%}'.format(models.score(X_train, y_train)))
        st.write('R squared of this model on test set: {:.2%}'.format(models.score(X_test, y_test))) 
    

elif app_mode == 'Predict the Price':
    st.markdown('Hello')
    st.header('Enter the characteristics of the diamond:')
    carat = st.number_input('Carat Weight:', min_value=0.1, max_value=10.0, value=1.0)
    cut = st.selectbox('Cut Rating:', ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'])
    color = st.selectbox('Color Rating:', ['J', 'I', 'H', 'G', 'F', 'E', 'D'])
    clarity = st.selectbox('Clarity Rating:', ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'])
    depth = st.number_input('Diamond Depth Percentage:', min_value=0.1, max_value=100.0, value=1.0)
    table = st.number_input('Diamond Table Percentage:', min_value=0.1, max_value=100.0, value=1.0)
    vol = st.number_input('Diamond Length, Width, Height (X, Y, Z) in mm:', min_value=0.1, max_value=100.0, value=1.0)
    

    if st.button('Predict Price'):
        price = predict(carat, cut, color, clarity, depth, table, vol)
        st.success(f'The predicted price of the diamond is ${price[0]:.2f} USD')
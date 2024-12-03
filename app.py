# Imports
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler

# Sidebar Navigation
page = st.sidebar.selectbox("Select a Page", ["Home", "Overview", "Exploratory Data Analysis", "Model Training and Evaluation"])

# Load Dataset
data_raw = pd.read_csv("data/Impact_of_Remote_Work_on_Mental_Health.csv")
data = pd.read_csv("data/train_cleaned.csv")

# Home Page
if page == "Home":
    st.title("Welcome to the Remote Work on Mental Health Studies App! 🌎💻🧠")
    st.write("Explore insights derived from employee feedback among various industries. Dive into the data, discover trends, and visualize key metrics.")
    st.image("images/remote_work.jpeg")

# Overview of the Data
if page == "Overview":
    st.header("Overview of the Data 🔍")
    st.write("With 5,000 records collected from employees worldwide, this dataset provides valuable insights into key areas like work location (remote, hybrid, onsite), stress levels, access to mental health resources, and job satisfaction. It’s designed to help researchers, HR professionals, and businesses assess the growing influence of remote work on productivity and well-being.")

    tab1, tab2, tab3 = st.tabs(["Data Dictionary", "Data Types", "Sample Data"])
    with tab1:
        st.write("### Data Dictionary:")
        st.write("- `Employee_ID`: Unique identifier for each employee.")
        st.write("- `Age`: The actual age of the employees.")
        st.write("- `Gender`: Gender of the employees - Male, Female, Non-binary, Prefer not to say.")
        st.write("- `Job_Role`: The job role of the employees - includes roles like HR, Data Scientist, Software Engineer, etc.")
        st.write("- `Industry`: The industry in which the employees work - such as Healthcare, IT, Education, etc.")
        st.write("- `Years_of_Experience`: Total years of professional experience of the employees.")
        st.write("- `Work_Location`: The work arrangement of the employees - Hybrid, Remote, or Onsite.")
        st.write("- `Hours_Worked_Per_Week`: Average number of hours worked per week by the employees.")
        st.write("- `Number_of_Virtual_Meetings`: Average number of virtual meetings attended per week.")
        st.write("- `Work_Life_Balance_Rating`: Self-reported rating of work-life balance - High, Medium, Low.")
        st.write("- `Stress_Level`: Self-reported level of stress - High, Medium, Low.")
        st.write("- `Mental_Health_Condition`: Self-reported mental health condition - Depression, Anxiety, Burnout, or NaN (not disclosed).")
        st.write("- `Access_to_Mental_Health_Resources`: Indicates whether the employee has access to mental health resources - Yes or No.")
        st.write("- `Productivity_Change`: Change in productivity due to remote work - Decrease, Increase, or No Change.")
        st.write("- `Social_Isolation_Rating`: Self-reported rating of social isolation - ranging from 1 to 5.")
        st.write("- `Satisfaction_with_Remote_Work`: Employee satisfaction with remote work - Unsatisfied, Satisfied, or Neutral.")
        st.write("- `Company_Support_for_Remote_Work`: Level of company support for remote work - ranging from 1 to 5.")
        st.write("- `Physical_Activity`: Frequency of physical activity - Daily, Weekly, or NaN (not disclosed).")
        st.write("- `Sleep_Quality`: Self-reported sleep quality - Good, Average, or Poor.")
        st.write("- `Region`: Geographic region of the employees - Europe, Asia, North America, South America, Oceania, or Africa.")
    with tab2:
        st.write("### Data Types:")
        st.write(data_raw.dtypes)
    with tab3:
        st.write("### Sample Data:")
        st.write(data_raw.head())

# EDA Page
if page == "Exploratory Data Analysis":
    st.header("Exploratory Data Analysis (EDA) 📊")

    # Count Plot
    st.subheader("Work Location by Job Role")
    plt.figure(figsize = (10, 5))
    sns.countplot(data = data_raw, x = "Work_Location", hue = "Job_Role")
    plt.title("Distributions of Work Location by Job Role")
    plt.xlabel("Work Location")
    plt.ylabel("Count")
    plt.legend(title = "Job Role", bbox_to_anchor = (1.05, 1), loc = "upper left")
    st.pyplot(plt)
    container = st.container(border = True)
    container.write("The above bar graph displays the distribution of work location by job role. The graph shows that most employees are working remotely, followed by hybrid, and finally onsite with the least amount of employees. The distribution of job roles is evenly distributed within work locations.")

    # Histogram
    st.subheader("Hours Worked per Week")
    plt.figure(figsize = (10, 5))
    sns.histplot(data = data_raw, x = "Hours_Worked_Per_Week", bins = 20)
    plt.title("Distribution of Hours Worked per Week")
    plt.xlabel("Hours")
    plt.ylabel("Frequency")
    st.pyplot(plt)
    container = st.container(border = True)
    container.write("The histogram above shows the distribution of hours worked per week within the dataset, with the data evenly spread across most hours with a slight uptick near 60 hours worked per week.")

    # Scatter Plot
    st.subheader("Work Life Balance Rating vs Social Isolation Rating")
    plt.figure(figsize = (10, 5))
    sns.scatterplot(x = "Work_Life_Balance_Rating", y = "Social_Isolation_Rating", data = data_raw, hue = "Satisfaction_with_Remote_Work")
    plt.title("Work Life Balance Rating vs Social Isolation Rating by Satisfaction with Remote Work")
    plt.xlabel("Work Life Balance Rating")
    plt.ylabel("Social Isolation Rating")
    plt.legend(title = "Satisfaction with Remote Work", bbox_to_anchor = (1.05, 1), loc = "upper left")
    st.pyplot(plt)
    container = st.container(border = True)
    container.write("The above scatterplot displays the relationship between work life balance rating and social isolation rating by satisfaction with remote work. Although the data is scattered, there is a general trend associated with unsatisfied with remote work and a high social isolation rating along with a cluster of employees that rate their work life balance highly with a low social isolation rating being satisfied with remote work.")

    # Box Plot
    st.subheader("Hours Worked per Week by Gender")
    plt.figure(figsize = (10, 5))
    stress_order = ["Low", "Medium", "High"]
    sns.boxplot(x = "Gender", y = "Hours_Worked_Per_Week", data = data_raw, hue = "Stress_Level", hue_order = stress_order)
    plt.title("Distributions of Hours Worked per Week by Gender by Stress Level")
    plt.xlabel("Gender")
    plt.ylabel("Hours")
    plt.legend(title = "Stress Level", bbox_to_anchor = (1.05, 1), loc = "upper left")
    st.pyplot(plt)
    container = st.container(border = True)
    container.write("The side-by-side boxplots above illustrate the distribution of hours worked per week by gender and stress level. The data reveals that men tend to work more hours on average each week, with a significant number reporting medium to high stress levels.")

elif page == "Model Training and Evaluation":
    st.title("Model Training and Evaluation 🛠️")

    # Sidebar for Model Selection
    st.sidebar.subheader("Choose a Machine Learning Model")
    model_option = st.sidebar.selectbox("Select a Model", ["K-Nearest Neighbors", "Logistic Regression", "Random Forest"])

    # Prepare the Data
    X = data.drop(columns = "Stress_Level")
    y = data["Stress_Level"]

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, stratify = y)

    # Scale the Data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize the Selected Model
    if model_option == "K-Nearest Neighbors":
        k = st.sidebar.slider("Select the Number of Neighbors (k)", min_value = 1, max_value = 31, value = 2)
        model = KNeighborsClassifier(n_neighbors = k)
    elif model_option == "Logistic Regression":
        model = LogisticRegression()
    else:
        model = RandomForestClassifier()

    # Train the Model on the Scaled Data
    model.fit(X_train_scaled, y_train)

    # Display Training and Test Accuracy
    st.write(f"**Model Selected: {model_option}**")
    st.write(f"Training Accuracy: {model.score(X_train_scaled, y_train):.2f}")
    st.write(f"Test Accuracy: {model.score(X_test_scaled, y_test):.2f}")

    # Display Confusion Matrix
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_estimator(model, X_test_scaled, y_test, ax = ax, cmap = "Blues")
    st.pyplot(fig)
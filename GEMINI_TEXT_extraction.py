import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
from wordcloud import WordCloud
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.feature_selection import SelectKBest, f_regression
from scipy.stats import skew
import time
from PIL import Image
import base64
import numpy as np 
import os
from dotenv import load_dotenv
import google.generativeai as genai
import requests
from io import StringIO
from io import BytesIO
import  plotly.io as pio
import kaleido
from PIL import Image as PILImage
import matplotlib.pyplot as plt 


# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("models/gemini-1.5-flash-latest")

def get_response(input_text, image=None, max_retries=3, base_wait_time=2):
    retry_count = 0
    while retry_count < max_retries:
        try:
            if image:
                response = model.generate_content([input_text, image])
            else:
                response = model.generate_content(input_text)
            return response.text, response
        except Exception as e:
            retry_count += 1
            wait_time = base_wait_time * (2 ** (retry_count - 1))  # Exponential backoff
            if retry_count < max_retries:
                st.warning(f"Error generating response: {e}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                st.error(f"Error generating response after {max_retries} attempts: {e}")
                return "", None


def profile_data(data):
    st.write("### Data Profiling Summary")
    st.write(data.describe(include='all'))
    st.write("### Missing Values")
    st.write(data.isnull().sum())
    st.write("### Data Types")
    st.write(data.dtypes)

def generate_summary(data):
    input_text = f"Summarize the following data:\n{data.to_string()}"
    start_time = time.time()
    summary, _ = get_response(input_text)
    end_time = time.time()
    st.write(f"Time taken for Data Summary: {end_time - start_time:.2f} seconds")
    return summary

def generate_smart_insights(data):
    input_text = f""" 1.**Key Insights:** Identify and explain significant trends, correlations, and patterns within the data. Highlight any variables that have strong relationships or notable interactions.
     2.**Anomalies:** Detect and describe any anomalies or outliers in the data. Explain how these anomalies deviate from expected patterns and their potential impact on the analysis.
     3. **Interesting Patterns:** Point out any interesting or unexpected patterns in the data. This could include seasonal effects, cyclical trends, or unique group behaviors.
     4. **Data Quality Issues:** Note any potential data quality issues, such as missing values or inconsistencies, and suggest how they might affect the analysis.
     5. **Visualization Recommendations:** Suggest the most effective visualizations to represent these insights, anomalies, and patterns. Provide rationale for why these visualizations would be useful.:\n{data.to_string()}"""
    start_time = time.time()
    insights, _ = get_response(input_text)
    end_time = time.time()
    st.write(f"Time taken for Smart Insights: {end_time - start_time:.2f} seconds")
    return insights

def parse_dashboard_suggestions(dashboard_suggestions):
    try:
        suggestions_df = pd.read_csv(StringIO(dashboard_suggestions), sep='\t', skipinitialspace=True)
        return suggestions_df
    except Exception as e:
        st.error(f"Error parsing dashboard suggestions: {e}")
        return pd.DataFrame()
 #############################################################################################   

    ##########################################################################################################

# def identify_id_columns(data):
    
#     id_columns = []
#     for col in data.columns:
#         # Check if column name contains 'id' or if all values are unique
#         if data[col].nunique() == len(data) or any(keyword in col.lower() for keyword in ['id', 'identifier']):
#             id_columns.append(col)
#     return id_columns
# Define the custom color palette
custom_color_palette = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']

# Assuming get_response is a function that generates insights using the Gemini model


def plot_chart(data, chart_type, x_col, y_col=None, title=None, insights=""):
    col1, col2 = st.columns([3, 1])
    chart_height = 500
    chart_width = 700
    margin = dict(l=20, r=20, t=50, b=20)

    if isinstance(x_col, pd.Index):
        x_col = x_col[0]

    if title is None:
        title = f"{chart_type} for {x_col}" if x_col else chart_type

    with col1:
        fig = None
        if chart_type == "Histogram":
            fig = px.histogram(data, x=x_col, marginal="box", nbins=30, title=title, color_discrete_sequence=custom_color_palette)
        elif chart_type == "Box Plot":
            fig = px.box(data, y=x_col, title=title, color_discrete_sequence=custom_color_palette)
        elif chart_type == "Count Plot":
            fig = px.bar(data, x=x_col, title=title, color_discrete_sequence=custom_color_palette)
        elif chart_type == "Scatter Plot":
            if y_col is None:
                st.error("Please provide a y column for the Scatter Plot.")
                return
            fig = px.scatter(data, x=x_col, y=y_col, title=title, color_discrete_sequence=custom_color_palette)
        elif chart_type == "Bubble Chart":
            size_col = st.sidebar.selectbox(f"Select size column for Bubble Chart ({x_col}, {y_col} vs ...)", data.columns.tolist(), key=f"size_col_{x_col}_{y_col}")
            fig = px.scatter(data, x=x_col, y=y_col, size=size_col, title=title, color_discrete_sequence=custom_color_palette)
        elif chart_type == "Heatmap":
            fig = px.imshow(data.corr(), title=title, color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
        elif chart_type == "Pair Plot":
            sns.set(style="whitegrid", palette="muted")
            pairplot = sns.pairplot(data)
            pairplot.fig.set_size_inches(12, 8)
            st.pyplot(pairplot.fig)
            return  
        elif chart_type == "Bar Plot":
            fig = px.bar(data, x=x_col, y=y_col, title=title, color_discrete_sequence=custom_color_palette)
        elif chart_type == "Pie Chart":
            fig = px.pie(data, names=x_col, title=title, color_discrete_sequence=custom_color_palette)
        elif chart_type == "Line Chart":
            fig = px.line(data, x=x_col, y=y_col, title=title, color_discrete_sequence=custom_color_palette)
        elif chart_type == "Density Plot":
            fig = px.density_contour(data, x=x_col, title=title, color_discrete_sequence=custom_color_palette)
        elif chart_type == "Violin Plot":
            fig, ax = plt.subplots()
            sns.violinplot(data=data, y=x_col, ax=ax)
            ax.set_title(title)
            st.pyplot(fig)
            return
        elif chart_type == "Boxen Plot":
            fig, ax = plt.subplots()
            sns.boxenplot(data=data, y=x_col, ax=ax)
            ax.set_title(title)
            st.pyplot(fig)
            return
        else:
            st.error(f"Unsupported chart type: {chart_type}")
            return

        if fig:
            fig.update_layout(height=chart_height, width=chart_width, margin=margin)
            st.plotly_chart(fig)

#############
        
        
            buffer = BytesIO()
            pio.write_image(fig, buffer, format='png')
            buffer.seek(0)
    
            # Convert buffer to PIL Image
            image = PILImage.open(buffer)

            # Generate insights using the Gemini model
            start_time = time.time()
            image_insights, _ = get_response("Generate insights for the chart image in detail with  3 to 4 bullet points.", image=image)
            end_time = time.time()
            st.write(f"Time taken for Chart Insights: {end_time - start_time:.2f} seconds")
    
            # Display insights without showing the image
            with col1:
                with st.expander("Generated Insights", expanded=False):
                    st.write(image_insights)


        
            

        

def generate_charts_from_suggestions(data, dashboard_suggestions):
    suggestions_df = parse_dashboard_suggestions(dashboard_suggestions)
    
    for index, row in suggestions_df.iterrows():
        try:
            section = row.get('Section', '').strip()
            chart_type = row.get('Chart Type', '').strip()
            data_cols = row.get('Data', '').strip().split(',')
            description = row.get('Description', '').strip()
            
            if chart_type.lower() in ["bar plot", "line chart", "scatter plot", "histogram", "pie chart"]:
                if chart_type.lower() in ["bar plot", "line chart", "scatter plot"]:
                    if len(data_cols) >= 2:
                        x_col = data_cols[0].strip()
                        y_col = data_cols[1].strip()
                        plot_chart(data, chart_type, x_col=x_col, y_col=y_col, title=description)
                    else:
                        st.warning(f"Not enough data columns for a {chart_type} in section {section}.")
                
                elif chart_type.lower() == "histogram":
                    if len(data_cols) >= 1:
                        x_col = data_cols[0].strip()
                        plot_chart(data, chart_type, x_col=x_col, title=description)
                    else:
                        st.warning(f"Not enough data columns for a {chart_type} in section {section}.")
                
                elif chart_type.lower() == "pie chart":
                    if len(data_cols) >= 1:
                        x_col = data_cols[0].strip()
                        plot_chart(data, chart_type, x_col=x_col, title=description)
                    else:
                        st.warning(f"Not enough data columns for a {chart_type} in section {section}.")
            
        except Exception as e:
            st.error(f"Error processing row {index}: {e}")


def generate_dashboard_suggestions(data):
    input_text = f"""Based on the following data, please suggest a dashboard layout. Your suggestions should include:

    1. **Section**: The section of the dashboard where the chart will be placed (e.g., Overview, Sales Analysis).
    2. **Chart Type**: The type of chart to be used (e.g., Bar Plot, Line Chart, Histogram).
    3. **Data**: The columns of data to be used for the chart.
    4. **Filters**: Any relevant filters or conditions to apply to the data for this chart (e.g., Date Range, Category).
    5. **Description**: A brief description of the chart and its purpose.

    Please format your suggestions in a table with the following columns:
    - Section
    - Chart Type
    - Data
    - Filters
    - Description

    Here is the data you should base your suggestions on:
    {data.to_string()}"""
    start_time = time.time()
    dashboard_suggestions, _ = get_response(input_text)
    end_time = time.time()
    st.write(f"Time taken for Dashboard Suggestions: {end_time - start_time:.2f} seconds")
    return dashboard_suggestions

def analyze_time_series(data, column):
    decomposition = seasonal_decompose(data[column], model='additive', period=1)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    return trend, seasonal, residual

def descriptive_statistics(data):
    return data.describe()

def correlation_analysis(data):
    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    return numeric_data.corr()

def download_link(object_to_download, download_filename, download_link_text):
    if isinstance(object_to_download, pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)
    b64 = base64.b64encode(object_to_download.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'





def auto_generate_charts(data):
    # Function to clean columns if they contain lists
    def clean_column(col):
        if isinstance(col.iloc[0], list):
            return col.apply(lambda x: ','.join(map(str, x)) if isinstance(x, list) else x)
        return col

    # Clean the data
    data = data.apply(clean_column)

    # Identify potential ID columns (drop these)
    potential_id_columns = [col for col in data.columns if data[col].nunique() == len(data)]
    if potential_id_columns:
        data = data.drop(columns=potential_id_columns)

    # Split columns by data type
    numeric_cols = data.select_dtypes(include=['number', 'float']).columns.tolist()
    categorical_cols = data.select_dtypes(include=['bool', 'category', 'object']).columns.tolist()

    # Identify time-related columns
    def identify_time_columns(data):
        time_cols = []
        for col in data.columns:
            # Check if the column name suggests it's time-related
            if "time" in col.lower() or "date" in col.lower() or "year" in col.lower() or "month" in col.lower() or "day" in col.lower():
                time_cols.append(col)
                continue

            # Check if the column's data type is datetime or can be converted to datetime
            if pd.api.types.is_datetime64_any_dtype(data[col]):
                time_cols.append(col)
                continue
            
            try:
                # Try converting to datetime and check if most values are valid dates
                converted_col = pd.to_datetime(data[col], errors='coerce')
                if converted_col.notna().sum() / len(converted_col) > 0.8:  # 80%+ valid dates
                    time_cols.append(col)
            except:
                continue

        return time_cols
    
    time_cols = identify_time_columns(data)

    if not numeric_cols and not categorical_cols:
        print("No suitable columns available for chart generation.")
        return []

    charts = []
    used_charts = set()

    # Generate charts for numerical columns
    for col in numeric_cols:
        # Histogram
        if ("Histogram", col) not in used_charts:
            charts.append(("Histogram", col))
            used_charts.add(("Histogram", col))
        
        # Box Plot
        if ("Box Plot", col) not in used_charts:
            charts.append(("Box Plot", col))
            used_charts.add(("Box Plot", col))
        
        # Density Plot
        if ("Density Plot", col) not in used_charts:
            charts.append(("Density Plot", col))
            used_charts.add(("Density Plot", col))

    # Generate line charts for time-related data
    for col in time_cols:
        if ("Line Chart", col) not in used_charts:
            charts.append(("Line Chart", col))
            used_charts.add(("Line Chart", col))

    # Generate charts for categorical columns
    for col in categorical_cols:
        # Count Plot
        if ("Count Plot", col) not in used_charts:
            charts.append(("Count Plot", col))
            used_charts.add(("Count Plot", col))

        # Pie Chart (only if few unique categories)
        if data[col].nunique() <= 10:
            if ("Pie Chart", col) not in used_charts:
                charts.append(("Pie Chart", col))
                used_charts.add(("Pie Chart", col))
        
        # Bar Plot
        if ("Bar Plot", col) not in used_charts:
            charts.append(("Bar Plot", col))
            used_charts.add(("Bar Plot", col))

    # Generate pairwise plots for numeric columns
    if len(numeric_cols) > 1:
        # Scatter Plot
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                if ("Scatter Plot", numeric_cols[i], numeric_cols[j]) not in used_charts:
                    charts.append(("Scatter Plot", numeric_cols[i], numeric_cols[j]))
                    used_charts.add(("Scatter Plot", numeric_cols[i], numeric_cols[j]))

        # Correlation Heatmap
        if ("Heatmap", None) not in used_charts:
            charts.append(("Heatmap", None))
            used_charts.add(("Heatmap", None))

        # Pair Plot
        if ("Pair Plot", None) not in used_charts:
            charts.append(("Pair Plot", None))
            used_charts.add(("Pair Plot", None))

    # Generate a Bubble Chart if 3+ numeric columns
    if len(numeric_cols) > 2:
        if ("Bubble Chart", tuple(numeric_cols[:3])) not in used_charts:
            charts.append(("Bubble Chart", tuple(numeric_cols[:3])))
            used_charts.add(("Bubble Chart", tuple(numeric_cols[:3])))

    return charts[:20]



#####################

def compare_insights(expert_insights, generated_insights):
    expert_points = [line.strip() for line in expert_insights.split("\n") if line.strip()]
    generated_points = [line.strip() for line in generated_insights.split("\n") if line.strip()]

    max_length = max(len(expert_points), len(generated_points))
    expert_points.extend([""] * (max_length - len(expert_points)))
    generated_points.extend([""] * (max_length - len(generated_points)))
    
    comparison_df = pd.DataFrame({
        "Expert Insight": expert_points,
        "Generated Insight": generated_points
    })

    return comparison_df

def fetch_data_from_api(api_url, params=None):
    try:
        response = requests.get(api_url, params=params)
        response.raise_for_status()
        data = response.json()
        df = pd.json_normalize(data)
        return df
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data from API: {e}")
        return pd.DataFrame()
    


def clean_data(data):
    
    data.replace(['N.A.', 'N/A', 'n.a.', 'na', 'NaN'], np.nan, inplace=True)
    
    # Iterate through columns to handle missing values based on data type and context
    for col in data.columns:
        if data[col].dtype == 'object':
            # Check if the column contains numeric values by trying to convert to numeric, preserving non-numeric entries
            if pd.to_numeric(data[col].str.replace(',', ''), errors='coerce').notna().all():
                data[col] = pd.to_numeric(data[col].str.replace(',', ''), errors='coerce')
            else:
                # For categorical data, fill missing values with the most frequent value (mode)
                data[col].fillna(data[col].mode()[0], inplace=True)
        
        # Fill numeric columns with the mean of the column
        elif data[col].dtype in ['float64', 'int64']:
            data[col].fillna(data[col].mean(), inplace=True)
    
    # Drop columns where more than 50% of the values are missing
    data.dropna(axis=1, thresh=int(0.8 * len(data)), inplace=True)
    
    # Drop rows where more than 50% of the values are missing
    data.dropna(axis=0, thresh=int(0.8 * len(data.columns)), inplace=True)

    return data




# st.write("Dashboard Suggestions DataFrame:", suggestions_df)

st.set_page_config(page_title="Data Insight Wizard", layout="wide")
st.header('Data Insight Wizard', divider='rainbow')

st.sidebar.header("Data & Image Analysis")
selection = st.sidebar.selectbox("Select Analysis Type", ["Image Insights", "Data Analysis"])

if selection == "Image Insights":
    uploaded_images = st.sidebar.file_uploader("Upload images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    
    if uploaded_images:
        for i, uploaded_image in enumerate(uploaded_images):
            with st.expander(f"Image Insights {i+1}", expanded=False):
                st.image(uploaded_image, caption=f"Uploaded Image {i+1}", use_column_width=True)
                image = Image.open(uploaded_image)
                start_time = time.time()
                summary, _ = get_response("Generate insights for the uploaded image.", image)
                end_time = time.time()
                st.write(f"Time taken for Image Insights {i+1}: {end_time - start_time:.2f} seconds")
                st.write(f"### Image Insights {i+1}:")
                st.write(summary)


elif selection == "Data Analysis":
    data_source = st.sidebar.radio("Select Data Source", ["Upload File","Multiple Files", "Fetch from API"])
    data = pd.DataFrame()
    
    if data_source == "Upload File":
        uploaded_file = st.sidebar.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])
    
        if uploaded_file is not None:
            if uploaded_file.name.endswith(".csv"):
                data = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(".xlsx"):
                data = pd.read_excel(uploaded_file, engine='openpyxl')
            else:
                st.warning("Please upload a CSV or Excel file.")
        else:
            st.info("No file uploaded yet.")
    
    elif data_source == "Fetch from API":
        api_url = st.sidebar.text_input("Enter API URL", "")
        if api_url:
            params = st.sidebar.text_input("Enter API Parameters (JSON format)", "{}")
            params = eval(params) if params else None
            data = fetch_data_from_api(api_url, params)
            
    elif data_source == "Multiple Files":
        uploaded_files = st.sidebar.file_uploader("Choose files", type=["csv", "xlsx"], accept_multiple_files=True)
    
        if len(uploaded_files) == 2:
            data1 = None
            data2 = None
            
            
            if uploaded_files[0].name.endswith('.csv'):
                data1 = pd.read_csv(uploaded_files[0])
            elif uploaded_files[0].name.endswith('.xlsx'):
                data1 = pd.read_excel(uploaded_files[0])
            
            if uploaded_files[1].name.endswith('.csv'):
                data2 = pd.read_csv(uploaded_files[1])
            elif uploaded_files[1].name.endswith('.xlsx'):
                data2 = pd.read_excel(uploaded_files[1])
            
            if data1 is not None and data2 is not None:
                merge_method = st.sidebar.selectbox("Select Merge Method", ["Concatenate", "Merge on Key"])
                
                if merge_method == "Merge on Key":
                    common_columns = data1.columns.intersection(data2.columns).tolist()
                    if common_columns:
                        selected_keys = st.sidebar.multiselect("Select Merge Keys", common_columns)
                        st.write("### Selected Merge Keys:")
                        st.write(selected_keys)
                        if selected_keys:
                            if st.sidebar.button("Submit"):
                                try:
                                    data = pd.merge(data1, data2, on=selected_keys)
                                    st.write("### Merged Data:")
                                    st.write(data.head())
                                except Exception as e:
                                    st.error(f"Error merging data: {e}")
                        else:
                            st.info("Please select at least one key for merging.")
                    
                    # data = pd.concat([data1, data2], ignore_index=True)
                    # st.write("### Concatenated Data:")
                    # st.write(data.head())
                    
                elif merge_method == "Concatenate":
                    if st.sidebar.button("Submit"):
                        data = pd.concat([data1, data2], ignore_index=True)
                        st.write("### Concatenated Data:")
                        st.write(data.head())
                            
            else:
                st.error("Both files need to be uploaded.")
        else:
            st.info("Please upload exactly two files for merging.")
                    
                    
            
    # If not empty, clean the data and then proceed
    if not data.empty:
        st.write("### Uploaded Data:")
        
        # Clean the data
        data = clean_data(data)
        
        st.write(data.head())

        profile_data(data)
        
        st.subheader("Descriptive Statistics")
        descriptive_stats = descriptive_statistics(data)
        st.write(descriptive_stats)
        st.markdown(download_link(descriptive_stats, "descriptive_statistics.csv", "Download Descriptive Statistics"), unsafe_allow_html=True)

        st.subheader("Correlation Analysis")
        correlation_matrix = correlation_analysis(data)
        st.write(correlation_matrix)
        st.markdown(download_link(correlation_matrix, "correlation_analysis.csv", "Download Correlation Analysis"), unsafe_allow_html=True)

        st.subheader("Smart Insights")
        insights = generate_smart_insights(data)
        st.write(insights)

        st.subheader("Auto-generated Charts")
        chart_suggestions = auto_generate_charts(data)
        for chart_type, col in chart_suggestions:
            title = f"{chart_type} for {col}" if col else chart_type
            chart_data = data[[col] + ([col] if col else [])].dropna() if col else data
            insights = generate_smart_insights(chart_data)
            plot_chart(data, chart_type, col, title=title, insights=insights)

        st.subheader("Dashboard Suggestions")
        dashboard_suggestions = generate_dashboard_suggestions(data)
        st.write(dashboard_suggestions)


        # st.subheader("Charts from Dashboard Suggestions")
        # generate_charts_from_suggestions(data, dashboard_suggestions)

        st.sidebar.header("Expert Dashboard Comparison")
        uploaded_expert_dashboard = st.sidebar.file_uploader("Upload an Expert Dashboard (Image, Power BI, Tableau, CSV, Excel)", type=["png", "jpg", "jpeg", "pbix", "twbx", "csv", "xlsx"])

        if uploaded_expert_dashboard:
            if uploaded_expert_dashboard.name.endswith((".png", ".jpg", ".jpeg")):
                st.image(uploaded_expert_dashboard, caption="Expert Dashboard", use_column_width=True)
                expert_dashboard_insights, _ = get_response("Generate insights for the uploaded expert dashboard image.", Image.open(uploaded_expert_dashboard))
            elif uploaded_expert_dashboard.name.endswith(".csv"):
                expert_dashboard_data = pd.read_csv(uploaded_expert_dashboard)
                expert_dashboard_insights = generate_summary(expert_dashboard_data)
            elif uploaded_expert_dashboard.name.endswith(".xlsx"):
                expert_dashboard_data = pd.read_excel(uploaded_expert_dashboard, engine='openpyxl')
                expert_dashboard_insights = generate_summary(expert_dashboard_data)

            st.write("### Expert Dashboard Insights:")
            st.write(expert_dashboard_insights)

            st.subheader("Comparison of Insights")
            comparison_df = compare_insights(expert_dashboard_insights, insights)
            st.write(comparison_df)
            st.markdown(download_link(comparison_df, "insight_comparison.csv", "Download Insight Comparison"), unsafe_allow_html=True)
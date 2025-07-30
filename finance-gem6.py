# Lastest worked 
# Support THAI text
import streamlit as st
import pandas as pd
from prophet import Prophet
import xgboost as xgb
import ollama
import openai # For OpenAI GPT API
from google.generativeai import GenerativeModel # For Google Gemini API
from google.generativeai.types import HarmCategory, HarmBlockThreshold # For safety settings
import matplotlib.pyplot as plt
import seaborn as sns
import io
import matplotlib.font_manager as fm # For Thai font handling
import os # For checking file existence
import requests # For downloading font

# --- Thai Font Configuration for Matplotlib ---
# กำหนด URL และชื่อไฟล์ฟอนต์
font_url = "https://github.com/google/fonts/raw/main/ofl/kanit/Kanit-Regular.ttf"
font_filename = "Kanit-Regular.ttf"
# ใช้ os.path.join เพื่อให้แน่ใจว่า path ถูกต้องบนทุก OS
font_path = os.path.join(os.getcwd(), font_filename) 

st.sidebar.subheader("สถานะการตั้งค่า Font ภาษาไทย")

# ใช้ st.cache_resource เพื่อให้ดาวน์โหลดเพียงครั้งเดียว
@st.cache_resource(ttl=3600 * 24 * 7) # Cache for 1 week
def download_font_once(url, filename, path):
    if not os.path.exists(path):
        st.sidebar.info(f"กำลังดาวน์โหลดฟอนต์ {filename}...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status() # ตรวจสอบว่า request สำเร็จ
            with open(path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            st.sidebar.success(f"ดาวน์โหลด {filename} สำเร็จ.")
            return True
        except requests.exceptions.RequestException as e:
            st.sidebar.error(f"ไม่สามารถดาวน์โหลด {filename} ได้: {e}. โปรดตรวจสอบการเชื่อมต่ออินเทอร์เน็ต.")
            return False
    else:
        st.sidebar.info(f"ฟอนต์ {filename} มีอยู่แล้ว.")
        return True

# เรียกใช้ฟังก์ชันดาวน์โหลด
font_downloaded = download_font_once(font_url, font_filename, font_path)

thai_font_name = "Kanit" # ชื่อ Font Family ที่คาดหวัง

if font_downloaded:
    try:
        # เพิ่มฟอนต์ลงใน Matplotlib font manager
        # นี่คือขั้นตอนสำคัญที่บอก Matplotlib ว่ามีฟอนต์นี้อยู่
        fm.fontManager.addfont(font_path)
        
        # กำหนดให้ Matplotlib ใช้ฟอนต์ Kanit เป็นค่าเริ่มต้น
        plt.rcParams['font.family'] = thai_font_name
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.unicode_minus'] = False # แก้ปัญหาเครื่องหมายลบเป็นสี่เหลี่ยม

        # ตรวจสอบว่าฟอนต์ถูกโหลดและสามารถใช้งานได้จริง
        # fm.findfont จะ return path ของ font ถ้าพบ, ไม่ใช่ชื่อ font
        if fm.findfont(thai_font_name, fontext='ttf'):
            kanit_font_prop = fm.FontProperties(fname=font_path)
            st.sidebar.success(f"ตั้งค่าฟอนต์ '{thai_font_name}' สำเร็จ! (ชื่อฟอนต์ที่ Matplotlib รู้จัก: '{kanit_font_prop.get_name()}')")
        else:
            st.sidebar.warning(f"Font '{thai_font_name}' ยังไม่ถูกรับรู้โดย Matplotlib. อาจแสดงผลภาษาไทยไม่ถูกต้อง.")
            st.sidebar.info("ใช้ Font สำรอง 'DejaVu Sans'.")
            plt.rcParams['font.family'] = 'DejaVu Sans' # Fallback
            plt.rcParams['font.size'] = 12
            plt.rcParams['axes.unicode_minus'] = False
            kanit_font_prop = None # กำหนดให้เป็น None หากเกิดข้อผิดพลาด
        
    except Exception as e:
        st.sidebar.error(f"เกิดข้อผิดพลาดในการตั้งค่า Font '{thai_font_name}': {e}. ใช้ Font สำรอง 'DejaVu Sans'.")
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.unicode_minus'] = False
        kanit_font_prop = None # กำหนดให้เป็น None หากเกิดข้อผิดพลาด
else:
    st.sidebar.error(f"ไม่สามารถดาวน์โหลดฟอนต์ '{thai_font_name}' ได้. ใช้ Font สำรอง 'DejaVu Sans'.")
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.unicode_minus'] = False
    kanit_font_prop = None # กำหนดให้เป็น None หากเกิดข้อผิดพลาด

# ... โค้ดส่วนที่เหลือของคุณ (st.set_page_config และอื่น ๆ) ...
# --- Streamlit Page Configuration ---
st.set_page_config(layout="wide", page_title="Real Estate Sales AI Assistant")

st.title("🏡 Sales AI Assistant")
st.write("เครื่องมือช่วยวิเคราะห์การขายโครงการอสังหาริมทรัพย์ด้วย AI")

# --- Sidebar for Settings and Data Folder Path ---
st.sidebar.header("⚙️ การตั้งค่าและข้อมูล")

# NEW: AI Model Provider Selection
ai_provider = st.sidebar.selectbox(
    "เลือก AI Model Provider:",
    ["Ollama (Local Llama)", "OpenAI (GPT)", "Google AI (Gemini)"]
)

if ai_provider == "Ollama (Local Llama)":
    ollama_base_url = st.sidebar.text_input(
        "URL ของ Ollama Server:",
        "http://10.10.32.78:11434" # Default value
    )
    ollama_model_name = st.sidebar.text_input(
        "ชื่อโมเดล Ollama (เช่น llama3.1:8b):",
        "llama3.1:8b" # Default value
    )
    # Set default values for other providers to avoid errors later
    openai_api_key = None
    openai_model_name = None
    google_api_key = None
    google_model_name = None

elif ai_provider == "OpenAI (GPT)":
    openai_api_key = st.sidebar.text_input("OpenAI API Key:", type="password")
    openai_model_name = st.sidebar.text_input(
        "ชื่อโมเดล OpenAI (เช่น gpt-3.5-turbo, gpt-4o):",
        "gpt-4o" # Default to a capable model
    )
    # Set default values for other providers
    ollama_base_url = None
    ollama_model_name = None
    google_api_key = None
    google_model_name = None

elif ai_provider == "Google AI (Gemini)":
    google_api_key = st.sidebar.text_input("Google AI API Key:", type="password")
    google_model_name = st.sidebar.text_input(
        "ชื่อโมเดล Google Gemini (เช่น gemini-pro, gemini-1.5-flash):",
        "gemini-1.5-flash" # Default to a capable model
    )
    # Set default values for other providers
    ollama_base_url = None
    ollama_model_name = None
    openai_api_key = None
    openai_model_name = None

# Data folder path input (เหมือนเดิม)
data_folder = st.sidebar.text_input(
    "ระบุชื่อโฟลเดอร์ข้อมูล (เช่น data/ หรือ /path/to/data/):",
    "data/" # Default value
)


# --- Data Loading & Preprocessing Functions ---

# Helper function to check file existence
def check_file_exists(file_path):
    if not os.path.exists(file_path):
        st.sidebar.error(f"ไม่พบไฟล์: {file_path}. โปรดตรวจสอบว่าโฟลเดอร์ข้อมูลและชื่อไฟล์ถูกต้อง.")
        return False
    return True

@st.cache_data
def load_and_preprocess_sales_data(file_path):
    """Loads and preprocesses the main sales data."""
    if not check_file_exists(file_path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date']).sort_values(by='Date') # Drop rows with invalid dates
        df['Units_Sold'] = pd.to_numeric(df['Units_Sold'], errors='coerce').fillna(0)
        df['Sales_Revenue'] = pd.to_numeric(df['Sales_Revenue'], errors='coerce').fillna(0)

        # Feature Engineering for sales_df itself (before merge)
        df['month'] = df['Date'].dt.month
        df['day_of_week'] = df['Date'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Lag features (ensure Project_ID is present for groupby)
        if 'Project_ID' in df.columns:
            df['sales_lag_7'] = df.groupby('Project_ID')['Units_Sold'].shift(7).fillna(0)
            df['sales_lag_30'] = df.groupby('Project_ID')['Units_Sold'].shift(30).fillna(0)
        else:
            df['sales_lag_7'] = 0
            df['sales_lag_30'] = 0

        # Convert categorical features from sales_history to numerical (if they exist)
        for col_name in ['Promotion_Applied', 'Lead_Source', 'Marketing_Channel']:
            if col_name in df.columns:
                df[f'{col_name}_encoded'] = df[col_name].astype('category').cat.codes
            else:
                df[f'{col_name}_encoded'] = 0 # Default if column is missing

        # Fill NaNs for numeric columns expected from sales_history.csv for XGBoost
        numeric_cols_from_sales = [
            'Avg_Competitor_Price_Nearby', 'Interest_Rate_Change_Indicator',
            'Nearby_Infrastructure_Dev_Score', 'Economic_Sentiment_Index'
        ]
        for col in numeric_cols_from_sales:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].mean() if not df[col].isnull().all() else 0)
            else:
                df[col] = 0 # Create and fill with 0 if missing in the uploaded CSV

        st.sidebar.success("โหลด sales_history.csv และประมวลผลแล้ว")
        return df
    except Exception as e:
        st.sidebar.error(f"Error loading or preprocessing Sales History file from {file_path}: {e}")
        st.exception(e)
        return pd.DataFrame()

@st.cache_data
def load_project_details(file_path):
    """Loads and preprocesses project details data."""
    if not check_file_exists(file_path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(file_path)
        df['Total_Units'] = pd.to_numeric(df['Total_Units'], errors='coerce').fillna(0)
        df['Units_Completed'] = pd.to_numeric(df['Units_Completed'], errors='coerce').fillna(0)
        df['Units_Sold_toDate'] = pd.to_numeric(df['Units_Sold_toDate'], errors='coerce').fillna(0)
        df['Proximity_to_BTS_MRT_Km'] = pd.to_numeric(df['Proximity_to_BTS_MRT_Km'], errors='coerce').fillna(df['Proximity_to_BTS_MRT_Km'].mean() if not df['Proximity_to_BTS_MRT_Km'].isnull().all() else 0)
        for col in ['Land_Cost_Per_SqM', 'Construction_Cost_Per_SqM', 'Average_Selling_Price_Per_SqM_Initial',
                    'Amenities_Score', 'Green_Space_Ratio', 'Parking_Ratio', 'Marketing_Budget_Initial']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].mean() if not df[col].isnull().all() else 0)
            else:
                df[col] = 0
        st.sidebar.success("โหลด project_details.csv และประมวลผลแล้ว")
        return df
    except Exception as e:
        st.sidebar.error(f"Error loading or preprocessing Project Details file from {file_path}: {e}")
        st.exception(e)
        return pd.DataFrame()

@st.cache_data
def load_macro_economic_data(file_path):
    """Loads and preprocesses macroeconomic data."""
    if not check_file_exists(file_path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date']).sort_values(by='Date') # Drop rows with invalid dates
        numeric_cols = ['Interest_Rate_Percent', 'Inflation_Rate_Percent', 'GDP_Growth_Rate_Percent',
                        'Consumer_Confidence_Index', 'Unemployment_Rate_Percent',
                        'Foreign_Investment_Index', 'Housing_Price_Index_Nationwide']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].mean() if not df[col].isnull().all() else 0)
            else:
                df[col] = 0 # Create and fill with 0 if missing
        st.sidebar.success("โหลด macro_economic_data.csv และประมวลผลแล้ว")
        return df
    except Exception as e:
        st.sidebar.error(f"Error loading or preprocessing Macroeconomic Indicators file from {file_path}: {e}")
        st.exception(e)
        return pd.DataFrame()

@st.cache_data
def load_marketing_activities(file_path):
    """Loads and preprocesses marketing activities data."""
    if not check_file_exists(file_path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(file_path)
        df['Activity_Date'] = pd.to_datetime(df['Activity_Date'], errors='coerce')
        df = df.dropna(subset=['Activity_Date']).sort_values(by='Activity_Date')
        df['Budget_Baht'] = pd.to_numeric(df['Budget_Baht'], errors='coerce').fillna(0)
        df['Leads_Generated'] = pd.to_numeric(df['Leads_Generated'], errors='coerce').fillna(0)
        st.sidebar.success("โหลด marketing_activities.csv และประมวลผลแล้ว")
        return df
    except Exception as e:
        st.sidebar.error(f"Error loading or preprocessing Marketing Activities file from {file_path}: {e}")
        st.exception(e)
        return pd.DataFrame()

@st.cache_data
def load_sales_funnel_log(file_path):
    """Loads and preprocesses sales funnel log data."""
    if not check_file_exists(file_path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(file_path)
        df['Lead_Date'] = pd.to_datetime(df['Lead_Date'], errors='coerce')
        df['Status_Change_Date'] = pd.to_datetime(df['Status_Change_Date'], errors='coerce')
        df = df.dropna(subset=['Lead_Date'])
        df['Is_Converted_to_Sale'] = df['Is_Converted_to_Sale'].astype(bool)
        st.sidebar.success("โหลด sales_funnel_log.csv และประมวลผลแล้ว")
        return df
    except Exception as e:
        st.sidebar.error(f"Error loading or preprocessing Sales Funnel Log file from {file_path}: {e}")
        st.exception(e)
        return pd.DataFrame()

@st.cache_data
def load_competitor_data(file_path):
    """Loads and preprocesses competitor data."""
    if not check_file_exists(file_path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date']).sort_values(by='Date')
        df['Avg_Price_Per_SqM'] = pd.to_numeric(df['Avg_Price_Per_SqM'], errors='coerce').fillna(df['Avg_Price_Per_SqM'].mean() if not df['Avg_Price_Per_SqM'].isnull().all() else 0)
        df['Sales_Volume_Estimate'] = pd.to_numeric(df['Sales_Volume_Estimate'], errors='coerce').fillna(0)
        st.sidebar.success("โหลด competitor_analysis.csv และประมวลผลแล้ว")
        return df
    except Exception as e:
        st.sidebar.error(f"Error loading or preprocessing Competitor Data file from {file_path}: {e}")
        st.exception(e)
        return pd.DataFrame()

@st.cache_data
def load_and_preprocess_feedback_data(file_path):
    """Loads and preprocesses customer feedback data."""
    if not check_file_exists(file_path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(file_path)
        if 'feedback_text' not in df.columns:
            st.warning("Column 'feedback_text' not found in feedback data. Skipping feedback analysis.")
            return pd.DataFrame() # Return empty if essential column is missing
        st.sidebar.success("โหลด customer_feedback.csv และประมวลผลแล้ว")
        return df
    except Exception as e:
        st.sidebar.error(f"Error loading or preprocessing Customer Feedback file from {file_path}: {e}")
        st.exception(e)
        return pd.DataFrame()


# --- Load all DataFrames from specified folder ---
st.sidebar.subheader("สถานะการโหลดไฟล์")

sales_df = load_and_preprocess_sales_data(os.path.join(data_folder, 'sales_history.csv'))
project_details_df = load_project_details(os.path.join(data_folder, 'project_details.csv'))
macro_economic_df = load_macro_economic_data(os.path.join(data_folder, 'macro_economic_data.csv'))
marketing_activities_df = load_marketing_activities(os.path.join(data_folder, 'marketing_activities.csv'))
sales_funnel_df = load_sales_funnel_log(os.path.join(data_folder, 'sales_funnel_log.csv'))
competitor_df = load_competitor_data(os.path.join(data_folder, 'competitor_analysis.csv'))
feedback_df = load_and_preprocess_feedback_data(os.path.join(data_folder, 'customer_feedback.csv'))


# --- Data Merging Section ---
st.subheader("🔄 สถานะการรวมข้อมูล (Data Merging Status)")

# 1. Merge Project Details into sales_df
if not sales_df.empty and not project_details_df.empty:
    st.info("กำลังพยายามรวม Sales Data กับ Project Details...")
    sales_df['Project_ID'] = sales_df['Project_ID'].astype(str)
    project_details_df['Project_ID'] = project_details_df['Project_ID'].astype(str)

    sales_df = pd.merge(sales_df, project_details_df, on='Project_ID', how='left')

    project_cols_expected = [
        'Project_Type', 'Location_District', 'Target_Audience', 'Project_Status',
        'Total_Units', 'Units_Completed', 'Units_Sold_toDate', 'Project_Size_Acres',
        'Proximity_to_BTS_MRT_Km', 'Proximity_to_Shopping_Km', 'Proximity_to_School_Km',
        'Land_Cost_Per_SqM', 'Construction_Cost_Per_SqM', 'Average_Selling_Price_Per_SqM_Initial',
        'Amenities_Score', 'Green_Space_Ratio', 'Parking_Ratio', 'Marketing_Budget_Initial'
    ]
    for col in project_cols_expected:
        if col not in sales_df.columns:
            if col in ['Project_Type', 'Location_District', 'Target_Audience', 'Project_Status']:
                sales_df[col] = 'Unknown'
            else:
                sales_df[col] = 0
        else:
            if sales_df[col].dtype == 'object':
                sales_df[col] = sales_df[col].fillna('Unknown')
            else:
                sales_df[col] = sales_df[col].fillna(0)

    st.success("รวม Project Details เข้ากับ Sales Data เสร็จสิ้น.")
    st.dataframe(sales_df[['Project_ID', 'Project_Type', 'Total_Units', 'Proximity_to_BTS_MRT_Km']].head())
else:
    st.warning("ไม่สามารถรวม Project Details ได้: Sales Data หรือ Project Details ยังว่างเปล่า. การวิเคราะห์บางส่วนอาจไม่สมบูรณ์.")
    for col in ['Project_Type', 'Total_Units', 'Proximity_to_BTS_MRT_Km', 'Amenities_Score', 'Marketing_Budget_Initial']:
        if col not in sales_df.columns:
            if col == 'Project_Type':
                sales_df[col] = 'Unknown'
            else:
                sales_df[col] = 0

# 2. Merge Macro Economic Data into sales_df
if not sales_df.empty and not macro_economic_df.empty:
    st.info("กำลังพยายามรวม Sales Data กับ Macroeconomic Data...")
    sales_df['Date_Month'] = sales_df['Date'].dt.to_period('M').dt.to_timestamp('M')
    macro_economic_df['Date_Month'] = macro_economic_df['Date'].dt.to_period('M').dt.to_timestamp('M')

    sales_df = pd.merge(sales_df, macro_economic_df.drop(columns=['Date']), on='Date_Month', how='left')
    sales_df.drop(columns=['Date_Month'], inplace=True) # Drop the temporary merge column

    macro_cols_expected = [
        'Interest_Rate_Percent', 'Inflation_Rate_Percent', 'GDP_Growth_Rate_Percent',
        'Consumer_Confidence_Index', 'Unemployment_Rate_Percent',
        'Foreign_Investment_Index', 'Housing_Price_Index_Nationwide'
    ]
    for col in macro_cols_expected:
        if col not in sales_df.columns:
            sales_df[col] = 0
        else:
            sales_df[col] = sales_df[col].fillna(0)

    st.success("รวม Macroeconomic Data เข้ากับ Sales Data เสร็จสิ้น.")
    st.dataframe(sales_df[['Date', 'Interest_Rate_Percent', 'Consumer_Confidence_Index']].head())
else:
    st.warning("ไม่สามารถรวม Macroeconomic Data ได้: Sales Data หรือ Macroeconomic Data ยังว่างเปล่า. การพยากรณ์อาจไม่แม่นยำเท่าที่ควร.")
    for col in ['Interest_Rate_Percent', 'Inflation_Rate_Percent', 'GDP_Growth_Rate_Percent', 'Consumer_Confidence_Index']:
        if col not in sales_df.columns:
            sales_df[col] = 0

st.subheader("ข้อมูลการขาย (Sales Data) หลังการรวมทั้งหมด")
if not sales_df.empty:
    st.dataframe(sales_df.head())
    st.write(f"Sales Data มี {len(sales_df)} แถว และ {len(sales_df.columns)} คอลัมน์.")
    st.write("คอลัมน์ที่พร้อมใช้งาน:", sales_df.columns.tolist())
else:
    st.error("Sales Data ยังว่างเปล่า กรุณาตรวจสอบการตั้งค่าโฟลเดอร์ข้อมูลและไฟล์ CSV.")

# --- Sales Forecasting Section (Moved after all merges) ---
st.header("📊 การพยากรณ์ยอดขาย")
if not sales_df.empty:
    st.write("แสดงข้อมูลการขายจริงและการพยากรณ์จากโมเดล Prophet + XGBoost")

    prophet_df = sales_df.groupby('Date')['Units_Sold'].sum().reset_index()
    prophet_df.rename(columns={'Date': 'ds', 'Units_Sold': 'y'}, inplace=True)

    if len(prophet_df) > 1: # Prophet needs at least 2 data points
        try:
            m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
            m.fit(prophet_df)

            future_periods = st.slider("จำนวนวันในอนาคตที่ต้องการพยากรณ์:", 7, 90, 30)
            future = m.make_future_dataframe(periods=future_periods)
            forecast_prophet = m.predict(future)

            daily_sales_features = sales_df.groupby('Date').agg({
                'month': 'first',
                'day_of_week': 'first',
                'is_weekend': 'first',
                'sales_lag_7': 'sum',
                'sales_lag_30': 'sum',
                'Avg_Competitor_Price_Nearby': 'mean',
                'Interest_Rate_Change_Indicator': 'mean',
                'Nearby_Infrastructure_Dev_Score': 'mean',
                'Economic_Sentiment_Index': 'mean',
                'Total_Units': 'sum', # From Project Details
                'Proximity_to_BTS_MRT_Km': 'mean', # From Project Details
                'Amenities_Score': 'mean', # From Project Details
                'Marketing_Budget_Initial': 'sum', # From Project Details
                'Inflation_Rate_Percent': 'mean', # From Macro Economic
                'GDP_Growth_Rate_Percent': 'mean', # From Macro Economic
                'Consumer_Confidence_Index': 'mean', # From Macro Economic
                'Units_Sold': 'sum' # Actual sales for training residuals
            }).reset_index()

            daily_sales_features = daily_sales_features.merge(
                forecast_prophet[['ds', 'yhat']],
                left_on='Date', right_on='ds', how='left'
            )
            daily_sales_features['prophet_yhat'] = daily_sales_features['yhat']
            daily_sales_features['residuals'] = daily_sales_features['Units_Sold'] - daily_sales_features['prophet_yhat']

            xgb_features = [
                'month', 'day_of_week', 'is_weekend',
                'sales_lag_7', 'sales_lag_30',
                'Avg_Competitor_Price_Nearby', 'Interest_Rate_Change_Indicator',
                'Nearby_Infrastructure_Dev_Score', 'Economic_Sentiment_Index',
                'Total_Units', 'Proximity_to_BTS_MRT_Km', 'Amenities_Score', 'Marketing_Budget_Initial',
                'Inflation_Rate_Percent', 'GDP_Growth_Rate_Percent', 'Consumer_Confidence_Index'
            ]
            xgb_features_existing = [col for col in xgb_features if col in daily_sales_features.columns]
            if len(xgb_features_existing) != len(xgb_features):
                st.warning(f"บางคอลัมน์ Feature สำหรับ XGBoost หายไปจากข้อมูลที่รวมกัน: {list(set(xgb_features) - set(xgb_features_existing))}. คุณภาพการพยากรณ์อาจได้รับผลกระทบ.")
            
            for col in xgb_features_existing:
                daily_sales_features[col] = pd.to_numeric(daily_sales_features[col], errors='coerce').fillna(daily_sales_features[col].mean() if not daily_sales_features[col].isnull().all() else 0)


            train_df_xgb = daily_sales_features.dropna(subset=['Units_Sold', 'residuals'] + xgb_features_existing)

            if not train_df_xgb.empty and len(train_df_xgb) > 1 and all(col in train_df_xgb.columns for col in xgb_features_existing):
                X_train_xgb = train_df_xgb[xgb_features_existing]
                y_train_xgb = train_df_xgb['residuals']

                xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
                xgb_model.fit(X_train_xgb, y_train_xgb)

                future_xgb_df = future.copy()
                future_xgb_df['month'] = future_xgb_df['ds'].dt.month
                future_xgb_df['day_of_week'] = future_xgb_df['ds'].dt.dayofweek
                future_xgb_df['is_weekend'] = future_xgb_df['day_of_week'].isin([5, 6]).astype(int)
                
                for col in xgb_features_existing:
                    if col not in ['month', 'day_of_week', 'is_weekend']:
                        if col in train_df_xgb.columns and not train_df_xgb[col].empty:
                            future_xgb_df[col] = train_df_xgb[col].iloc[-1]
                        else:
                            future_xgb_df[col] = 0

                for col in xgb_features_existing:
                    future_xgb_df[col] = pd.to_numeric(future_xgb_df[col], errors='coerce').fillna(0)

                xgb_residual_pred = xgb_model.predict(future_xgb_df[xgb_features_existing])

                final_forecast_df = forecast_prophet.copy()
                final_forecast_df['xgb_residual_pred'] = xgb_residual_pred
                final_forecast_df['yhat_hybrid'] = final_forecast_df['yhat'] + final_forecast_df['xgb_residual_pred']
                final_forecast_df['yhat_hybrid'] = final_forecast_df['yhat_hybrid'].apply(lambda x: max(0, round(x)))

                st.subheader("กราฟพยากรณ์ยอดขายรวม")
                fig = plt.figure(figsize=(12, 6))
                plt.plot(prophet_df['ds'], prophet_df['y'], label='ยอดขายจริงรวม', color='blue')
                plt.plot(final_forecast_df['ds'], final_forecast_df['yhat_hybrid'], label='พยากรณ์ (Prophet + XGBoost)', color='green', linestyle='--')
                plt.plot(forecast_prophet['ds'], forecast_prophet['yhat'], label='พยากรณ์ (Prophet Baseline)', color='orange', linestyle=':')
                
                forecast_start_date = sales_df['Date'].max()
                plt.axvline(x=forecast_start_date, color='red', linestyle='--', label='จุดเริ่มต้นพยากรณ์', alpha=0.7)

                plt.title('ยอดขายโครงการอสังหาริมทรัพย์: จริง vs. พยากรณ์')
                plt.xlabel('วันที่')
                plt.ylabel('จำนวนยูนิตที่ขายได้')
                plt.legend()
                plt.grid(True)
                st.pyplot(fig)

                st.subheader("ข้อมูลพยากรณ์ล่าสุด")
                st.dataframe(final_forecast_df[['ds', 'yhat_hybrid']].tail(future_periods))
            else:
                st.warning("ไม่สามารถฝึกโมเดล XGBoost ได้: ข้อมูลมีไม่เพียงพอ, มีค่าว่างมากเกินไป, หรือคอลัมน์ Features ที่คาดหวังไม่ครบถ้วน.")
                st.write("Current daily_sales_features columns:", daily_sales_features.columns.tolist())
                st.write("Expected xgb_features:", xgb_features_existing)
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการรันโมเดลพยากรณ์: {e}")
            st.exception(e)
    else:
        st.warning("โปรดอัปโหลดข้อมูลการขายที่มีอย่างน้อย 2 แถว เพื่อให้ Prophet สามารถฝึกโมเดลได้")
else:
    st.info("กรุณาตรวจสอบการตั้งค่าโฟลเดอร์ข้อมูลและไฟล์ CSV เพื่อเริ่มต้นการพยากรณ์")

# --- NEW: Forecasting by Project Type ---
if not sales_df.empty and 'Project_Type' in sales_df.columns:
    st.subheader("กราฟพยากรณ์ยอดขายตามประเภทโครงการ")
    project_types_for_selection = ['ยอดขายรวมทั้งหมด'] + sales_df['Project_Type'].unique().tolist()
    selected_project_type_forecast = st.selectbox(
        "เลือกประเภทโครงการเพื่อดูการพยากรณ์:",
        options=project_types_for_selection
    )

    if selected_project_type_forecast == 'ยอดขายรวมทั้งหมด':
        # Re-display the overall forecast graph
        st.write("แสดงกราฟพยากรณ์ยอดขายรวมทั้งหมด")
        fig = plt.figure(figsize=(12, 6))
        plt.plot(prophet_df['ds'], prophet_df['y'], label='ยอดขายจริงรวม', color='blue')
        plt.plot(final_forecast_df['ds'], final_forecast_df['yhat_hybrid'], label='พยากรณ์ (Prophet + XGBoost)', color='green', linestyle='--')
        plt.plot(forecast_prophet['ds'], forecast_prophet['yhat'], label='พยากรณ์ (Prophet Baseline)', color='orange', linestyle=':')
        
        forecast_start_date = sales_df['Date'].max()
        plt.axvline(x=forecast_start_date, color='red', linestyle='--', label='จุดเริ่มต้นพยากรณ์', alpha=0.7)

        plt.title('ยอดขายโครงการอสังหาริมทรัพย์: จริง vs. พยากรณ์ (รวมทุกประเภท)')
        plt.xlabel('วันที่')
        plt.ylabel('จำนวนยูนิตที่ขายได้')
        plt.legend()
        plt.grid(True)
        st.pyplot(fig)
    else:
        st.write(f"แสดงกราฟพยากรณ์ยอดขายสำหรับโครงการประเภท: **{selected_project_type_forecast}**")
        
        filtered_sales_df_for_forecast = sales_df[sales_df['Project_Type'] == selected_project_type_forecast].copy()
        
        if not filtered_sales_df_for_forecast.empty and len(filtered_sales_df_for_forecast) > 1:
            prophet_df_filtered = filtered_sales_df_for_forecast.groupby('Date')['Units_Sold'].sum().reset_index()
            prophet_df_filtered.rename(columns={'Date': 'ds', 'Units_Sold': 'y'}, inplace=True)

            try:
                m_filtered = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
                m_filtered.fit(prophet_df_filtered)
                future_filtered = m_filtered.make_future_dataframe(periods=future_periods)
                forecast_prophet_filtered = m_filtered.predict(future_filtered)
                
                # For simplicity, we are only using Prophet for per-project-type forecast.
                # Adding XGBoost here would require more robust feature engineering per type.
                
                fig = plt.figure(figsize=(12, 6))
                plt.plot(prophet_df_filtered['ds'], prophet_df_filtered['y'], label=f'ยอดขายจริง {selected_project_type_forecast}', color='blue')
                plt.plot(forecast_prophet_filtered['ds'], forecast_prophet_filtered['yhat'], label=f'พยากรณ์ {selected_project_type_forecast} (Prophet)', color='green', linestyle='--')
                
                forecast_start_date_filtered = filtered_sales_df_for_forecast['Date'].max()
                plt.axvline(x=forecast_start_date_filtered, color='red', linestyle='--', label='จุดเริ่มต้นพยากรณ์', alpha=0.7)

                plt.title(f'ยอดขายโครงการอสังหาริมทรัพย์: จริง vs. พยากรณ์ ({selected_project_type_forecast})')
                plt.xlabel('วันที่')
                plt.ylabel('จำนวนยูนิตที่ขายได้')
                plt.legend()
                plt.grid(True)
                st.pyplot(fig)

                st.subheader(f"ข้อมูลพยากรณ์สำหรับ {selected_project_type_forecast}")
                st.dataframe(forecast_prophet_filtered[['ds', 'yhat']].tail(future_periods))

            except Exception as e:
                st.warning(f"ไม่สามารถพยากรณ์สำหรับประเภทโครงการ {selected_project_type_forecast} ได้: {e}. อาจเป็นเพราะข้อมูลไม่เพียงพอสำหรับประเภทนี้.")
        else:
            st.info(f"ไม่พบข้อมูลเพียงพอสำหรับประเภทโครงการ '{selected_project_type_forecast}' ที่เลือก เพื่อทำการพยากรณ์.")
else:
    st.info("ไม่สามารถแสดงกราฟพยากรณ์ตามประเภทโครงการได้, อาจต้องตรวจสอบข้อมูล Project Details.")


# --- Additional Sales Insights for Executive ---
st.header("📈 ข้อมูลเชิงลึกการขายที่สำคัญ")
if not sales_df.empty:
    st.write("---")
    
    if 'Project_Type' in sales_df.columns:
        sales_by_type = sales_df.groupby('Project_Type')['Sales_Revenue'].sum().sort_values(ascending=False)
        st.write("**ยอดขายรวมตามประเภทโครงการ:**")
        st.dataframe(sales_by_type)
    else:
        st.warning("คอลัมน์ 'Project_Type' ไม่พร้อมใช้งานสำหรับการวิเคราะห์นี้ (อาจต้องตรวจสอบข้อมูล Project Details).")

    sales_by_project = sales_df.groupby('Project_Name')['Sales_Revenue'].sum().sort_values(ascending=False)
    st.write("**โครงการที่มีผลงานดี/แย่ที่สุด (ตามรายได้):**")
    col_top, col_bottom = st.columns(2)
    with col_top:
        st.write("**Top 5 โครงการ**")
        st.dataframe(sales_by_project.head(5))
    with col_bottom:
        st.write("**Bottom 5 โครงการ**")
        st.dataframe(sales_by_project.tail(5))

    if 'Lead_Source' in sales_df.columns:
        sales_by_lead_source = sales_df.groupby('Lead_Source')['Sales_Revenue'].sum().sort_values(ascending=False)
        st.write("**ยอดขายตามแหล่งที่มาของ Lead:**")
        st.dataframe(sales_by_lead_source)
    else:
        st.warning("คอลัมน์ 'Lead_Source' ไม่พร้อมใช้งานสำหรับการวิเคราะห์นี้.")

    if 'Promotion_Applied' in sales_df.columns:
        sales_by_promotion = sales_df.groupby('Promotion_Applied')['Sales_Revenue'].sum().sort_values(ascending=False)
        st.write("**ยอดขายตามโปรโมชั่นที่ใช้:**")
        st.dataframe(sales_by_promotion)
    else:
        st.warning("คอลัมน์ 'Promotion_Applied' ไม่พร้อมใช้งานสำหรับการวิเคราะห์นี้.")
else:
    st.info("กรุณาตรวจสอบการตั้งค่าโฟลเดอร์ข้อมูลและไฟล์ CSV เพื่อแสดงข้อมูลเชิงลึก")


# --- NEW SECTION: Project Details Analysis ---
st.header("🏢 การวิเคราะห์ข้อมูลโครงการ")
if not project_details_df.empty:
    st.write("---") # Divider
    st.subheader("ภาพรวมข้อมูลโครงการ")
    st.dataframe(project_details_df.head())

    if 'Units_Sold_toDate' in project_details_df.columns and 'Proximity_to_BTS_MRT_Km' in project_details_df.columns:
        st.subheader("ความสัมพันธ์ระหว่างยอดขายกับระยะทางจาก BTS/MRT")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=project_details_df, x='Proximity_to_BTS_MRT_Km', y='Units_Sold_toDate', hue='Project_Type', ax=ax)
        ax.set_title('ยอดขายสะสม vs. ระยะทางจาก BTS/MRT')
        ax.set_xlabel('ระยะทางจาก BTS/MRT (กม.)')
        ax.set_ylabel('จำนวนยูนิตที่ขายได้ (สะสม)')
        st.pyplot(fig)
    else:
        st.info("ไม่พบข้อมูล 'Units_Sold_toDate' หรือ 'Proximity_to_BTS_MRT_Km' ในไฟล์ข้อมูลโครงการ.")
else:
    st.info("กรุณาตรวจสอบการตั้งค่าโฟลเดอร์ข้อมูลและไฟล์ CSV เพื่อวิเคราะห์ข้อมูลโครงการ")

# --- NEW SECTION: Marketing Activities Analysis ---
st.header("📣 การวิเคราะห์กิจกรรมการตลาด")
if not marketing_activities_df.empty:
    st.write("---") # Divider
    st.subheader("งบประมาณการตลาดและจำนวน Leads ที่ได้")
    marketing_summary = marketing_activities_df.groupby('Marketing_Channel').agg(
        Total_Budget=('Budget_Baht', 'sum'),
        Total_Leads=('Leads_Generated', 'sum')
    ).sort_values(by='Total_Budget', ascending=False)
    st.dataframe(marketing_summary)

    if 'Total_Budget' in marketing_summary.columns and 'Total_Leads' in marketing_summary.columns:
        st.subheader("งบประมาณการตลาด vs. จำนวน Leads ที่ได้ (ตามช่องทาง)")
        fig, ax = plt.subplots(figsize=(12, 6))
        
        marketing_summary_plot = marketing_summary.copy()
        
        # Corrected label format for plot function
        marketing_summary_plot.plot(kind='bar', y=['Total_Budget'], ax=ax, width=0.4, position=1, label=['งบประมาณรวม (บาท)'])
        ax2 = ax.twinx()
        marketing_summary_plot.plot(kind='bar', y=['Total_Leads'], ax=ax2, width=0.4, position=0, color='orange', label=['จำนวน Leads รวม'])
        
        ax.set_title('งบประมาณการตลาด vs. จำนวน Leads ที่ได้ (ตามช่องทาง)')
        ax.set_xlabel('ช่องทางการตลาด')
        ax.set_ylabel('งบประมาณ (บาท)')
        ax2.set_ylabel('จำนวน Leads')
        ax.ticklabel_format(style='plain', axis='y')
        ax2.ticklabel_format(style='plain', axis='y')
        plt.xticks(rotation=45, ha='right')
        
        # Combine legends from both axes
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        fig.legend(lines + lines2, labels + labels2, loc="upper right", bbox_to_anchor=(0.95, 0.95))

        plt.tight_layout(rect=[0, 0, 0.9, 1])
        st.pyplot(fig)
    else:
        st.info("ไม่พบข้อมูล 'Total_Budget' หรือ 'Total_Leads' ในข้อมูลสรุปการตลาด.")
else:
    st.info("กรุณาตรวจสอบการตั้งค่าโฟลเดอร์ข้อมูลและไฟล์ CSV เพื่อวิเคราะห์กิจกรรมการตลาด")

# --- NEW SECTION: Sales Funnel Analysis ---
st.header("📊 การวิเคราะห์ Sales Funnel")
if not sales_funnel_df.empty:
    st.write("---") # Divider
    st.subheader("ภาพรวม Sales Funnel")
    
    total_leads = len(sales_funnel_df)
    converted_leads = sales_funnel_df[sales_funnel_df['Is_Converted_to_Sale'] == True]
    conversion_rate = (len(converted_leads) / total_leads) * 100 if total_leads > 0 else 0

    st.info(f"จำนวน Lead ทั้งหมด: {total_leads:,.0f} | จำนวน Lead ที่เปลี่ยนเป็นยอดขาย: {len(converted_leads):,.0f} | อัตราการเปลี่ยนเป็นยอดขาย (Conversion Rate): {conversion_rate:.2f}%")

    leads_by_source = sales_funnel_df.groupby('Lead_Source').size().sort_values(ascending=False)
    st.subheader("จำนวน Leads ตามแหล่งที่มา")
    fig, ax = plt.subplots(figsize=(10, 6))
    leads_by_source.plot(kind='bar', ax=ax)
    ax.set_title('จำนวน Leads ตามแหล่งที่มา')
    ax.set_xlabel('แหล่งที่มาของ Lead')
    ax.set_ylabel('จำนวน Leads')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
else:
    st.info("กรุณาตรวจสอบการตั้งค่าโฟลเดอร์ข้อมูลและไฟล์ CSV เพื่อวิเคราะห์ Sales Funnel")

# --- NEW SECTION: Competitor Analysis ---
st.header("📊 การวิเคราะห์คู่แข่ง")
if not competitor_df.empty:
    st.write("---") # Divider
    st.subheader("ภาพรวมข้อมูลคู่แข่ง")
    st.dataframe(competitor_df.head())

    st.subheader("ราคาเฉลี่ยต่อ ตร.ม. ของคู่แข่ง")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=competitor_df.sort_values(by='Avg_Price_Per_SqM', ascending=False), x='Competitor_Project_Name', y='Avg_Price_Per_SqM', ax=ax)
    ax.set_title('ราคาเฉลี่ยต่อ ตร.ม. ของคู่แข่ง')
    ax.set_xlabel('ชื่อโครงการคู่แข่ง')
    ax.set_ylabel('ราคาเฉลี่ยต่อ ตร.ม. (บาท)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
else:
    st.info("กรุณาตรวจสอบการตั้งค่าโฟลเดอร์ข้อมูลและไฟล์ CSV เพื่อวิเคราะห์คู่แข่ง")

# --- NEW SECTION: Macroeconomic Data Insights ---
st.header("📈 ข้อมูลเศรษฐกิจมหภาค")
if not macro_economic_df.empty:
    st.write("---") # Divider
    st.subheader("แนวโน้มอัตราดอกเบี้ยและอัตราเงินเฟ้อ")
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(macro_economic_df['Date'], macro_economic_df['Interest_Rate_Percent'], color='blue', label='อัตราดอกเบี้ย (%)')
    ax1.set_xlabel('วันที่')
    ax1.set_ylabel('อัตราดอกเบี้ย (%)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2 = ax1.twinx()
    ax2.plot(macro_economic_df['Date'], macro_economic_df['Inflation_Rate_Percent'], color='red', label='อัตราเงินเฟ้อ (%)')
    ax2.set_ylabel('อัตราเงินเฟ้อ (%)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    fig.tight_layout()
    plt.title('แนวโน้มอัตราดอกเบี้ยและอัตราเงินเฟ้อ')
    fig.legend(loc="upper left", bbox_to_anchor=(0.1,0.9))
    st.pyplot(fig)

    st.subheader("ดัชนีความเชื่อมั่นผู้บริโภคและ GDP Growth")
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(macro_economic_df['Date'], macro_economic_df['Consumer_Confidence_Index'], color='green', label='ดัชนีความเชื่อมั่นผู้บริโภค')
    ax1.set_xlabel('วันที่')
    ax1.set_ylabel('ดัชนีความเชื่อมั่น', color='green')
    ax1.tick_params(axis='y', labelcolor='green')

    ax2 = ax1.twinx()
    ax2.plot(macro_economic_df['Date'], macro_economic_df['GDP_Growth_Rate_Percent'], color='purple', label='GDP Growth Rate (%)')
    ax2.set_ylabel('GDP Growth Rate (%)', color='purple')
    ax2.tick_params(axis='y', labelcolor='purple')

    fig.tight_layout()
    plt.title('แนวโน้มดัชนีความเชื่อมั่นผู้บริโภคและ GDP Growth Rate')
    fig.legend(loc="upper left", bbox_to_anchor=(0.1,0.9))
    st.pyplot(fig)

else:
    st.info("กรุณาตรวจสอบการตั้งค่าโฟลเดอร์ข้อมูลและไฟล์ CSV เพื่อวิเคราะห์เศรษฐกิจมหภาค")

# --- Customer Feedback Analysis Section (with Llama 3.1 8B) ---
st.header("💬 การวิเคราะห์ความคิดเห็นลูกค้า (โดย Llama 3.1 8B)")
if not feedback_df.empty:
    client = ollama.Client(host=ollama_base_url)

    @st.cache_data
    def analyze_sentiment(text):
        if not text: return "ไม่สามารถวิเคราะห์ได้ (ข้อความว่าง)"
        try:
            response = client.chat(
                model=ollama_model_name,
                messages=[
                    {
                        'role': 'user',
                        'content': f"Analyze the sentiment of the following real estate customer feedback as POSITIVE, NEGATIVE, or NEUTRAL, and explain why briefly (in Thai if possible): '{text}'",
                    },
                ],
                options={'temperature': 0.2}
            )
            return response['message']['content']
        except Exception as e:
            st.error(f"Error connecting to Ollama (Sentiment): {e}. Please check URL or server status and model.")
            return "ไม่สามารถวิเคราะห์ได้"

    @st.cache_data
    def summarize_feedback(text):
        if not text: return "ไม่สามารถสรุปได้ (ข้อความว่าง)"
        try:
            response = client.chat(
                model=ollama_model_name,
                messages=[
                    {
                        'role': 'user',
                        'content': f"Summarize the key points of the following real estate customer feedback in one concise sentence (in Thai if possible): '{text}'",
                    },
                ],
                options={'temperature': 0.2}
            )
            return response['message']['content']
        except Exception as e:
            st.error(f"Error connecting to Ollama (Summary): {e}. Please check URL or server status and model.")
            return "ไม่สามารถสรุปได้"

    if st.button("วิเคราะห์ความคิดเห็นลูกค้า"):
        with st.spinner("กำลังวิเคราะห์ความคิดเห็น... อาจใช้เวลาสักครู่"):
            if 'feedback_text' in feedback_df.columns:
                feedback_subset = feedback_df.head(50).copy()
                feedback_subset['feedback_text'] = feedback_subset['feedback_text'].astype(str)

                feedback_subset['sentiment_analysis'] = feedback_subset['feedback_text'].apply(analyze_sentiment)
                feedback_subset['summary'] = feedback_subset['feedback_text'].apply(summarize_feedback)
                
                st.subheader("ผลการวิเคราะห์ความคิดเห็น")
                sentiment_counts = feedback_subset['sentiment_analysis'].apply(lambda x: "POSITIVE" if "POSITIVE" in x.upper() else ("NEGATIVE" if "NEGATIVE" in x.upper() else "NEUTRAL" if "NEUTRAL" in x.upper() else "UNKNOWN")).value_counts()
                
                if not sentiment_counts.empty and len(sentiment_counts) > 1:
                    fig_pie = plt.figure(figsize=(8, 8))
                    plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel"))
                    plt.title('สัดส่วนความรู้สึกของลูกค้า')
                    st.pyplot(fig_pie)
                else:
                    st.write("ไม่พบข้อมูล Sentiment ที่ชัดเจน หรือมีประเภท Sentiment เพียงพอสำหรับแสดงกราฟ.")

                st.subheader("ตัวอย่างความคิดเห็นและการวิเคราะห์")
                st.dataframe(feedback_subset[['feedback_text', 'sentiment_analysis', 'summary']])
            else:
                st.warning("ไม่พบคอลัมน์ 'feedback_text' ในไฟล์ข้อมูลความคิดเห็นลูกค้า. โปรดตรวจสอบชื่อคอลัมน์.")
else:
    st.info("กรุณาตรวจสอบการตั้งค่าโฟลเดอร์ข้อมูลและไฟล์ CSV เพื่อเริ่มต้นการวิเคราะห์ความคิดเห็นลูกค้า")

# --- Interactive AI Chat (Q&A with Llama 3.1 8B) ---
st.header("🤖 ถาม-ตอบกับ AI (Llama 3.1 8B)")
st.write("คุณสามารถถามคำถามเกี่ยวกับข้อมูลการขาย หรือสิ่งที่คุณต้องการให้ AI ช่วยวิเคราะห์เพิ่มเติมได้")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

st.subheader("ลองถามคำถามเหล่านี้ดูสิ:")

suggested_questions = [
    "พยากรณ์ยอดขายรวมของโครงการสำหรับ 30 วันข้างหน้าให้หน่อย",
    "โครงการประเภทไหนที่มียอดขายสูงสุด และทำไมถึงเป็นแบบนั้น",
    "ช่องทางการตลาดใดที่สร้าง Leads ได้มากที่สุด",
    "อัตราการเปลี่ยน Lead เป็นยอดขายโดยรวมของเราตอนนี้เป็นเท่าไหร่",
    "ราคาเฉลี่ยต่อ ตร.ม. ของคู่แข่งในเขตกรุงเทพฯ เป็นเท่าไหร่",
    "ดัชนีความเชื่อมั่นผู้บริโภคที่ลดลง ส่งผลต่อยอดขายโครงการของเราอย่างไรบ้างในอดีต"
]

# Variable to hold the chosen prompt from suggestions
chosen_prompt = None

# Display suggested questions as buttons in columns
cols = st.columns(3)
for i, question in enumerate(suggested_questions):
    with cols[i % 3]: # Distribute buttons into 3 columns
        if st.button(question, key=f"q_suggest_{i}"):
            chosen_prompt = question # Store the chosen question
            # No st.rerun() here yet, we'll handle the prompt processing below

# Or use a selectbox for many questions
selected_question_from_list = st.selectbox(
    "หรือเลือกคำถามจากรายการ:",
    [""] + suggested_questions,
    key="select_q_from_list"
)
if selected_question_from_list:
    if st.button("ถามคำถามที่เลือก", key="ask_selected_q"):
        chosen_prompt = selected_question_from_list # Store the chosen question
        # No st.rerun() here yet


# Get user input from chat_input
user_input_prompt = st.chat_input("คุณมีคำถามอะไรเกี่ยวกับข้อมูลการขายไหม?")

# Determine the actual prompt to process
# Prioritize chosen_prompt from suggestions if available, otherwise use chat_input
actual_prompt = chosen_prompt if chosen_prompt else user_input_prompt

if actual_prompt:
    # Add user message to chat history if it's new (not already added by button logic)
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user" and st.session_state.messages[-1]["content"] == actual_prompt:
        pass
    else:
        st.session_state.messages.append({"role": "user", "content": actual_prompt})

    # Display the user message immediately
    with st.chat_message("user"):
        st.markdown(actual_prompt)

    with st.chat_message("assistant"):
        with st.spinner("AI กำลังคิดคำตอบ..."):
            assistant_response = "ขออภัย, เกิดข้อผิดพลาดในการเชื่อมต่อ AI."
            try:
                context_data = ""

                # --- Build context_data based on available DataFrames ---
                # (ส่วนนี้ยังคงเดิมทั้งหมด)
                if not sales_df.empty:
                    total_revenue = sales_df['Sales_Revenue'].sum()
                    total_units = sales_df['Units_Sold'].sum()
                    context_data += (
                        f"ข้อมูลสรุปการขาย:\n"
                        f"- ยอดขายรวม (รายได้): {total_revenue:,.0f} บาท\n"
                        f"- จำนวนยูนิตที่ขายได้รวม: {total_units:,.0f} ยูนิต\n"
                    )
                    if 'Project_Type' in sales_df.columns and not sales_df['Project_Type'].empty:
                        sales_by_type = sales_df.groupby('Project_Type')['Sales_Revenue'].sum().to_string()
                        context_data += f"- ยอดขายตามประเภทโครงการ:\n{sales_by_type}\n"
                    if 'Lead_Source' in sales_df.columns and not sales_df['Lead_Source'].empty:
                        top_lead_source = sales_df.groupby('Lead_Source')['Units_Sold'].sum().idxmax()
                        context_data += f"- แหล่งที่มาของ Lead ที่ทำยอดขายสูงสุด: {top_lead_source}\n"
                    context_data += "\nข้อมูลยอดขายล่าสุดบางส่วน:\n" + sales_df.tail(5).to_string()

                if not feedback_df.empty and 'summary' in feedback_df.columns and not feedback_df['summary'].empty:
                    context_data += "\n\nสรุปความคิดเห็นลูกค้าบางส่วน:\n"
                    context_data += feedback_df['summary'].dropna().sample(min(5, len(feedback_df)), random_state=42).to_string(index=False)

                if not project_details_df.empty:
                    context_data += "\n\nข้อมูลสรุปโครงการ:\n"
                    context_data += project_details_df.describe(include='all').to_string()
                    context_data += f"\n- โครงการทั้งหมด: {len(project_details_df)} โครงการ\n"
                    context_data += f"- จำนวนยูนิตรวมทุกโครงการ: {project_details_df['Total_Units'].sum():,.0f} ยูนิต\n"
                    
                if not marketing_activities_df.empty:
                    total_marketing_budget = marketing_activities_df['Budget_Baht'].sum()
                    total_leads_generated = marketing_activities_df['Leads_Generated'].sum()
                    if 'Marketing_Channel' in marketing_activities_df.columns and not marketing_activities_df['Marketing_Channel'].empty:
                        top_marketing_channel = marketing_activities_df.groupby('Marketing_Channel')['Leads_Generated'].sum().idxmax()
                        context_data += f"- ช่องทางการตลาดที่สร้าง Leads ได้สูงสุด: {top_marketing_channel}\n"
                    context_data += (
                        f"\n\nข้อมูลกิจกรรมการตลาด:\n"
                        f"- งบประมาณการตลาดรวม: {total_marketing_budget:,.0f} บาท\n"
                        f"- จำนวน Leads ที่สร้างได้รวม: {total_leads_generated:,.0f} Leads\n"
                    )

                if not sales_funnel_df.empty:
                    total_leads_funnel = len(sales_funnel_df)
                    converted_leads_funnel = sales_funnel_df[sales_funnel_df['Is_Converted_to_Sale'] == True]
                    conversion_rate_funnel = (len(converted_leads_funnel) / total_leads_funnel) * 100 if total_leads_funnel > 0 else 0
                    context_data += (
                        f"\n\nข้อมูล Sales Funnel:\n"
                        f"- จำนวน Leads ใน Funnel: {total_leads_funnel:,.0f}\n"
                        f"- อัตราการเปลี่ยนเป็นยอดขายจาก Funnel: {conversion_rate_funnel:.2f}%\n"
                    )
                    
                if not competitor_df.empty:
                    avg_comp_price = competitor_df['Avg_Price_Per_SqM'].mean()
                    context_data += (
                        f"\n\nข้อมูลคู่แข่ง:\n"
                        f"- ราคาเฉลี่ยต่อ ตร.ม. ของคู่แข่ง: {avg_comp_price:,.0f} บาท\n"
                        f"- จำนวนโครงการคู่แข่งที่ติดตาม: {len(competitor_df)} โครงการ\n"
                    )

                if not macro_economic_df.empty:
                    context_data += "\n\nข้อมูลเศรษฐกิจมหภาค (ล่าสุด):\n"
                    context_data += macro_economic_df.tail(1).to_string()

                llm_prompt = (
                    f"คุณคือผู้ช่วยวิเคราะห์ข้อมูลการขายอสังหาริมทรัพย์ "
                    f"โปรดตอบคำถามต่อไปนี้โดยอ้างอิงจากข้อมูลบริบทที่ได้รับ "
                    f"และให้คำแนะนำที่เป็นประโยชน์ต่อผู้บริหารอสังหาฯ "
                    f"หากข้อมูลไม่เพียงพอ ให้ระบุว่าต้องการข้อมูลอะไรเพิ่มเติม:\n\n"
                    f"**คำถาม:** {actual_prompt}\n\n"
                    f"**ข้อมูลบริบท:**\n{context_data}\n\n"
                    f"**คำตอบ:**"
                )

                if ai_provider == "Ollama (Local Llama)":
                    if ollama_base_url and ollama_model_name:
                        client_ollama = ollama.Client(host=ollama_base_url)
                        response = client_ollama.chat(
                            model=ollama_model_name,
                            messages=[
                                {"role": "system", "content": "You are a helpful real estate sales data analysis assistant for a large real estate company in Thailand. Provide insights and recommendations for executives based on provided data."},
                                {"role": "user", "content": llm_prompt},
                            ],
                            options={'temperature': 0.7}
                        )
                        assistant_response = response['message']['content']
                    else:
                        st.warning("โปรดกำหนด Ollama Server URL และ Model Name ใน Sidebar.")
                        assistant_response = "Ollama ไม่ได้รับการกำหนดค่า."

                elif ai_provider == "OpenAI (GPT)":
                    if openai_api_key and openai_model_name:
                        client_openai = openai.OpenAI(api_key=openai_api_key)
                        response = client_openai.chat.completions.create(
                            model=openai_model_name,
                            messages=[
                                {"role": "system", "content": "You are a helpful real estate sales data analysis assistant for a large real estate company in Thailand. Provide insights and recommendations for executives based on provided data."},
                                {"role": "user", "content": llm_prompt},
                            ],
                            temperature=0.7,
                        )
                        assistant_response = response.choices[0].message.content
                    else:
                        st.warning("โปรดใส่ OpenAI API Key และ Model Name ใน Sidebar.")
                        assistant_response = "OpenAI ไม่ได้รับการกำหนดค่า."

                elif ai_provider == "Google AI (Gemini)":
                    if google_api_key and google_model_name:
                        from google.generativeai import configure
                        configure(api_key=google_api_key)
                        model_gemini = GenerativeModel(
                            model_name=google_model_name,
                            # Adjust safety settings if needed for your use case
                            safety_settings={
                                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                            }
                        )
                        # For Gemini, system instructions are not directly supported in the same way as OpenAI/Ollama in chat.
                        # You can put the system prompt as the first message or integrate it into the user prompt.
                        # Here, we integrate it into the user prompt for simplicity.
                        full_gemini_prompt = (
                            "คุณคือผู้ช่วยวิเคราะห์ข้อมูลการขายอสังหาริมทรัพย์ และให้คำแนะนำที่เป็นประโยชน์ต่อผู้บริหารอสังหาฯ\n\n" + llm_prompt
                        )
                        response = model_gemini.generate_content(
                            full_gemini_prompt,
                            generation_config={"temperature": 0.7}
                        )
                        assistant_response = response.text
                    else:
                        st.warning("โปรดใส่ Google AI API Key และ Model Name ใน Sidebar.")
                        assistant_response = "Google AI ไม่ได้รับการกำหนดค่า."

                st.markdown(assistant_response)
                st.session_state.messages.append({"role": "assistant", "content": assistant_response})

            except openai.APIError as e:
                st.error(f"เกิดข้อผิดพลาดจาก OpenAI API: {e}. ตรวจสอบ API Key หรือสถานะบริการ.")
                st.session_state.messages.append({"role": "assistant", "content": f"ขออภัยค่ะ, เกิดข้อผิดพลาดจาก OpenAI: {e}"})
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาดในการเชื่อมต่อ AI: {e}. โปรดตรวจสอบการกำหนดค่าหรือ Server.")
                st.session_state.messages.append({"role": "assistant", "content": f"ขออภัยค่ะ, ไม่สามารถเชื่อมต่อกับ AI ได้ในขณะนี้ หรือมีข้อผิดพลาดในการประมวลผลข้อมูล: {e}"})

    st.rerun()

    # Always rerun if there's an actual_prompt processed
    # This ensures the chat input field clears and the new message is displayed
    st.rerun() # <<< เพิ่ม st.rerun() ไว้ท้ายสุดของ block นี้
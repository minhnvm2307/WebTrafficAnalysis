import streamlit as st

def apply_light_theme():
    st.markdown("""
        <style>
        /* Main background */
        .stApp {
            background-color: #f8f9fa;
        }
        
        /* Custom Card Style */
        .css-card {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
            margin-bottom: 20px;
            border: 1px solid #e9ecef;
        }
        
        /* Metrics styling */
        .metric-label {
            font-size: 14px;
            color: #6c757d;
            font-weight: 600;
        }
        .metric-value {
            font-size: 28px;
            color: #212529;
            font-weight: 700;
        }
        
        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background-color: #ffffff;
            border-right: 1px solid #e9ecef;
        }
        
        /* Table header */
        thead tr th {
            background-color: #f1f3f5 !important;
            color: #495057 !important;
        }
        </style>
    """, unsafe_allow_html=True)

def card_start():
    st.markdown('<div class="css-card">', unsafe_allow_html=True)

def card_end():
    st.markdown('</div>', unsafe_allow_html=True)
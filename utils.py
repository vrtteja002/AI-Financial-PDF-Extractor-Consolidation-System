"""
Utility Functions Module
Common utilities for the financial PDF extractor application
"""

import streamlit as st
import os
from typing import Dict, Any, List
import logging
from datetime import datetime
import json

def setup_page_config():
    """Setup Streamlit page configuration."""
    app_title = os.getenv("APP_TITLE", "Financial PDF Extractor & Analyzer")
    
    st.set_page_config(
        page_title=app_title,
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/your-repo/financial-extractor',
            'Report a bug': 'https://github.com/your-repo/financial-extractor/issues',
            'About': f"""
            # {app_title}
            
            This application uses GPT-4 Vision to extract and consolidate financial data from PDF statements.
            
            **Features:**
            - PDF to image conversion
            - AI-powered data extraction
            - Financial ratio calculations
            - Interactive visualizations
            - Excel export functionality
            
            **Version:** 1.0.0
            """
        }
    )

def load_custom_css():
    """Load custom CSS styling for the application."""
    st.markdown("""
    <style>
        /* Main header styling */
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
            padding: 1rem;
            background: linear-gradient(90deg, #f0f8ff, #e6f3ff);
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        /* Section headers */
        .section-header {
            font-size: 1.5rem;
            font-weight: bold;
            color: #2c3e50;
            margin-top: 2rem;
            margin-bottom: 1rem;
            padding: 0.5rem 0;
            border-bottom: 2px solid #3498db;
        }
        
        /* Metric cards */
        .metric-card {
            background: linear-gradient(135deg, #f8f9fa, #e9ecef);
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 4px solid #1f77b4;
            margin: 0.5rem 0;
            box-shadow: 0 2px 6px rgba(0,0,0,0.1);
            transition: transform 0.2s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        
        /* Progress bar styling */
        .stProgress .st-progress-bar {
            background: linear-gradient(90deg, #1f77b4, #17becf);
            height: 20px;
            border-radius: 10px;
        }
        
        /* Success/Warning/Error message styling */
        .success-message {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
            padding: 1rem;
            border-radius: 5px;
            margin: 1rem 0;
        }
        
        .warning-message {
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            color: #856404;
            padding: 1rem;
            border-radius: 5px;
            margin: 1rem 0;
        }
        
        .error-message {
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
            padding: 1rem;
            border-radius: 5px;
            margin: 1rem 0;
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background: linear-gradient(180deg, #f8f9fa, #ffffff);
        }
        
        /* Button styling */
        .stButton > button {
            background: linear-gradient(90deg, #1f77b4, #17becf);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.75rem 1.5rem;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            background: linear-gradient(90deg, #155fa0, #148fb3);
            transform: translateY(-1px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        
        /* File uploader styling */
        .stFileUploader > div > div {
            background: linear-gradient(135deg, #f8f9fa, #e9ecef);
            border: 2px dashed #3498db;
            border-radius: 10px;
            padding: 2rem;
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            padding-left: 20px;
            padding-right: 20px;
            background-color: #f0f2f6;
            border-radius: 8px;
            color: #262730;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(90deg, #1f77b4, #17becf);
            color: white;
        }
        
        /* DataFrame styling */
        .stDataFrame {
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            overflow: hidden;
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            background: linear-gradient(90deg, #f8f9fa, #e9ecef);
            border-radius: 8px;
            padding: 0.5rem;
        }
        
        /* Hide Streamlit default elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Responsive design */
        @media (max-width: 768px) {
            .main-header {
                font-size: 2rem;
            }
            
            .section-header {
                font-size: 1.25rem;
            }
        }
    </style>
    """, unsafe_allow_html=True)

def setup_logging():
    """Setup logging configuration."""
    log_level = logging.DEBUG if os.getenv("DEBUG", "False").lower() == "true" else logging.INFO
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'financial_extractor_{datetime.now().strftime("%Y%m%d")}.log'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def validate_environment():
    """Validate environment variables and dependencies."""
    required_env_vars = []  # Add any required env vars here
    missing_vars = []
    
    for var in required_env_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        st.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        return False
    
    return True

def format_currency(amount: float, currency: str = "SAR") -> str:
    """Format currency amounts with proper formatting."""
    if abs(amount) >= 1_000_000_000:
        return f"{amount/1_000_000_000:.1f}B {currency}"
    elif abs(amount) >= 1_000_000:
        return f"{amount/1_000_000:.1f}M {currency}"
    elif abs(amount) >= 1_000:
        return f"{amount/1_000:.1f}K {currency}"
    else:
        return f"{amount:,.0f} {currency}"

def format_percentage(value: float, decimal_places: int = 2) -> str:
    """Format percentage values."""
    return f"{value:.{decimal_places}f}%"

def format_ratio(value: float, decimal_places: int = 2) -> str:
    """Format ratio values."""
    return f"{value:.{decimal_places}f}x"

def create_download_link(data: bytes, filename: str, link_text: str) -> str:
    """Create a download link for data."""
    import base64
    
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:file/octet-stream;base64,{b64}" download="{filename}">{link_text}</a>'
    return href

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero."""
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (TypeError, ZeroDivisionError):
        return default

def clean_company_name(name: str) -> str:
    """Clean and standardize company names."""
    if not name:
        return "Unknown Company"
    
    # Remove common suffixes and clean up
    name = name.strip()
    name = name.replace("Limited Liability Company", "LLC")
    name = name.replace("Joint Stock Company", "JSC")
    name = name.replace("Company", "Co.")
    
    # Remove file extensions
    name = name.replace(".pdf", "").replace(".PDF", "")
    
    # Limit length
    if len(name) > 50:
        name = name[:50] + "..."
    
    return name

def validate_financial_data(data: Dict[str, Any]) -> List[str]:
    """Validate extracted financial data and return list of issues."""
    issues = []
    
    # Check balance sheet equation
    balance_sheet = data.get("balance_sheet", {})
    assets = balance_sheet.get("assets", {}).get("total_assets", 0)
    liabilities = balance_sheet.get("liabilities", {}).get("total_liabilities", 0)
    equity = balance_sheet.get("equity", {}).get("total_equity", 0)
    
    if assets > 0:
        balance_diff = abs(assets - (liabilities + equity))
        if balance_diff > (assets * 0.05):  # 5% tolerance
            issues.append(f"Balance sheet equation doesn't balance (difference: {balance_diff:,.0f})")
    
    # Check for negative values in typically positive accounts
    income = data.get("income_statement", {})
    if income.get("revenue", 0) < 0:
        issues.append("Revenue is negative")
    
    # Check current ratio reasonableness
    current_assets = balance_sheet.get("assets", {}).get("current_assets", {}).get("total_current_assets", 0)
    current_liabilities = balance_sheet.get("liabilities", {}).get("current_liabilities", {}).get("total_current_liabilities", 0)
    
    if current_liabilities > 0:
        current_ratio = current_assets / current_liabilities
        if current_ratio > 50:  # Unreasonably high
            issues.append(f"Current ratio seems too high ({current_ratio:.2f})")
    
    return issues

def create_data_quality_report(all_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create a data quality report for all extracted data."""
    report = {
        "total_files_processed": len(all_data),
        "successful_extractions": 0,
        "files_with_issues": [],
        "common_issues": {},
        "completeness_stats": {
            "balance_sheet_complete": 0,
            "income_statement_complete": 0,
            "cash_flow_complete": 0
        }
    }
    
    for data in all_data:
        filename = data.get("metadata", {}).get("filename", "Unknown")
        
        # Check for successful extraction
        has_data = (
            data.get("balance_sheet", {}).get("assets", {}).get("total_assets", 0) > 0 or
            data.get("income_statement", {}).get("revenue", 0) > 0
        )
        
        if has_data:
            report["successful_extractions"] += 1
        
        # Check completeness
        bs = data.get("balance_sheet", {})
        if (bs.get("assets", {}).get("total_assets", 0) > 0 and
            bs.get("liabilities", {}).get("total_liabilities", 0) > 0 and
            bs.get("equity", {}).get("total_equity", 0) > 0):
            report["completeness_stats"]["balance_sheet_complete"] += 1
        
        income = data.get("income_statement", {})
        if (income.get("revenue", 0) > 0 and income.get("net_income", 0) != 0):
            report["completeness_stats"]["income_statement_complete"] += 1
        
        cf = data.get("cash_flow", {})
        if cf.get("operating_activities", {}).get("net_cash_operating", 0) != 0:
            report["completeness_stats"]["cash_flow_complete"] += 1
        
        # Validate data and collect issues
        issues = validate_financial_data(data)
        if issues:
            report["files_with_issues"].append({
                "filename": filename,
                "issues": issues
            })
            
            # Track common issues
            for issue in issues:
                issue_type = issue.split("(")[0].strip()  # Get the main issue type
                report["common_issues"][issue_type] = report["common_issues"].get(issue_type, 0) + 1
    
    return report

def display_data_quality_report(report: Dict[str, Any]):
    """Display the data quality report in Streamlit."""
    st.markdown("### 📋 Data Quality Report")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Files Processed", 
            report["total_files_processed"]
        )
    
    with col2:
        success_rate = (report["successful_extractions"] / report["total_files_processed"]) * 100 if report["total_files_processed"] > 0 else 0
        st.metric(
            "Success Rate", 
            f"{success_rate:.1f}%"
        )
    
    with col3:
        st.metric(
            "Complete Balance Sheets", 
            report["completeness_stats"]["balance_sheet_complete"]
        )
    
    with col4:
        st.metric(
            "Complete Income Statements", 
            report["completeness_stats"]["income_statement_complete"]
        )
    
    # Show issues if any
    if report["files_with_issues"]:
        st.markdown("#### ⚠️ Files with Data Quality Issues")
        
        with st.expander(f"View Details ({len(report['files_with_issues'])} files)"):
            for file_issue in report["files_with_issues"]:
                st.write(f"**{file_issue['filename']}:**")
                for issue in file_issue['issues']:
                    st.write(f"  - {issue}")
    
    # Show common issues
    if report["common_issues"]:
        st.markdown("#### 📊 Common Issues")
        
        issues_df = pd.DataFrame(
            list(report["common_issues"].items()),
            columns=["Issue Type", "Frequency"]
        )
        
        fig_issues = px.bar(
            issues_df,
            x="Frequency",
            y="Issue Type",
            orientation='h',
            title="Most Common Data Quality Issues"
        )
        st.plotly_chart(fig_issues, use_container_width=True)

def create_processing_summary(processing_time: float, files_processed: int, successful_extractions: int):
    """Create and display a processing summary."""
    st.markdown("### ⏱️ Processing Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Processing Time", f"{processing_time:.1f}s")
    
    with col2:
        avg_time = processing_time / files_processed if files_processed > 0 else 0
        st.metric("Average Time per File", f"{avg_time:.1f}s")
    
    with col3:
        success_rate = (successful_extractions / files_processed) * 100 if files_processed > 0 else 0
        st.metric("Success Rate", f"{success_rate:.1f}%")

def save_session_state(data: Dict[str, Any], session_id: str = None):
    """Save current session state for recovery."""
    if session_id is None:
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    filename = f"session_{session_id}.json"
    
    try:
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        return filename
    except Exception as e:
        st.error(f"Failed to save session: {str(e)}")
        return None

def load_session_state(filename: str) -> Dict[str, Any]:
    """Load previous session state."""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Failed to load session: {str(e)}")
        return {}
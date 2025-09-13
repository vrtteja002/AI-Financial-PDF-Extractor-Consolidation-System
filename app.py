import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import pdf2image
import openai
import io
import base64
import json
import re
from typing import Dict, List, Any, Optional
import zipfile
import tempfile
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import custom modules
from financial_extractor import FinancialDataExtractor
from data_processor import create_consolidated_dataframe
from visualizations import create_visualizations
from utils import setup_page_config, load_custom_css

def main():
    # Setup page configuration
    setup_page_config()
    
    # Load custom CSS
    load_custom_css()
    
    st.markdown('<div class="main-header">📊 Financial PDF Extractor & Consolidation System</div>', unsafe_allow_html=True)
    
    # Initialize session state variables
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    if 'consolidated_df' not in st.session_state:
        st.session_state.consolidated_df = pd.DataFrame()
    if 'all_extracted_data' not in st.session_state:
        st.session_state.all_extracted_data = []
    
    # Sidebar configuration
    st.sidebar.markdown("### 🔧 Configuration")
    
    # Add reset button in sidebar
    if st.sidebar.button("🔄 Reset Application", help="Clear all data and start over"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    
    # Get API key from environment or user input
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        api_key = st.sidebar.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key to use GPT-4 Vision for PDF extraction"
        )
    else:
        st.sidebar.success("✅ API Key loaded from environment")
    
    if not api_key:
        st.warning("⚠️ Please enter your OpenAI API key in the sidebar to continue.")
        st.info("💡 You can get your API key from: https://platform.openai.com/api-keys")
        st.info("💡 Or add it to your .env file as OPENAI_API_KEY=your_key_here")
        return
    
    # Show different sections based on processing state
    if not st.session_state.processing_complete:
        show_upload_section(api_key)
    else:
        show_results_section()

def show_upload_section(api_key):
    """Show the file upload and processing section."""
    
    # Initialize extractor
    extractor = FinancialDataExtractor(api_key)
    
    # File upload section
    st.markdown('<div class="section-header">📁 Upload Financial PDFs</div>', unsafe_allow_html=True)
    
    max_files = int(os.getenv("MAX_FILES", 23))
    max_size_mb = int(os.getenv("MAX_FILE_SIZE_MB", 50))
    
    uploaded_files = st.file_uploader(
        f"Upload PDF files (up to {max_files} files, max {max_size_mb}MB each)",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload your financial statement PDFs for extraction and consolidation"
    )
    
    if uploaded_files:
        if len(uploaded_files) > max_files:
            st.error(f"❌ Too many files! Maximum allowed: {max_files}")
            return
            
        st.success(f"✅ {len(uploaded_files)} PDF files uploaded successfully!")
        
        # Process files button
        if st.button("🚀 Extract & Consolidate Financial Data", type="primary", key="process_button"):
            process_files(extractor, uploaded_files)

def process_files(extractor, uploaded_files):
    """Process the uploaded PDF files and store results in session state."""
    
    all_extracted_data = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Processing container
    processing_container = st.container()
    
    with processing_container:
        st.markdown("### 🔄 Processing Files...")
        
        for i, uploaded_file in enumerate(uploaded_files):
            try:
                progress = (i + 1) / len(uploaded_files)
                progress_bar.progress(progress)
                status_text.text(f"Processing {uploaded_file.name}...")
                
                # Check file size
                file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
                max_size_mb = int(os.getenv("MAX_FILE_SIZE_MB", 50))
                if file_size_mb > max_size_mb:
                    st.error(f"File {uploaded_file.name} is too large ({file_size_mb:.1f}MB). Max size: {max_size_mb}MB")
                    continue
                
                # Convert PDF to images
                images = extractor.pdf_to_images(uploaded_file)
                
                if images:
                    # Extract data using GPT-4 Vision
                    extracted_data = extractor.extract_financial_data(images, uploaded_file.name)
                    
                    # Calculate ratios
                    ratios = extractor.calculate_financial_ratios(extracted_data)
                    extracted_data["ratios"] = ratios
                    
                    all_extracted_data.append(extracted_data)
                    
                    # Show extraction preview
                    with st.expander(f"📋 Preview: {uploaded_file.name}", expanded=False):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Header Information:**")
                            header_info = extracted_data.get("header_info", {})
                            if header_info:
                                for key, value in header_info.items():
                                    if value and str(value).strip() and str(value) != "0":
                                        st.write(f"**{key.replace('_', ' ').title()}:** {value}")
                            else:
                                st.write("No header information extracted")
                        
                        with col2:
                            st.markdown("**Key Financial Ratios:**")
                            if ratios:
                                for ratio, value in ratios.items():
                                    if isinstance(value, (int, float)) and value != 0:
                                        if 'ratio' in ratio.lower():
                                            st.metric(ratio.replace("_", " ").title(), f"{value:.2f}")
                                        elif 'margin' in ratio.lower() or ratio in ['roa', 'roe']:
                                            st.metric(ratio.replace("_", " ").title(), f"{value:.1f}%")
                                        else:
                                            st.metric(ratio.replace("_", " ").title(), f"{value:,.0f}")
                            else:
                                st.write("No ratios calculated")
                
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                continue
    
    # Complete processing
    progress_bar.progress(1.0)
    status_text.success("Processing complete!")
    
    if all_extracted_data:
        # Store results in session state
        st.session_state.all_extracted_data = all_extracted_data
        
        # Create consolidated DataFrame
        try:
            consolidated_df = create_consolidated_dataframe(all_extracted_data)
            st.session_state.consolidated_df = consolidated_df
            st.session_state.processing_complete = True
            
            st.success(f"🎉 Successfully processed {len(all_extracted_data)} files!")
            st.info("📊 Results are now available. The page will refresh to show analysis.")

            
            
            # Auto-refresh to show results
            st.rerun()
                
        except Exception as e:
            st.error(f"Error creating consolidated data: {str(e)}")
            st.info("❌ Processing failed. Please try again with different files.")
    else:
        st.error("❌ No files were successfully processed. Please check your PDFs and try again.")

def show_results_section():
    """Show the results, analytics, and download section."""
    
    if st.session_state.consolidated_df.empty or not st.session_state.all_extracted_data:
        st.error("❌ No processed data found. Please upload and process files first.")
        if st.button("🔄 Start Over", type="primary"):
            st.session_state.processing_complete = False
            st.rerun()
        return
    
    consolidated_df = st.session_state.consolidated_df
    all_extracted_data = st.session_state.all_extracted_data
    
    # Show summary statistics
    st.markdown('<div class="section-header">📊 Consolidated Results</div>', unsafe_allow_html=True)
    
    # Display summary metrics with error handling
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Companies Processed", len(consolidated_df))
    
    with col2:
        try:
            total_revenue = consolidated_df["revenue"].sum() if "revenue" in consolidated_df.columns else 0
            st.metric("Total Revenue", f"{total_revenue:,.0f}")
        except:
            st.metric("Total Revenue", "N/A")
    
    with col3:
        try:
            avg_margin = consolidated_df["net_profit_margin"].mean() if "net_profit_margin" in consolidated_df.columns else 0
            st.metric("Avg Profit Margin", f"{avg_margin:.1f}%")
        except:
            st.metric("Avg Profit Margin", "N/A")
    
    with col4:
        try:
            avg_current_ratio = consolidated_df["current_ratio"].mean() if "current_ratio" in consolidated_df.columns else 0
            st.metric("Avg Current Ratio", f"{avg_current_ratio:.2f}")
        except:
            st.metric("Avg Current Ratio", "N/A")
    
    # Show consolidated data table with search
    st.markdown("### 📋 Consolidated Financial Data")
    
    # Add search functionality
    col1, col2 = st.columns([3, 1])
    with col1:
        search_term = st.text_input("🔍 Search companies", placeholder="Enter company name...")
    with col2:
        show_all = st.checkbox("Show all columns", value=False)
    
    # Filter dataframe based on search
    display_df = consolidated_df.copy()
    if search_term and "company_name" in display_df.columns:
        mask = display_df["company_name"].str.contains(search_term, case=False, na=False)
        display_df = display_df[mask]
    
    # Select columns to display
    if not show_all:
        key_columns = [
            "company_name", "year", "period", "auditor", "revenue", "net_income", 
            "total_assets", "total_equity", "current_ratio", "debt_to_equity", "net_profit_margin"
        ]
        available_columns = [col for col in key_columns if col in display_df.columns]
        if available_columns:
            display_df = display_df[available_columns]
    
    st.dataframe(display_df, use_container_width=True, height=400)
    
    # Create and display visualizations
    st.markdown('<div class="section-header">📈 Financial Analysis Dashboard</div>', unsafe_allow_html=True)
    
    try:
        create_visualizations(consolidated_df)
    except Exception as e:
        st.error(f"Error creating visualizations: {str(e)}")
        st.info("Some visualizations may not be available due to data format issues.")
    
    # Download section with session state
    show_download_section()
    
    # Analysis insights
    show_analysis_insights(consolidated_df)
    
    # Option to process more files
    st.markdown('<div class="section-header">🔄 Process More Files</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📁 Upload More PDFs", type="secondary"):
            st.session_state.processing_complete = False
            st.rerun()
    
    with col2:
        if st.button("🗑️ Clear All Data", type="secondary"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

def show_download_section():
    """Show download options using session state data."""
    
    st.markdown('<div class="section-header">💾 Download Results</div>', unsafe_allow_html=True)
    
    consolidated_df = st.session_state.consolidated_df
    all_extracted_data = st.session_state.all_extracted_data
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Excel download
        try:
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                # Main consolidated sheet
                consolidated_df.to_excel(writer, sheet_name='Consolidated_Data', index=False)
                
                # Summary statistics sheet
                summary_data = {
                    "Metric": [
                        "Total Companies",
                        "Total Revenue", 
                        "Average Profit Margin (%)",
                        "Average Current Ratio",
                        "Companies with Positive Revenue",
                        "Companies with Positive Net Income"
                    ],
                    "Value": [
                        len(consolidated_df),
                        consolidated_df["revenue"].sum() if "revenue" in consolidated_df.columns else 0,
                        consolidated_df["net_profit_margin"].mean() if "net_profit_margin" in consolidated_df.columns else 0,
                        consolidated_df["current_ratio"].mean() if "current_ratio" in consolidated_df.columns else 0,
                        len(consolidated_df[consolidated_df["revenue"] > 0]) if "revenue" in consolidated_df.columns else 0,
                        len(consolidated_df[consolidated_df["net_income"] > 0]) if "net_income" in consolidated_df.columns else 0
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary_Statistics', index=False)
                
                # Individual company sheets
                for i, data in enumerate(all_extracted_data):
                    try:
                        company_name = data.get("metadata", {}).get("filename", f"Company_{i+1}")
                        # Clean sheet name for Excel compatibility
                        sheet_name = company_name.replace('.pdf', '')[:31]
                        sheet_name = re.sub(r'[\\/*?[\]:]+', '_', sheet_name)
                        
                        # Create flattened data for the company
                        flattened_data = {}
                        
                        # Header info
                        header = data.get("header_info", {})
                        for key, value in header.items():
                            flattened_data[f"header_{key}"] = value
                        
                        # Financial data
                        for section in ["balance_sheet", "income_statement", "cash_flow"]:
                            section_data = data.get(section, {})
                            for key, value in section_data.items():
                                if isinstance(value, dict):
                                    for subkey, subvalue in value.items():
                                        flattened_data[f"{section}_{key}_{subkey}"] = subvalue
                                else:
                                    flattened_data[f"{section}_{key}"] = value
                        
                        # Ratios
                        ratios = data.get("ratios", {})
                        for key, value in ratios.items():
                            flattened_data[f"ratio_{key}"] = value
                        
                        company_df = pd.DataFrame([flattened_data])
                        company_df.to_excel(writer, sheet_name=sheet_name, index=False)
                        
                    except Exception:
                        continue  # Skip problematic companies
            
            st.download_button(
                label="📥 Download Excel File",
                data=excel_buffer.getvalue(),
                file_name=f"financial_consolidation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="excel_download_session",
                help="Download consolidated data in Excel format"
            )
            
        except Exception as e:
            st.error(f"Error creating Excel file: {str(e)}")
    
    with col2:
        # JSON download for raw data
        try:
            json_data = json.dumps(all_extracted_data, indent=2, default=str, ensure_ascii=False)
            st.download_button(
                label="📥 Download Raw JSON",
                data=json_data.encode('utf-8'),
                file_name=f"financial_raw_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                key="json_download_session",
                help="Download raw extracted data in JSON format"
            )
        except Exception as e:
            st.error(f"Error creating JSON file: {str(e)}")
    
    with col3:
        # CSV download for consolidated data
        try:
            csv_data = consolidated_df.to_csv(index=False, encoding='utf-8')
            st.download_button(
                label="📥 Download CSV File",
                data=csv_data.encode('utf-8'),
                file_name=f"financial_consolidated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="csv_download_session",
                help="Download consolidated data in CSV format"
            )
        except Exception as e:
            st.error(f"Error creating CSV file: {str(e)}")

def show_analysis_insights(consolidated_df):
    """Show AI-generated insights from the financial analysis."""
    
    st.markdown('<div class="section-header">🧠 AI Insights</div>', unsafe_allow_html=True)
    
    if consolidated_df.empty:
        st.info("No data available for analysis.")
        return
    
    insights_col1, insights_col2 = st.columns(2)
    
    with insights_col1:
        st.markdown("**🎯 Key Findings:**")
        
        try:
            # Top performer by revenue
            if "revenue" in consolidated_df.columns and consolidated_df['revenue'].sum() > 0:
                top_revenue_idx = consolidated_df['revenue'].idxmax()
                top_revenue_company = consolidated_df.loc[top_revenue_idx]
                company_name = top_revenue_company.get('company_name', 'Unknown')
                revenue = top_revenue_company.get('revenue', 0)
                st.write(f"• **Highest Revenue:** {company_name} ({revenue:,.0f})")
            
            # Best profitability
            if "net_profit_margin" in consolidated_df.columns and consolidated_df['net_profit_margin'].max() > 0:
                best_margin_idx = consolidated_df['net_profit_margin'].idxmax()
                best_margin_company = consolidated_df.loc[best_margin_idx]
                company_name = best_margin_company.get('company_name', 'Unknown')
                margin = best_margin_company.get('net_profit_margin', 0)
                st.write(f"• **Best Profit Margin:** {company_name} ({margin:.1f}%)")
            
            # Best liquidity
            if "current_ratio" in consolidated_df.columns and consolidated_df['current_ratio'].max() > 0:
                best_liquidity_idx = consolidated_df['current_ratio'].idxmax()
                best_liquidity_company = consolidated_df.loc[best_liquidity_idx]
                company_name = best_liquidity_company.get('company_name', 'Unknown')
                ratio = best_liquidity_company.get('current_ratio', 0)
                st.write(f"• **Best Liquidity:** {company_name} (CR: {ratio:.2f})")
        
        except Exception as e:
            st.write("• Unable to calculate key findings due to data format issues")
    
    with insights_col2:
        st.markdown("**⚠️ Risk Indicators:**")
        
        try:
            # Low current ratios
            if "current_ratio" in consolidated_df.columns:
                low_liquidity = consolidated_df[consolidated_df['current_ratio'] < 1.0]
                if len(low_liquidity) > 0:
                    st.write(f"• **{len(low_liquidity)} companies** with current ratio < 1.0")
            
            # High debt ratios
            if "debt_to_equity" in consolidated_df.columns:
                high_debt = consolidated_df[consolidated_df['debt_to_equity'] > 2.0]
                if len(high_debt) > 0:
                    st.write(f"• **{len(high_debt)} companies** with high debt-to-equity (>2.0)")
            
            # Negative margins
            if "net_profit_margin" in consolidated_df.columns:
                negative_margins = consolidated_df[consolidated_df['net_profit_margin'] < 0]
                if len(negative_margins) > 0:
                    st.write(f"• **{len(negative_margins)} companies** with negative profit margins")
        
        except Exception as e:
            st.write("• Unable to calculate risk indicators due to data format issues")

if __name__ == "__main__":
    main()
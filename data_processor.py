"""
Data Processing Module - COMPLETE UPDATED VERSION
Handles consolidation and structuring of extracted financial data with proper header preservation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
import json
import io
import re
from datetime import datetime

def create_consolidated_dataframe(all_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """Create a consolidated DataFrame from all extracted data with proper header preservation."""
    rows = []
    
    for data in all_data:
        row = {}
        
        # Header info - ENHANCED VERSION with proper string handling
        header = data.get("header_info", {})
        
        # Ensure proper string conversion and avoid numeric overwrites
        row["company_name"] = clean_header_field(header.get("company_name", ""))
        row["year"] = clean_header_field(header.get("year", ""))
        row["period"] = clean_header_field(header.get("period", ""))
        row["auditor"] = clean_header_field(header.get("auditor", ""))
        row["opinion_type"] = clean_header_field(header.get("opinion_type", ""))
        row["currency"] = clean_header_field(header.get("currency", "SAR"))
        
        # If currency is still empty, set default
        if not row["currency"]:
            row["currency"] = "SAR"
        
        # Balance Sheet - Assets
        assets = data.get("balance_sheet", {}).get("assets", {})
        current_assets = assets.get("current_assets", {})
        non_current_assets = assets.get("non_current_assets", {})
        
        row.update({
            "total_assets": safe_float(assets.get("total_assets", 0)),
            "total_current_assets": safe_float(current_assets.get("total_current_assets", 0)),
            "cash_and_equivalents": safe_float(current_assets.get("cash_and_equivalents", 0)),
            "accounts_receivable": safe_float(current_assets.get("accounts_receivable", 0)),
            "inventory": safe_float(current_assets.get("inventory", 0)),
            "prepaid_expenses": safe_float(current_assets.get("prepaid_expenses", 0)),
            "other_current_assets": safe_float(current_assets.get("other_current_assets", 0)),
            "total_non_current_assets": safe_float(non_current_assets.get("total_non_current_assets", 0)),
            "property_plant_equipment": safe_float(non_current_assets.get("property_plant_equipment", 0)),
            "intangible_assets": safe_float(non_current_assets.get("intangible_assets", 0)),
            "investments": safe_float(non_current_assets.get("investments", 0)),
            "other_non_current_assets": safe_float(non_current_assets.get("other_non_current_assets", 0))
        })
        
        # Balance Sheet - Liabilities
        liabilities = data.get("balance_sheet", {}).get("liabilities", {})
        current_liabilities = liabilities.get("current_liabilities", {})
        non_current_liabilities = liabilities.get("non_current_liabilities", {})
        
        row.update({
            "total_liabilities": safe_float(liabilities.get("total_liabilities", 0)),
            "total_current_liabilities": safe_float(current_liabilities.get("total_current_liabilities", 0)),
            "accounts_payable": safe_float(current_liabilities.get("accounts_payable", 0)),
            "short_term_debt": safe_float(current_liabilities.get("short_term_debt", 0)),
            "accrued_expenses": safe_float(current_liabilities.get("accrued_expenses", 0)),
            "other_current_liabilities": safe_float(current_liabilities.get("other_current_liabilities", 0)),
            "total_non_current_liabilities": safe_float(non_current_liabilities.get("total_non_current_liabilities", 0)),
            "long_term_debt": safe_float(non_current_liabilities.get("long_term_debt", 0)),
            "deferred_tax": safe_float(non_current_liabilities.get("deferred_tax", 0)),
            "other_non_current_liabilities": safe_float(non_current_liabilities.get("other_non_current_liabilities", 0))
        })
        
        # Balance Sheet - Equity
        equity = data.get("balance_sheet", {}).get("equity", {})
        row.update({
            "total_equity": safe_float(equity.get("total_equity", 0)),
            "share_capital": safe_float(equity.get("share_capital", 0)),
            "retained_earnings": safe_float(equity.get("retained_earnings", 0)),
            "other_equity": safe_float(equity.get("other_equity", 0))
        })
        
        # Income Statement
        income = data.get("income_statement", {})
        row.update({
            "revenue": safe_float(income.get("revenue", 0)),
            "cost_of_goods_sold": safe_float(income.get("cost_of_goods_sold", 0)),
            "gross_profit": safe_float(income.get("gross_profit", 0)),
            "operating_expenses": safe_float(income.get("operating_expenses", 0)),
            "operating_income": safe_float(income.get("operating_income", 0)),
            "interest_expense": safe_float(income.get("interest_expense", 0)),
            "interest_income": safe_float(income.get("interest_income", 0)),
            "other_income": safe_float(income.get("other_income", 0)),
            "income_before_tax": safe_float(income.get("income_before_tax", 0)),
            "tax_expense": safe_float(income.get("tax_expense", 0)),
            "net_income": safe_float(income.get("net_income", 0))
        })
        
        # Cash Flow Statement
        cf = data.get("cash_flow", {})
        operating = cf.get("operating_activities", {})
        investing = cf.get("investing_activities", {})
        financing = cf.get("financing_activities", {})
        
        row.update({
            # Operating activities
            "cf_net_income": safe_float(operating.get("net_income", 0)),
            "cf_depreciation": safe_float(operating.get("depreciation", 0)),
            "cf_working_capital_changes": safe_float(operating.get("working_capital_changes", 0)),
            "cf_other_operating": safe_float(operating.get("other_operating", 0)),
            "operating_cash_flow": safe_float(operating.get("net_cash_operating", 0)),
            
            # Investing activities
            "cf_capital_expenditures": safe_float(investing.get("capital_expenditures", 0)),
            "cf_acquisitions": safe_float(investing.get("acquisitions", 0)),
            "cf_investments": safe_float(investing.get("investments", 0)),
            "cf_other_investing": safe_float(investing.get("other_investing", 0)),
            "investing_cash_flow": safe_float(investing.get("net_cash_investing", 0)),
            
            # Financing activities
            "cf_debt_proceeds": safe_float(financing.get("debt_proceeds", 0)),
            "cf_debt_payments": safe_float(financing.get("debt_payments", 0)),
            "cf_equity_proceeds": safe_float(financing.get("equity_proceeds", 0)),
            "cf_dividends_paid": safe_float(financing.get("dividends_paid", 0)),
            "cf_other_financing": safe_float(financing.get("other_financing", 0)),
            "financing_cash_flow": safe_float(financing.get("net_cash_financing", 0)),
            
            # Net cash flow
            "net_change_cash": safe_float(cf.get("net_change_cash", 0)),
            "beginning_cash": safe_float(cf.get("beginning_cash", 0)),
            "ending_cash": safe_float(cf.get("ending_cash", 0))
        })
        
        # Calculated Ratios
        if "ratios" in data:
            ratios = data["ratios"]
            row.update({
                "current_ratio": safe_float(ratios.get("current_ratio", 0)),
                "quick_ratio": safe_float(ratios.get("quick_ratio", 0)),
                "debt_to_assets": safe_float(ratios.get("debt_to_assets", 0)),
                "debt_to_equity": safe_float(ratios.get("debt_to_equity", 0)),
                "equity_ratio": safe_float(ratios.get("equity_ratio", 0)),
                "net_profit_margin": safe_float(ratios.get("net_profit_margin", 0)),
                "gross_profit_margin": safe_float(ratios.get("gross_profit_margin", 0)),
                "operating_margin": safe_float(ratios.get("operating_margin", 0)),
                "roa": safe_float(ratios.get("roa", 0)),
                "roe": safe_float(ratios.get("roe", 0)),
                "asset_turnover": safe_float(ratios.get("asset_turnover", 0)),
                "working_capital": safe_float(ratios.get("working_capital", 0))
            })
        
        # Notes and qualitative information - handle as strings
        notes = data.get("notes_info", {})
        row.update({
            "share_capital_changes": clean_text_field(notes.get("share_capital_changes", "")),
            "related_party_transactions": clean_text_field(notes.get("related_party_transactions", "")),
            "going_concern_issues": clean_text_field(notes.get("going_concern_issues", "")),
            "significant_estimates": clean_text_field(notes.get("significant_estimates", "")),
            "subsequent_events": clean_text_field(notes.get("subsequent_events", ""))
        })
        
        # Metadata
        metadata = data.get("metadata", {})
        row.update({
            "filename": str(metadata.get("filename", "")),
            "extraction_date": str(metadata.get("extraction_date", "")),
            "pages_processed": safe_float(metadata.get("pages_processed", 0)),
            "successful_extractions": safe_float(metadata.get("successful_extractions", 0))
        })
        
        rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Clean and validate the DataFrame
    df = clean_dataframe(df)
    
    # Fix data types to ensure proper export
    df = fix_dataframe_types(df)
    
    return df

def clean_header_field(value) -> str:
    """Clean header fields ensuring they remain as strings."""
    if value is None or value == "" or pd.isna(value):
        return ""
    
    # Convert to string and clean
    str_value = str(value).strip()
    
    # Don't treat numeric strings as empty
    if str_value in ["0", "0.0", "nan", "None", "null"]:
        return ""
    
    return str_value

def clean_text_field(value) -> str:
    """Clean text fields for notes and qualitative data."""
    if value is None or value == "" or pd.isna(value):
        return ""
    
    str_value = str(value).strip()
    
    # Remove obvious placeholder values
    if str_value.lower() in ["0", "nan", "none", "null", "n/a", ""]:
        return ""
    
    return str_value

def safe_float(value) -> float:
    """Safely convert value to float, handling various formats."""
    if pd.isna(value) or value is None:
        return 0.0
    
    if isinstance(value, (int, float)):
        return float(value) if not pd.isna(value) else 0.0
    
    if isinstance(value, str):
        # Handle parentheses (negative numbers)
        if value.strip().startswith('(') and value.strip().endswith(')'):
            value = '-' + value.strip()[1:-1]
        
        # Remove commas, spaces, currency symbols
        value = value.replace(',', '').replace(' ', '').replace('SAR', '').replace('$', '').strip()
        
        # Handle empty strings
        if not value or value == '-':
            return 0.0
        
        try:
            return float(value)
        except ValueError:
            return 0.0
    
    return 0.0

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate the consolidated DataFrame."""
    # Replace infinite values with 0
    df = df.replace([np.inf, -np.inf], 0)
    
    # Fill NaN values with appropriate defaults
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(0)
    
    text_columns = df.select_dtypes(include=['object']).columns
    df[text_columns] = df[text_columns].fillna('')
    
    # Ensure year is properly formatted - FIXED REGEX
    if 'year' in df.columns:
        df['year'] = df['year'].astype(str).str.extract(r'(\d{4})').fillna('')
    
    # Calculate derived fields if missing
    df = calculate_derived_fields(df)
    
    return df

def fix_dataframe_types(df: pd.DataFrame) -> pd.DataFrame:
    """Fix DataFrame column types for Arrow compatibility and Excel export."""
    df_fixed = df.copy()
    
    # Define header columns that should remain as strings
    header_columns = ['company_name', 'year', 'period', 'auditor', 'opinion_type', 'currency', 'filename', 'extraction_date']
    text_columns = ['share_capital_changes', 'related_party_transactions', 'going_concern_issues', 'significant_estimates', 'subsequent_events']
    
    # Convert header and text columns to strings
    for col in header_columns + text_columns:
        if col in df_fixed.columns:
            df_fixed[col] = df_fixed[col].astype(str).replace(['nan', 'None', '0.0'], '')
            # Clean up the strings
            df_fixed[col] = df_fixed[col].apply(lambda x: x.strip() if x and x != 'nan' else '')
    
    # Ensure numeric columns are properly typed
    numeric_cols = df_fixed.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col not in header_columns + text_columns:  # Don't convert text columns to numeric
            df_fixed[col] = pd.to_numeric(df_fixed[col], errors='coerce').fillna(0)
    
    # Ensure remaining object columns that should be numeric are converted
    for col in df_fixed.select_dtypes(include=['object']).columns:
        if col not in header_columns + text_columns:
            # Try to convert to numeric if it looks like numbers
            try:
                df_fixed[col] = pd.to_numeric(df_fixed[col], errors='coerce').fillna(0)
            except:
                # If conversion fails, keep as string but clean it
                df_fixed[col] = df_fixed[col].astype(str).replace(['nan', 'None'], '')
    
    return df_fixed

def calculate_derived_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate derived financial fields where possible."""
    # Calculate gross profit if missing
    if all(col in df.columns for col in ['gross_profit', 'revenue', 'cost_of_goods_sold']):
        mask = (df['gross_profit'] == 0) & (df['revenue'] != 0) & (df['cost_of_goods_sold'] != 0)
        df.loc[mask, 'gross_profit'] = df.loc[mask, 'revenue'] - df.loc[mask, 'cost_of_goods_sold']
    
    # Calculate total assets if missing (sum of current and non-current)
    if all(col in df.columns for col in ['total_assets', 'total_current_assets', 'total_non_current_assets']):
        mask = (df['total_assets'] == 0) & (df['total_current_assets'] != 0) & (df['total_non_current_assets'] != 0)
        df.loc[mask, 'total_assets'] = df.loc[mask, 'total_current_assets'] + df.loc[mask, 'total_non_current_assets']
    
    # Calculate total liabilities if missing
    if all(col in df.columns for col in ['total_liabilities', 'total_current_liabilities', 'total_non_current_liabilities']):
        mask = (df['total_liabilities'] == 0) & (df['total_current_liabilities'] != 0) & (df['total_non_current_liabilities'] != 0)
        df.loc[mask, 'total_liabilities'] = df.loc[mask, 'total_current_liabilities'] + df.loc[mask, 'total_non_current_liabilities']
    
    # Verify balance sheet equation: Assets = Liabilities + Equity
    if all(col in df.columns for col in ['total_assets', 'total_liabilities', 'total_equity']):
        df['balance_sheet_check'] = df['total_assets'] - (df['total_liabilities'] + df['total_equity'])
        # Flag significant imbalances (more than 1% difference)
        df['balance_sheet_balanced'] = abs(df['balance_sheet_check']) < (df['total_assets'] * 0.01)
    
    return df

def create_summary_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Create summary statistics for the consolidated data."""
    if df.empty:
        return {}
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    summary = {
        "total_companies": len(df),
        "years_covered": sorted([year for year in df['year'].unique() if year and year != '']) if 'year' in df.columns else [],
        "currencies": [curr for curr in df['currency'].unique() if curr and curr != ''] if 'currency' in df.columns else [],
        
        # Financial aggregates
        "total_revenue": float(df['revenue'].sum()) if 'revenue' in df.columns else 0,
        "total_assets": float(df['total_assets'].sum()) if 'total_assets' in df.columns else 0,
        "total_equity": float(df['total_equity'].sum()) if 'total_equity' in df.columns else 0,
        "total_net_income": float(df['net_income'].sum()) if 'net_income' in df.columns else 0,
        
        # Average ratios
        "avg_current_ratio": float(df['current_ratio'].mean()) if 'current_ratio' in df.columns else 0,
        "avg_debt_to_equity": float(df['debt_to_equity'].mean()) if 'debt_to_equity' in df.columns else 0,
        "avg_profit_margin": float(df['net_profit_margin'].mean()) if 'net_profit_margin' in df.columns else 0,
        "avg_roe": float(df['roe'].mean()) if 'roe' in df.columns else 0,
        
        # Data quality metrics
        "companies_with_complete_bs": len(df[(df['total_assets'] > 0) & (df['total_liabilities'] > 0) & (df['total_equity'] > 0)]),
        "companies_with_complete_is": len(df[(df['revenue'] > 0) & (df['net_income'] != 0)]),
        "companies_with_complete_cf": len(df[df['operating_cash_flow'] != 0]) if 'operating_cash_flow' in df.columns else 0,
        "companies_with_header_info": len(df[df['company_name'].str.len() > 0]) if 'company_name' in df.columns else 0,
    }
    
    return summary

def export_to_excel(df: pd.DataFrame, all_data: List[Dict[str, Any]], filename: str = None) -> bytes:
    """Export consolidated data and individual company data to Excel."""
    if filename is None:
        filename = f"financial_consolidation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    
    excel_buffer = io.BytesIO()
    
    try:
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            # Write consolidated data
            df.to_excel(writer, sheet_name='Consolidated_Data', index=False)
            
            # Write summary statistics
            summary = create_summary_statistics(df)
            summary_df = pd.DataFrame(list(summary.items()), columns=['Metric', 'Value'])
            summary_df.to_excel(writer, sheet_name='Summary_Statistics', index=False)
            
            # Write individual company data
            for i, data in enumerate(all_data):
                try:
                    company_name = data.get("metadata", {}).get("filename", f"Company_{i+1}")
                    # Clean sheet name (Excel sheet names have limitations)
                    sheet_name = clean_sheet_name(company_name)
                    
                    # Flatten the nested data for Excel
                    flattened_data = flatten_dict(data)
                    company_df = pd.DataFrame([flattened_data])
                    
                    company_df.to_excel(writer, sheet_name=sheet_name, index=False)
                except Exception as e:
                    # If sheet name causes issues, use a generic name
                    try:
                        company_df = pd.DataFrame([flatten_dict(data)])
                        company_df.to_excel(writer, sheet_name=f'Company_{i+1}', index=False)
                    except:
                        continue  # Skip this company if all fails
        
        return excel_buffer.getvalue()
    
    except Exception as e:
        # Return empty buffer if Excel creation fails
        return b''

def clean_sheet_name(name: str) -> str:
    """Clean sheet name for Excel compatibility."""
    # Remove file extension
    name = name.replace('.pdf', '').replace('.PDF', '')
    
    # Replace invalid characters
    invalid_chars = ['[', ']', '*', '?', ':', '\\', '/']
    for char in invalid_chars:
        name = name.replace(char, '_')
    
    # Limit length (Excel limit is 31 characters)
    return name[:31]

def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
    """Flatten nested dictionary for Excel export."""
    items = []
    
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            # Ensure proper data type conversion
            if isinstance(v, (int, float)):
                items.append((new_key, float(v) if not pd.isna(v) else 0.0))
            else:
                items.append((new_key, str(v) if v is not None else ""))
    
    return dict(items)

def validate_data_integrity(df: pd.DataFrame) -> List[str]:
    """Validate data integrity and return list of issues."""
    issues = []
    
    if df.empty:
        issues.append("DataFrame is empty")
        return issues
    
    # Check for missing company names
    if 'company_name' in df.columns:
        missing_names = len(df[df['company_name'].str.len() == 0])
        if missing_names > 0:
            issues.append(f"{missing_names} companies missing company names")
    
    # Check balance sheet integrity
    if all(col in df.columns for col in ['total_assets', 'total_liabilities', 'total_equity']):
        unbalanced = df[abs(df['total_assets'] - (df['total_liabilities'] + df['total_equity'])) > (df['total_assets'] * 0.05)]
        if len(unbalanced) > 0:
            issues.append(f"{len(unbalanced)} companies with unbalanced balance sheets")
    
    # Check for negative current ratios
    if 'current_ratio' in df.columns:
        negative_ratios = len(df[df['current_ratio'] < 0])
        if negative_ratios > 0:
            issues.append(f"{negative_ratios} companies with negative current ratios")
    
    return issues

def get_data_quality_report(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate comprehensive data quality report."""
    report = {
        "total_records": len(df),
        "data_integrity_issues": validate_data_integrity(df),
        "completeness": {},
        "summary_stats": create_summary_statistics(df)
    }
    
    # Check completeness for key fields
    key_fields = ['company_name', 'year', 'revenue', 'total_assets', 'net_income']
    for field in key_fields:
        if field in df.columns:
            if field in ['company_name', 'year']:
                # For text fields, check for non-empty strings
                complete_count = len(df[df[field].str.len() > 0])
            else:
                # For numeric fields, check for non-zero values
                complete_count = len(df[df[field] != 0])
            
            report["completeness"][field] = {
                "complete_records": complete_count,
                "completion_rate": (complete_count / len(df)) * 100 if len(df) > 0 else 0
            }
    
    return report
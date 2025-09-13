"""
Financial Data Extractor Module - ALL PAGES VERSION
Enhanced to process ALL pages of PDF documents without limits
"""

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
import tempfile
import os
from datetime import datetime

class FinancialDataExtractor:
    def __init__(self, api_key: str):
        """Initialize the extractor with OpenAI API key."""
        self.client = openai.OpenAI(api_key=api_key)
        self.pdf_dpi = int(os.getenv("PDF_DPI", 200))
        self.max_tokens = int(os.getenv("MAX_TOKENS", 3500))
        self.temperature = float(os.getenv("TEMPERATURE", 0.1))
        
    def pdf_to_images(self, pdf_file) -> List[Image.Image]:
        """Convert ALL PDF pages to images for vision processing."""
        try:
            # Convert ALL PDF pages to images - NO PAGE LIMIT
            images = pdf2image.convert_from_bytes(
                pdf_file.read(),
                dpi=self.pdf_dpi,
                fmt='PNG',
                thread_count=1  # Reduced for stability
                # NO LIMITS: Process all pages
            )
            st.info(f"📄 Converted PDF to {len(images)} pages for processing")
            return images
        except Exception as e:
            st.error(f"Error converting PDF to images: {str(e)}")
            # Show troubleshooting info
            self._show_pdf_troubleshooting()
            return []
    
    def _show_pdf_troubleshooting(self):
        """Show PDF troubleshooting information."""
        with st.expander("🔧 PDF Processing Troubleshooting", expanded=True):
            st.markdown("""
            **PDF Conversion Failed. Try these solutions:**
            
            **Quick Fix:**
            ```bash
            pip install PyMuPDF
            ```
            
            **For Poppler Issues:**
            - **Windows**: `conda install -c conda-forge poppler`
            - **macOS**: `brew install poppler`  
            - **Linux**: `sudo apt-get install poppler-utils`
            
            **Alternative Solutions:**
            1. Try smaller PDF files (< 50MB)
            2. Convert PDF to images externally and upload images
            3. Use text-based PDFs (not scanned images)
            4. Split large PDFs into smaller files
            """)
    
    def encode_image(self, image: Image.Image) -> str:
        """Encode PIL Image to base64 string with optimization."""
        # Optimize image to reduce token usage and improve reliability
        max_dimension = 1024
        if image.width > max_dimension or image.height > max_dimension:
            # Calculate new dimensions while maintaining aspect ratio
            if image.width > image.height:
                new_width = max_dimension
                new_height = int((max_dimension * image.height) / image.width)
            else:
                new_height = max_dimension
                new_width = int((max_dimension * image.width) / image.height)
            
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", optimize=True, quality=90)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str
    
    def get_extraction_prompt(self) -> str:
        """Get the enhanced extraction prompt with better header extraction."""
        return """You are a financial analyst expert. Analyze this financial statement image and extract ALL information with high precision.

CRITICAL: You must respond with VALID JSON ONLY. No explanations, no markdown, no extra text - just pure JSON.

PAY SPECIAL ATTENTION to extracting header information:
- Company name (look at document title, header, letterhead - may be in Arabic and English)
- Period information (look for "for the period ended", "as at", "year ended", date ranges)
- Auditor name (audit firm name, usually at bottom of document or in audit opinion section)
- Audit opinion type (unqualified, qualified, adverse, disclaimer, clean opinion)

LOOK FOR THESE SPECIFIC PATTERNS:
- Company names often appear multiple times - get the full official name
- Periods like "For the year ended 31 December 2023", "As at 30 June 2023", "Quarter ended 31 March 2023"
- Auditor firms: KPMG, PricewaterhouseCoopers (PwC), Ernst & Young (EY), Deloitte, BDO, Grant Thornton, RSM, local audit firms
- Opinion types: "Unqualified Opinion", "Clean Opinion", "Qualified Opinion", "Adverse Opinion", "Disclaimer"

Return this exact structure with extracted values:

{
    "header_info": {
        "company_name": "Extract the full official company name from document header/title",
        "year": "Extract main year like 2023 or 2022",
        "period": "Extract full period description like 'For the year ended 31 December 2023' or 'As at 30 June 2023'",
        "auditor": "Extract audit firm name if visible",
        "opinion_type": "Extract audit opinion type if mentioned",
        "currency": "Extract currency (SAR, USD, etc.) or SAR as default"
    },
    "balance_sheet": {
        "assets": {
            "current_assets": {
                "cash_and_equivalents": 0,
                "accounts_receivable": 0,
                "inventory": 0,
                "prepaid_expenses": 0,
                "other_current_assets": 0,
                "total_current_assets": 0
            },
            "non_current_assets": {
                "property_plant_equipment": 0,
                "intangible_assets": 0,
                "investments": 0,
                "other_non_current_assets": 0,
                "total_non_current_assets": 0
            },
            "total_assets": 0
        },
        "liabilities": {
            "current_liabilities": {
                "accounts_payable": 0,
                "short_term_debt": 0,
                "accrued_expenses": 0,
                "other_current_liabilities": 0,
                "total_current_liabilities": 0
            },
            "non_current_liabilities": {
                "long_term_debt": 0,
                "deferred_tax": 0,
                "other_non_current_liabilities": 0,
                "total_non_current_liabilities": 0
            },
            "total_liabilities": 0
        },
        "equity": {
            "share_capital": 0,
            "retained_earnings": 0,
            "other_equity": 0,
            "total_equity": 0
        }
    },
    "income_statement": {
        "revenue": 0,
        "cost_of_goods_sold": 0,
        "gross_profit": 0,
        "operating_expenses": 0,
        "operating_income": 0,
        "interest_expense": 0,
        "interest_income": 0,
        "other_income": 0,
        "income_before_tax": 0,
        "tax_expense": 0,
        "net_income": 0
    },
    "cash_flow": {
        "operating_activities": {
            "net_income": 0,
            "depreciation": 0,
            "working_capital_changes": 0,
            "other_operating": 0,
            "net_cash_operating": 0
        },
        "investing_activities": {
            "capital_expenditures": 0,
            "acquisitions": 0,
            "investments": 0,
            "other_investing": 0,
            "net_cash_investing": 0
        },
        "financing_activities": {
            "debt_proceeds": 0,
            "debt_payments": 0,
            "equity_proceeds": 0,
            "dividends_paid": 0,
            "other_financing": 0,
            "net_cash_financing": 0
        },
        "net_change_cash": 0,
        "beginning_cash": 0,
        "ending_cash": 0
    },
    "notes_info": {
        "share_capital_changes": "",
        "related_party_transactions": "",
        "going_concern_issues": "",
        "significant_estimates": "",
        "subsequent_events": ""
    }
}

EXTRACTION RULES:
1. Numbers in parentheses (1,234) = negative -1234
2. Remove ALL commas: 1,234,567 → 1234567
3. Handle Saudi Riyal (SAR) formatting correctly
4. If field not visible = 0 (numbers) or "" (text) - DO NOT use 0 for text fields
5. Extract exact values as numbers
6. Look for Arabic/English mixed content
7. Alternative names: "Net Profit"="Net Income", "Fixed Assets"="Property Plant Equipment"
8. For header info, look at the TOP and BOTTOM of documents carefully

HEADER EXTRACTION IS CRITICAL - spend extra attention on:
- Document titles and headers
- Date information and period descriptions  
- Audit firm signatures and opinions
- Company names in multiple languages

RESPOND WITH JSON ONLY - NO OTHER TEXT."""
    
    def extract_financial_data(self, images: List[Image.Image], filename: str) -> Dict[str, Any]:
        """Extract comprehensive financial data from ALL PDF images using GPT-4 Vision."""
        
        if not images:
            st.error("❌ No images to process")
            return self._get_empty_data_structure(filename)
        
        extraction_prompt = self.get_extraction_prompt()
        
        all_extracted_data = {
            "header_info": {},
            "balance_sheet": {"assets": {}, "liabilities": {}, "equity": {}},
            "income_statement": {},
            "cash_flow": {},
            "notes_info": {}
        }
        
        successful_extractions = 0
        total_pages = len(images)
        
        # Create progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Show total pages being processed
        st.info(f"📖 Processing ALL {total_pages} pages from {filename}")
        
        # Process ALL pages - no limits
        for i, image in enumerate(images):
            try:
                progress = (i + 1) / total_pages
                progress_bar.progress(progress)
                status_text.text(f"🔍 Processing page {i+1}/{total_pages} using GPT-4 Vision...")
                
                # Use header-focused prompt for first page, standard for others
                if i == 0:
                    current_prompt = self.get_header_focused_prompt()
                else:
                    current_prompt = extraction_prompt
                
                # Encode image with optimization
                base64_image = self.encode_image(image)
                
                # Make API call with robust parsing
                extracted = self._make_robust_api_call(base64_image, current_prompt, i+1)
                
                if extracted:
                    # Merge data from this page
                    self._deep_merge_dict(all_extracted_data, extracted)
                    successful_extractions += 1
                    
                    # Show success with more detail for header info
                    company_name = extracted.get("header_info", {}).get("company_name", "")
                    period = extracted.get("header_info", {}).get("period", "")
                    auditor = extracted.get("header_info", {}).get("auditor", "")
                    
                    if company_name and i == 0:  # Show detailed info for first page
                        status_text.success(f"✅ Page {i+1}: Found {company_name}" + 
                                          (f" - {period}" if period else "") +
                                          (f" - Audited by {auditor}" if auditor else ""))
                    elif any([extracted.get("income_statement", {}).get("revenue", 0),
                             extracted.get("balance_sheet", {}).get("assets", {}).get("total_assets", 0),
                             extracted.get("cash_flow", {}).get("operating_activities", {}).get("net_cash_operating", 0)]):
                        status_text.success(f"✅ Page {i+1}: Financial data extracted")
                    else:
                        status_text.info(f"📄 Page {i+1}: Processed (limited data)")
                else:
                    status_text.warning(f"⚠️ Page {i+1}: No valid data extracted")
                
                # Brief pause to prevent API rate limiting
                if i > 0 and i % 10 == 0:  # Every 10 pages
                    status_text.info(f"⏸️ Brief pause after {i+1} pages to prevent rate limiting...")
                    import time
                    time.sleep(2)
                        
            except Exception as e:
                st.error(f"❌ Error processing page {i+1}: {str(e)}")
                continue
        
        # Final status
        progress_bar.progress(1.0)
        if successful_extractions > 0:
            status_text.success(f"🎉 Successfully processed {successful_extractions}/{total_pages} pages")
            
            # Show extracted header info summary
            header_info = all_extracted_data.get("header_info", {})
            if any(header_info.values()):
                st.info(f"📋 Extracted: {header_info.get('company_name', 'Unknown Company')} | " +
                       f"{header_info.get('period', 'No period')} | " +
                       f"Auditor: {header_info.get('auditor', 'Not found')}")
        else:
            status_text.error("❌ No pages were successfully processed")
        
        # Clean and validate extracted data
        all_extracted_data = self.clean_numerical_data(all_extracted_data)
        
        # Post-process header information
        all_extracted_data = self._post_process_header_info(all_extracted_data, filename)
        
        # Add metadata
        all_extracted_data["metadata"] = {
            "filename": filename,
            "extraction_date": datetime.now().isoformat(),
            "pages_processed": total_pages,
            "successful_extractions": successful_extractions
        }
        
        return all_extracted_data
    
    def get_header_focused_prompt(self) -> str:
        """Get a header-focused prompt for the first page."""
        return """You are a financial analyst expert. This appears to be the first page of a financial statement. 

FOCUS PRIMARILY ON HEADER INFORMATION - this is the most important task:

1. COMPANY NAME: Look at the very top of the document - company names often appear in large text, logos, letterheads
2. PERIOD: Look for phrases like "For the year ended...", "As at...", "Period ended...", date ranges
3. AUDITOR: Look at the bottom of the page or signature areas for audit firm names
4. OPINION: Look for audit opinion statements

Also extract any visible financial data, but PRIORITIZE the header information.

Return this JSON structure, paying special attention to header_info:

{
    "header_info": {
        "company_name": "EXTRACT THE FULL COMPANY NAME FROM THE TOP OF THE DOCUMENT",
        "year": "EXTRACT YEAR LIKE 2023",
        "period": "EXTRACT FULL PERIOD LIKE 'For the year ended 31 December 2023'",
        "auditor": "EXTRACT AUDITOR NAME IF VISIBLE",
        "opinion_type": "EXTRACT OPINION TYPE IF VISIBLE",
        "currency": "EXTRACT CURRENCY OR DEFAULT TO SAR"
    },
    "balance_sheet": {
        "assets": {
            "current_assets": {
                "cash_and_equivalents": 0,
                "accounts_receivable": 0,
                "inventory": 0,
                "total_current_assets": 0
            },
            "non_current_assets": {
                "property_plant_equipment": 0,
                "intangible_assets": 0,
                "total_non_current_assets": 0
            },
            "total_assets": 0
        },
        "liabilities": {
            "current_liabilities": {
                "accounts_payable": 0,
                "short_term_debt": 0,
                "total_current_liabilities": 0
            },
            "non_current_liabilities": {
                "long_term_debt": 0,
                "total_non_current_liabilities": 0
            },
            "total_liabilities": 0
        },
        "equity": {
            "share_capital": 0,
            "retained_earnings": 0,
            "total_equity": 0
        }
    },
    "income_statement": {
        "revenue": 0,
        "cost_of_goods_sold": 0,
        "gross_profit": 0,
        "operating_expenses": 0,
        "operating_income": 0,
        "net_income": 0
    },
    "cash_flow": {
        "operating_activities": {"net_cash_operating": 0},
        "investing_activities": {"net_cash_investing": 0},
        "financing_activities": {"net_cash_financing": 0},
        "net_change_cash": 0
    },
    "notes_info": {
        "share_capital_changes": "",
        "related_party_transactions": "",
        "going_concern_issues": ""
    }
}

CRITICAL: Look very carefully at the document header/title area for company name and date information.
If you see Arabic text, include both Arabic and English if both are present.
RESPOND WITH JSON ONLY - NO OTHER TEXT."""
    
    def _post_process_header_info(self, data: Dict[str, Any], filename: str) -> Dict[str, Any]:
        """Post-process header information to improve quality."""
        header = data.get("header_info", {})
        
        # Clean up company name
        if not header.get("company_name") or header.get("company_name") == "0":
            # Try to extract from filename
            company_from_file = self._extract_company_from_filename(filename)
            if company_from_file:
                header["company_name"] = company_from_file
        
        # Clean up period information
        period = header.get("period", "")
        if period and period != "0":
            # Normalize period format
            header["period"] = self._normalize_period(period)
        
        # Clean up year
        year = header.get("year", "")
        if not year or year == "0":
            # Try to extract year from period
            if period:
                year_match = re.search(r'\b(20\d{2})\b', period)
                if year_match:
                    header["year"] = year_match.group(1)
        
        # Ensure currency is set
        if not header.get("currency") or header.get("currency") == "0":
            header["currency"] = "SAR"  # Default to Saudi Riyal
        
        data["header_info"] = header
        return data
    
    def _extract_company_from_filename(self, filename: str) -> str:
        """Extract company name from filename."""
        try:
            # Remove file extension
            name = filename.replace('.pdf', '').replace('.PDF', '')
            
            # Remove common patterns
            name = re.sub(r'\b(financial|statement|annual|report|audit|audited)\b', '', name, flags=re.IGNORECASE)
            name = re.sub(r'\b(20\d{2})\b', '', name)  # Remove years
            name = re.sub(r'[_-]+', ' ', name)  # Replace underscores/dashes with spaces
            name = name.strip()
            
            if len(name) > 3:  # Only if we have a meaningful name
                return name
        except:
            pass
        return ""
    
    def _normalize_period(self, period: str) -> str:
        """Normalize period format for consistency."""
        try:
            # Common patterns to look for
            period = period.strip()
            
            # Handle numeric periods like "312023" -> "31 December 2023"
            numeric_match = re.match(r'^(\d{1,2})(\d{4})$', period.replace(' ', ''))
            if numeric_match:
                day = numeric_match.group(1)
                year = numeric_match.group(2)
                month_map = {'31': 'December', '30': 'June', '31': 'March'}
                month = month_map.get(day, 'December')
                return f"As at {day} {month} {year}"
            
            # Return as-is if already properly formatted
            return period
        except:
            return period
    
    def _make_robust_api_call(self, base64_image: str, prompt: str, page_num: int) -> Optional[Dict[str, Any]]:
        """Make API call with robust JSON parsing."""
        max_retries = 2
        
        for attempt in range(max_retries):
            try:
                # Make API call
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature
                )
                
                # Get response content
                content = response.choices[0].message.content.strip()
                
                # Try robust JSON parsing
                parsed_data = self._parse_json_robust(content, page_num)
                
                if parsed_data:
                    return parsed_data
                else:
                    if attempt == 0:  # Only show warning on first attempt
                        st.warning(f"Page {page_num}: Retrying JSON parsing...")
                    
            except Exception as e:
                if attempt == 0:
                    st.warning(f"API error on page {page_num}: {str(e)} - Retrying...")
                if attempt == max_retries - 1:
                    return None
                continue
        
        return None
    
    def _parse_json_robust(self, content: str, page_num: int) -> Optional[Dict[str, Any]]:
        """Robust JSON parsing with 5 different methods."""
        
        # Method 1: Direct JSON parsing
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        
        # Method 2: Extract from code blocks
        json_patterns = [
            r'```json\s*(\{.*?\})\s*```',
            r'```\s*(\{.*?\})\s*```',
            r'`(\{.*?\})`'
        ]
        
        for pattern in json_patterns:
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    continue
        
        # Method 3: Find JSON by braces
        start_idx = content.find('{')
        if start_idx != -1:
            brace_count = 0
            end_idx = start_idx
            
            for i, char in enumerate(content[start_idx:], start_idx):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i + 1
                        break
            
            if end_idx > start_idx:
                json_str = content[start_idx:end_idx]
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    pass
        
        # Method 4: Clean and retry
        try:
            # Remove common prefixes/suffixes
            cleaned = content.strip()
            prefixes = ["```json", "```", "Here's the JSON:", "JSON:", "Response:", "Here is the", "The JSON is:"]
            suffixes = ["```", "```json"]
            
            for prefix in prefixes:
                if cleaned.startswith(prefix):
                    cleaned = cleaned[len(prefix):].strip()
            
            for suffix in suffixes:
                if cleaned.endswith(suffix):
                    cleaned = cleaned[:-len(suffix)].strip()
            
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        
        # Method 5: Fallback - extract minimal data
        try:
            minimal_data = self._extract_fallback_data(content)
            if minimal_data:
                return minimal_data
        except:
            pass
        
        # Only show debug info if debug mode is enabled
        if os.getenv("DEBUG", "False").lower() == "true":
            with st.expander(f"🔍 Debug: Page {page_num} Raw Response (Failed JSON Parse)", expanded=False):
                st.text(content[:2000] + "..." if len(content) > 2000 else content)
        
        return None
    
    def _extract_fallback_data(self, content: str) -> Optional[Dict[str, Any]]:
        """Extract basic data using regex patterns when JSON parsing fails."""
        data = {
            "header_info": {},
            "balance_sheet": {"assets": {}, "liabilities": {}, "equity": {}},
            "income_statement": {},
            "cash_flow": {},
            "notes_info": {}
        }
        
        found_data = False
        
        # Extract company name - enhanced patterns
        company_patterns = [
            r'company[_\s]*name["\s:]+([^",\n]+)',
            r'"company_name":\s*"([^"]+)"',
            r'Company:\s*([^,\n]+)',
            r'([A-Z][A-Za-z\s&]+(?:Company|Corporation|Corp|LLC|Ltd|Limited))',
            r'شركة\s+([^،\n]+)',  # Arabic pattern
            r'COMPANY[:\s]+([A-Z][A-Za-z\s&]+)'
        ]
        
        for pattern in company_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                company_name = match.group(1).strip().replace('"', '')
                if len(company_name) > 3:  # Meaningful name
                    data["header_info"]["company_name"] = company_name
                    found_data = True
                    break
        
        # Extract period information
        period_patterns = [
            r'"period":\s*"([^"]+)"',
            r'period[:\s]+([^,\n]+)',
            r'for the (?:year|period) ended[:\s]+([^,\n]+)',
            r'as at[:\s]+([^,\n]+)',
            r'(\d{1,2}\s+\w+\s+\d{4})'
        ]
        
        for pattern in period_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                period = match.group(1).strip().replace('"', '')
                if len(period) > 4:  # Meaningful period
                    data["header_info"]["period"] = period
                    found_data = True
                    break
        
        # Extract year
        year_match = re.search(r'"year":\s*"?(\d{4})"?|\b(20\d{2})\b', content)
        if year_match:
            data["header_info"]["year"] = year_match.group(1) or year_match.group(2)
            found_data = True
        
        # Extract auditor
        auditor_patterns = [
            r'"auditor":\s*"([^"]+)"',
            r'auditor[:\s]+([^,\n]+)',
            r'(KPMG|PricewaterhouseCoopers|PwC|Ernst & Young|EY|Deloitte|BDO|Grant Thornton|RSM)',
            r'audited by[:\s]+([^,\n]+)'
        ]
        
        for pattern in auditor_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                auditor = match.group(1).strip().replace('"', '')
                if len(auditor) > 2:
                    data["header_info"]["auditor"] = auditor
                    found_data = True
                    break
        
        # Extract key financial numbers
        number_patterns = {
            'revenue': [
                r'"revenue":\s*(\d+(?:,\d{3})*)',
                r'revenue["\s:]+(\d+(?:,\d{3})*)',
                r'sales["\s:]+(\d+(?:,\d{3})*)'
            ],
            'total_assets': [
                r'"total_assets":\s*(\d+(?:,\d{3})*)',
                r'total\s+assets["\s:]+(\d+(?:,\d{3})*)'
            ],
            'net_income': [
                r'"net_income":\s*(-?\d+(?:,\d{3})*)',
                r'net\s+income["\s:]+(-?\d+(?:,\d{3})*)',
                r'net\s+profit["\s:]+(-?\d+(?:,\d{3})*)'
            ]
        }
        
        for field, patterns in number_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    try:
                        value = float(match.group(1).replace(',', ''))
                        if field == 'revenue':
                            data["income_statement"]["revenue"] = value
                        elif field == 'total_assets':
                            data["balance_sheet"]["assets"]["total_assets"] = value
                        elif field == 'net_income':
                            data["income_statement"]["net_income"] = value
                        found_data = True
                        break
                    except ValueError:
                        continue
        
        return data if found_data else None
    
    def _get_empty_data_structure(self, filename: str) -> Dict[str, Any]:
        """Return empty data structure when processing fails."""
        return {
            "header_info": {},
            "balance_sheet": {"assets": {}, "liabilities": {}, "equity": {}},
            "income_statement": {},
            "cash_flow": {},
            "notes_info": {},
            "metadata": {
                "filename": filename,
                "extraction_date": datetime.now().isoformat(),
                "pages_processed": 0,
                "successful_extractions": 0,
                "extraction_failed": True
            }
        }
    
    def _deep_merge_dict(self, target: dict, source: dict) -> None:

        """Recursively merge source dict into target dict with header priority."""
        for key, value in source.items():
            if key == "header_info" and isinstance(value, dict):
                # Special handling for header info - always preserve non-empty values
                if "header_info" not in target:
                    target["header_info"] = {}
            
                for header_key, header_value in value.items():
                    if header_value and str(header_value).strip() and str(header_value) not in ["0", ""]:
                        # Always update if target doesn't have this field or has empty value
                        current_value = target["header_info"].get(header_key, "")
                        if not current_value or str(current_value).strip() in ["", "0"]:
                            target["header_info"][header_key] = header_value
                        
            elif isinstance(value, dict) and key in target and isinstance(target[key], dict):
                self._deep_merge_dict(target[key], value)
            else:
                # For non-header data, only update if the new value is meaningful
                if (isinstance(value, str) and value.strip() and value != "0") or \
                (isinstance(value, (int, float)) and value != 0) or \
                (isinstance(value, dict) and value):
                    target[key] = value
    
    def calculate_financial_ratios(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate key financial ratios from extracted data."""
        ratios = {}
        
        try:
            # Balance Sheet ratios
            bs = data.get("balance_sheet", {})
            assets = bs.get("assets", {})
            liabilities = bs.get("liabilities", {})
            equity = bs.get("equity", {})
            
            total_assets = self._safe_float(assets.get("total_assets", 0))
            total_liabilities = self._safe_float(liabilities.get("total_liabilities", 0))
            total_equity = self._safe_float(equity.get("total_equity", 0))
            
            current_assets = self._safe_float(assets.get("current_assets", {}).get("total_current_assets", 0))
            current_liabilities = self._safe_float(liabilities.get("current_liabilities", {}).get("total_current_liabilities", 0))
            
            # Liquidity ratios
            if current_liabilities > 0:
                ratios["current_ratio"] = current_assets / current_liabilities
            
            # Quick ratio
            cash = self._safe_float(assets.get("current_assets", {}).get("cash_and_equivalents", 0))
            receivables = self._safe_float(assets.get("current_assets", {}).get("accounts_receivable", 0))
            if current_liabilities > 0:
                ratios["quick_ratio"] = (cash + receivables) / current_liabilities
            
            # Leverage ratios
            if total_assets > 0:
                ratios["debt_to_assets"] = total_liabilities / total_assets
                ratios["equity_ratio"] = total_equity / total_assets
            
            if total_equity > 0:
                ratios["debt_to_equity"] = total_liabilities / total_equity
            
            # Income Statement ratios
            income = data.get("income_statement", {})
            revenue = self._safe_float(income.get("revenue", 0))
            net_income = self._safe_float(income.get("net_income", 0))
            gross_profit = self._safe_float(income.get("gross_profit", 0))
            operating_income = self._safe_float(income.get("operating_income", 0))
            
            # Profitability ratios
            if revenue > 0:
                ratios["net_profit_margin"] = (net_income / revenue) * 100
                ratios["gross_profit_margin"] = (gross_profit / revenue) * 100
                ratios["operating_margin"] = (operating_income / revenue) * 100
            
            if total_assets > 0:
                ratios["roa"] = (net_income / total_assets) * 100
            
            if total_equity > 0:
                ratios["roe"] = (net_income / total_equity) * 100
            
            # Asset turnover
            if total_assets > 0:
                ratios["asset_turnover"] = revenue / total_assets
            
            # Working capital
            ratios["working_capital"] = current_assets - current_liabilities
            
        except Exception as e:
            st.warning(f"Error calculating ratios: {str(e)}")
        
        return ratios
    
    def _safe_float(self, value) -> float:
        """Safely convert value to float."""
        try:
            if isinstance(value, (int, float)):
                return float(value) if not pd.isna(value) else 0.0
            elif isinstance(value, str):
                # Clean string and convert
                cleaned = re.sub(r'[^\d.-]', '', value.replace(',', ''))
                return float(cleaned) if cleaned and cleaned != '-' else 0.0
            else:
                return 0.0
        except (ValueError, TypeError):
            return 0.0
    
    def clean_numerical_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and normalize numerical data in the extracted information."""
        def clean_number(value):
            if isinstance(value, str):
                # Handle empty strings
                if not value.strip():
                    return 0
                
                # Remove currency symbols and common text
                value = value.replace('SAR', '').replace('€', '')
                value = value.replace('Million', '000000').replace('million', '000000')
                value = value.replace('Thousand', '000').replace('thousand', '000')
                value = value.replace('K', '000').replace('M', '000000').replace('B', '000000000')
                
                # Remove parentheses and treat as negative
                if value.strip().startswith('(') and value.strip().endswith(')'):
                    value = '-' + value.strip()[1:-1]
                
                # Remove commas, spaces, and other formatting except digits, dots, and minus
                value = re.sub(r'[^\d.-]', '', value)
                
                # Handle empty result or just minus sign
                if not value or value == '-':
                    return 0
                
                # Try to convert to float
                try:
                    return float(value)
                except ValueError:
                    return 0
            elif isinstance(value, (int, float)):
                return float(value) if not pd.isna(value) else 0.0
            else:
                return value
        
        def clean_dict(d):
            if isinstance(d, dict):
                return {k: clean_dict(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [clean_dict(item) for item in d]
            else:
                return clean_number(d)
        
        return clean_dict(data)
    
    def estimate_processing_time(self, num_pages: int) -> str:
        """Estimate processing time based on number of pages."""
        # Rough estimate: 10-15 seconds per page including API calls
        estimated_seconds = num_pages * 12
        
        if estimated_seconds < 60:
            return f"~{estimated_seconds} seconds"
        elif estimated_seconds < 3600:
            return f"~{estimated_seconds // 60} minutes"
        else:
            hours = estimated_seconds // 3600
            minutes = (estimated_seconds % 3600) // 60
            return f"~{hours}h {minutes}m"
    
    def show_processing_tips(self, total_pages: int):
        """Show tips for processing large documents."""
        if total_pages > 50:
            st.warning("⚠️ Large document detected! Processing tips:")
            st.info(f"""
            📊 **Processing {total_pages} pages will take approximately {self.estimate_processing_time(total_pages)}**
            
            💡 **Tips for better results:**
            - Ensure stable internet connection
            - Don't close the browser tab during processing
            - Consider processing in smaller batches if issues occur
            - The system will automatically pause every 10 pages to prevent rate limiting
            """)
        elif total_pages > 20:
            st.info(f"📄 Processing {total_pages} pages (estimated time: {self.estimate_processing_time(total_pages)})")
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get processing statistics for monitoring."""
        return {
            "max_tokens": self.max_tokens,
            "pdf_dpi": self.pdf_dpi,
            "temperature": self.temperature,
            "model": "gpt-4o",
            "features": [
                "All pages processing",
                "Header extraction",
                "Financial data extraction", 
                "Ratio calculations",
                "Multi-language support",
                "Robust error handling"
            ]
        }
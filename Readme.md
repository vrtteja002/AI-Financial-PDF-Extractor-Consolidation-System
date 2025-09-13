# 📊 Financial PDF Extractor & Consolidation System

A powerful AI-driven application that extracts, analyzes, and consolidates financial data from PDF statements using GPT-4 Vision API. Transform your financial document processing workflow with intelligent automation and comprehensive analytics.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.28+-red.svg)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4%20Vision-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 🌟 Features

### 📄 Document Processing
- **All-Page Processing**: Extracts data from ALL pages of PDF documents (no page limits)
- **Multi-Language Support**: Handles Arabic/English mixed content
- **Robust PDF Conversion**: Advanced PDF-to-image conversion with error handling
- **Batch Processing**: Process up to 23 PDF files simultaneously

### 🧠 AI-Powered Extraction
- **GPT-4 Vision Integration**: Uses OpenAI's most advanced vision model
- **Header Information Extraction**: Company names, periods, auditors, opinion types
- **Financial Statement Analysis**: Balance sheets, income statements, cash flow statements
- **Smart Data Validation**: Automatic balance sheet equation verification

### 📊 Financial Analytics
- **Financial Ratio Calculations**: 12+ key financial ratios automatically computed
- **Risk Analysis Dashboard**: Comprehensive risk assessment and categorization
- **Performance Scoring**: Multi-dimensional performance evaluation system
- **Interactive Visualizations**: Professional charts and dashboards using Plotly

### 💾 Data Export & Management
- **Multiple Export Formats**: Excel (multi-sheet), CSV, JSON
- **Session State Management**: Persistent data across app sessions
- **Data Quality Reports**: Comprehensive validation and quality metrics
- **Download Ready Files**: Timestamped exports with detailed breakdowns

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- OpenAI API key with GPT-4 Vision access
- Poppler utilities (for PDF processing)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/financial-pdf-extractor.git
   cd financial-pdf-extractor
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Poppler (PDF processing)**
   
   **Windows:**
   ```bash
   conda install -c conda-forge poppler
   ```
   
   **macOS:**
   ```bash
   brew install poppler
   ```
   
   **Linux:**
   ```bash
   sudo apt-get install poppler-utils
   ```

4. **Set up environment variables**
   ```bash
   # Create .env file
   cp .env.example .env
   
   # Edit .env and add your OpenAI API key
   OPENAI_API_KEY=your_openai_api_key_here
   ```

5. **Run the application**
   ```bash
   python run.py
   ```
   
   Or directly with Streamlit:
   ```bash
   streamlit run app.py
   ```

6. **Open your browser**
   
   Navigate to `http://localhost:8501`

## 📁 Project Structure

```
financial-pdf-extractor/
├── app.py                    # Main Streamlit application
├── financial_extractor.py   # Core extraction logic with GPT-4 Vision
├── data_processor.py        # Data consolidation and processing
├── visualizations.py        # Interactive charts and dashboards
├── utils.py                 # Utility functions and helpers
├── run.py                   # Application launcher script
├── requirements.txt         # Python dependencies
├── .env.example             # Environment variables template
└── README.md               # This file
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | Your OpenAI API key | Required |
| `MAX_FILES` | Maximum files per batch | 23 |
| `MAX_FILE_SIZE_MB` | Maximum file size | 50 |
| `PDF_DPI` | PDF conversion quality | 200 |
| `MAX_TOKENS` | GPT-4 response limit | 3500 |
| `TEMPERATURE` | AI response creativity | 0.1 |
| `DEBUG` | Enable debug mode | False |

### Customization

- **Processing Limits**: Adjust file limits in `.env`
- **AI Parameters**: Tune extraction prompts in `financial_extractor.py`
- **Visualization Themes**: Modify chart styles in `visualizations.py`
- **Data Schema**: Extend extraction fields in `data_processor.py`

## 📊 Extracted Data Points

### Header Information
- Company name (multilingual support)
- Reporting period and year
- Auditor name and opinion type
- Currency and document metadata

### Balance Sheet
- **Assets**: Current/non-current breakdown, cash, receivables, inventory, PPE
- **Liabilities**: Current/non-current breakdown, payables, debt obligations
- **Equity**: Share capital, retained earnings, comprehensive breakdown

### Income Statement
- Revenue, cost of goods sold, gross profit
- Operating expenses and operating income
- Interest income/expense, taxes
- Net income and comprehensive income

### Cash Flow Statement
- **Operating Activities**: Net income, depreciation, working capital changes
- **Investing Activities**: Capital expenditures, acquisitions, investments
- **Financing Activities**: Debt transactions, equity changes, dividends

### Financial Ratios (Auto-Calculated)
- **Liquidity**: Current ratio, quick ratio
- **Leverage**: Debt-to-equity, debt-to-assets, equity ratio
- **Profitability**: Net/gross/operating margins, ROA, ROE
- **Efficiency**: Asset turnover, working capital

## 📈 Analytics & Visualizations

### Interactive Dashboards
- **Overview Dashboard**: Revenue comparisons, asset-revenue scatter plots
- **Profitability Analysis**: Margin comparisons, ROA vs ROE analysis
- **Financial Position**: Asset composition, capital structure
- **Risk Analysis**: Liquidity vs leverage matrix, risk categorization
- **Performance Metrics**: Multi-dimensional scoring and ranking

### Risk Assessment
- **Automated Risk Flags**: Liquidity, leverage, profitability warnings
- **Risk Categorization**: Low, moderate, high, critical risk classification
- **Performance Scoring**: Comprehensive 0-100 performance evaluation

## 🔬 Technical Architecture

### Core Components

1. **PDF Processing Pipeline**
   - PDF → Images conversion using `pdf2image`
   - Image optimization for API efficiency
   - Error handling and retry mechanisms

2. **AI Extraction Engine**
   - GPT-4 Vision API integration
   - Robust JSON parsing with 5 fallback methods
   - Header-focused extraction for first pages
   - Progressive data consolidation across pages

3. **Data Processing Layer**
   - Nested dictionary flattening
   - Financial ratio calculations
   - Data validation and integrity checks
   - Multi-format export capabilities

4. **Visualization Engine**
   - Plotly-based interactive charts
   - Responsive design principles
   - Real-time data updates
   - Export-ready visualizations

### Performance Optimizations

- **Image Compression**: Automatic image optimization for API calls
- **Smart Pagination**: Efficient processing of large documents
- **Memory Management**: Session-based data persistence
- **Error Recovery**: Robust error handling and continuation

## 🔮 Roadmap & Future Enhancements

### Planned Features

#### 🚀 Performance Improvements
- **Parallel Processing**: Multi-threaded PDF processing for 3-5x speed improvement
  - *Note: Currently limited by hardware compatibility - will be implemented when compatible hardware is available*
- **Batch API Calls**: Simultaneous processing of multiple pages
- **Caching System**: Smart caching for repeated document processing
- **GPU Acceleration**: CUDA support for faster image processing

#### 🧠 AI & ML Enhancements
- **Custom Fine-Tuned Models**: Domain-specific financial models
- **OCR Integration**: Fallback OCR for low-quality PDFs
- **Anomaly Detection**: AI-powered data validation
- **Predictive Analytics**: Financial forecasting capabilities

#### 📊 Advanced Analytics
- **Benchmarking**: Industry comparison capabilities
- **Trend Analysis**: Multi-period comparative analysis
- **ESG Metrics**: Environmental, social, governance indicators
- **Compliance Checks**: Regulatory requirement validation

#### 🔧 Technical Enhancements
- **API Endpoints**: RESTful API for programmatic access
- **Database Integration**: PostgreSQL/MongoDB support
- **Cloud Deployment**: Docker containerization
- **Real-time Processing**: WebSocket-based live updates

### Contributing

We welcome contributions! Areas where help is especially appreciated:

- **Performance Optimization**: Parallel processing implementation
- **New Visualizations**: Additional chart types and dashboards
- **Data Validation**: Enhanced accuracy checks
- **Documentation**: Tutorials and examples
- **Testing**: Unit tests and integration tests

## 🐛 Troubleshooting

### Common Issues

**PDF Conversion Fails**
```bash
# Install Poppler
# Windows: conda install -c conda-forge poppler
# macOS: brew install poppler
# Linux: sudo apt-get install poppler-utils
```

**API Rate Limits**
- The system automatically pauses every 10 pages
- Reduce `MAX_TOKENS` in `.env` if needed
- Consider upgrading your OpenAI plan

**Memory Issues with Large PDFs**
- Process files in smaller batches
- Reduce `PDF_DPI` in `.env`
- Split large PDFs before processing

**Extraction Accuracy Issues**
- Ensure PDFs are high-quality (not heavily compressed)
- Try different `TEMPERATURE` settings
- Check that text is clearly visible in the PDF

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Support

- **Documentation**: Check this README and inline code comments
- **Issues**: Report bugs via GitHub Issues
- **Discussions**: Use GitHub Discussions for questions
- **Email**: [your-email@domain.com] for private inquiries

## 🙏 Acknowledgments

- **OpenAI**: For the powerful GPT-4 Vision API
- **Streamlit**: For the excellent web framework
- **Plotly**: For beautiful interactive visualizations
- **pdf2image**: For reliable PDF processing
- **Community**: All contributors and users providing feedback

---

**⭐ Star this repository if you find it useful!**

Built with ❤️ for the financial technology community
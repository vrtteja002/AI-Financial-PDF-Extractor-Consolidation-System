"""
Fixed Visualizations Module
Creates interactive charts and dashboards for financial analysis - FIXED VERSION
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any
import io
import base64

def create_visualizations(df: pd.DataFrame):
    """Create comprehensive interactive visualizations for financial analysis."""
    
    if df.empty:
        st.warning("No data available for visualization.")
        return
    
    # Filter out companies with no data and handle NaN values
    df_filtered = df.copy()
    
    # Fill NaN values to prevent errors
    numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns
    df_filtered[numeric_cols] = df_filtered[numeric_cols].fillna(0)
    
    text_cols = df_filtered.select_dtypes(include=['object']).columns
    df_filtered[text_cols] = df_filtered[text_cols].fillna('')
    
    # Filter companies with meaningful data
    df_filtered = df_filtered[
        (df_filtered['revenue'] > 0) | (df_filtered['total_assets'] > 0)
    ].copy()
    
    if df_filtered.empty:
        st.warning("No companies with financial data found for visualization.")
        return
    
    # Create risk dashboard first (overview)
    create_risk_dashboard(df_filtered)
    
    # Create visualization tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", "💰 Profitability", "🏦 Financial Position", 
        "💧 Liquidity & Leverage", "📈 Performance Metrics"
    ])
    
    with tab1:
        create_overview_charts(df_filtered)
    
    with tab2:
        create_profitability_charts(df_filtered)
    
    with tab3:
        create_financial_position_charts(df_filtered)
    
    with tab4:
        create_liquidity_leverage_charts(df_filtered)
    
    with tab5:
        create_performance_metrics(df_filtered)

def create_overview_charts(df: pd.DataFrame):
    """Create overview charts showing key metrics."""
    st.subheader("📊 Financial Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Revenue comparison
        if 'revenue' in df.columns and df['revenue'].sum() > 0:
            revenue_df = df[df['revenue'] > 0].nlargest(10, 'revenue')
            if not revenue_df.empty:
                fig_revenue = px.bar(
                    revenue_df, 
                    x="company_name", 
                    y="revenue", 
                    title="Top 10 Companies by Revenue",
                    color="revenue",
                    color_continuous_scale="viridis",
                    hover_data=["year", "net_income"]
                )
                fig_revenue.update_layout(
                    xaxis_tickangle=-45,
                    height=500,
                    showlegend=False
                )
                st.plotly_chart(fig_revenue, use_container_width=True)
    
    with col2:
        # Assets vs Revenue scatter
        if all(col in df.columns for col in ['total_assets', 'revenue']):
            scatter_df = df[(df['total_assets'] > 0) & (df['revenue'] > 0)]
            if not scatter_df.empty:
                fig_scatter = px.scatter(
                    scatter_df,
                    x="total_assets",
                    y="revenue",
                    size= np.maximum(scatter_df["net_income"].abs(), 1)
                    ,
                    color="year",
                    hover_name="company_name",
                    title="Assets vs Revenue",
                    hover_data=["current_ratio", "debt_to_equity"]
                )
                fig_scatter.update_layout(height=500)
                st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Market composition pie chart
    if 'revenue' in df.columns and df['revenue'].sum() > 0:
        # Group smaller companies into "Others"
        df_pie = df[df['revenue'] > 0].copy()
        if not df_pie.empty:
            total_revenue = df_pie['revenue'].sum()
            df_pie['revenue_pct'] = (df_pie['revenue'] / total_revenue) * 100
            
            # Companies with less than 3% grouped as "Others"
            mask = df_pie['revenue_pct'] < 3
            others_revenue = df_pie[mask]['revenue'].sum()
            df_pie = df_pie[~mask]
            
            if others_revenue > 0:
                others_row = pd.DataFrame({
                    'company_name': ['Others'],
                    'revenue': [others_revenue]
                })
                df_pie = pd.concat([df_pie, others_row], ignore_index=True)
            
            fig_pie = px.pie(
                df_pie,
                values="revenue",
                names="company_name",
                title="Revenue Distribution by Company"
            )
            fig_pie.update_layout(height=400)
            st.plotly_chart(fig_pie, use_container_width=True)

def create_profitability_charts(df: pd.DataFrame):
    """Create profitability analysis charts."""
    st.subheader("💰 Profitability Analysis")
    
    # Profitability metrics comparison
    profitability_cols = ['net_profit_margin', 'gross_profit_margin', 'operating_margin']
    available_cols = [col for col in profitability_cols if col in df.columns]
    
    if available_cols:
        fig_prof = make_subplots(
            rows=1, cols=len(available_cols),
            subplot_titles=[col.replace('_', ' ').title() + ' (%)' for col in available_cols],
        )
        
        for i, metric in enumerate(available_cols, 1):
            df_metric = df[(df[metric] != 0) & (pd.notnull(df[metric]))].nlargest(10, metric)
            
            if not df_metric.empty:
                fig_prof.add_trace(
                    go.Bar(
                        x=df_metric["company_name"], 
                        y=df_metric[metric], 
                        name=metric.replace('_', ' ').title(),
                        showlegend=False
                    ),
                    row=1, col=i
                )
                
                # Update x-axis for each subplot
                fig_prof.update_xaxes(tickangle=-45, row=1, col=i)
        
        fig_prof.update_layout(
            title_text="Profitability Metrics Comparison",
            height=500
        )
        st.plotly_chart(fig_prof, use_container_width=True)
    
    # ROA vs ROE scatter plot
    if all(col in df.columns for col in ['roa', 'roe']):
        df_returns = df[(df['roa'] != 0) | (df['roe'] != 0)]
        df_returns = df_returns[(pd.notnull(df_returns['roa'])) & (pd.notnull(df_returns['roe']))]
        
        if not df_returns.empty:
            fig_returns = px.scatter(
                df_returns,
                x="roa",
                y="roe",
                size="revenue",
                color="net_profit_margin",
                hover_name="company_name",
                title="Return on Assets vs Return on Equity",
                labels={"roa": "ROA (%)", "roe": "ROE (%)"},
                color_continuous_scale="RdYlGn"
            )
            fig_returns.add_hline(y=0, line_dash="dash", line_color="red")
            fig_returns.add_vline(x=0, line_dash="dash", line_color="red")
            fig_returns.update_layout(height=500)
            st.plotly_chart(fig_returns, use_container_width=True)
    
    # Net income trend (if multiple years available)
    if 'year' in df.columns and len(df['year'].unique()) > 1:
        df_trend = df.groupby(['company_name', 'year'])['net_income'].sum().reset_index()
        if not df_trend.empty:
            fig_trend = px.line(
                df_trend,
                x="year",
                y="net_income",
                color="company_name",
                title="Net Income Trends Over Time"
            )
            fig_trend.update_layout(height=500)
            st.plotly_chart(fig_trend, use_container_width=True)

def create_financial_position_charts(df: pd.DataFrame):
    """Create financial position analysis charts."""
    st.subheader("🏦 Financial Position Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Assets composition stacked bar
        asset_cols = ['total_current_assets', 'total_non_current_assets']
        if all(col in df.columns for col in asset_cols):
            df_assets = df[(df['total_assets'] > 0)].nlargest(10, 'total_assets')
            
            if not df_assets.empty:
                fig_assets = go.Figure()
                
                fig_assets.add_trace(go.Bar(
                    name='Current Assets',
                    x=df_assets['company_name'],
                    y=df_assets['total_current_assets']
                ))
                
                fig_assets.add_trace(go.Bar(
                    name='Non-Current Assets',
                    x=df_assets['company_name'],
                    y=df_assets['total_non_current_assets']
                ))
                
                fig_assets.update_layout(
                    barmode='stack',
                    title='Asset Composition by Company',
                    xaxis_tickangle=-45,
                    height=500
                )
                st.plotly_chart(fig_assets, use_container_width=True)
    
    with col2:
        # Liabilities and equity composition
        if all(col in df.columns for col in ['total_liabilities', 'total_equity']):
            df_structure = df[(df['total_assets'] > 0)].nlargest(10, 'total_assets')
            
            if not df_structure.empty:
                fig_structure = go.Figure()
                
                fig_structure.add_trace(go.Bar(
                    name='Total Liabilities',
                    x=df_structure['company_name'],
                    y=df_structure['total_liabilities']
                ))
                
                fig_structure.add_trace(go.Bar(
                    name='Total Equity',
                    x=df_structure['company_name'],
                    y=df_structure['total_equity']
                ))
                
                fig_structure.update_layout(
                    barmode='stack',
                    title='Capital Structure (Liabilities vs Equity)',
                    xaxis_tickangle=-45,
                    height=500
                )
                st.plotly_chart(fig_structure, use_container_width=True)

def create_liquidity_leverage_charts(df: pd.DataFrame):
    """Create liquidity and leverage analysis charts."""
    st.subheader("💧 Liquidity & Leverage Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Current ratio analysis
        if 'current_ratio' in df.columns:
            df_liquidity = df[(df['current_ratio'] > 0) & (pd.notnull(df['current_ratio']))].copy()
            if not df_liquidity.empty:
                fig_current = px.histogram(
                    df_liquidity,
                    x="current_ratio",
                    nbins=20,
                    title="Distribution of Current Ratios",
                    color_discrete_sequence=['lightblue']
                )
                fig_current.add_vline(x=1.0, line_dash="dash", line_color="red", 
                                    annotation_text="Benchmark: 1.0")
                if len(df_liquidity) > 0:
                    avg_ratio = df_liquidity['current_ratio'].mean()
                    fig_current.add_vline(x=avg_ratio, 
                                        line_dash="dash", line_color="green", 
                                        annotation_text=f"Average: {avg_ratio:.2f}")
                st.plotly_chart(fig_current, use_container_width=True)
    
    with col2:
        # Debt-to-equity analysis
        if 'debt_to_equity' in df.columns:
            df_leverage = df[(df['debt_to_equity'] > 0) & (pd.notnull(df['debt_to_equity']))].copy()
            if not df_leverage.empty:
                fig_debt = px.histogram(
                    df_leverage,
                    x="debt_to_equity",
                    nbins=20,
                    title="Distribution of Debt-to-Equity Ratios",
                    color_discrete_sequence=['lightcoral']
                )
                fig_debt.add_vline(x=1.0, line_dash="dash", line_color="red", 
                                 annotation_text="Benchmark: 1.0")
                if len(df_leverage) > 0:
                    avg_debt = df_leverage['debt_to_equity'].mean()
                    fig_debt.add_vline(x=avg_debt, 
                                     line_dash="dash", line_color="green", 
                                     annotation_text=f"Average: {avg_debt:.2f}")
                st.plotly_chart(fig_debt, use_container_width=True)
    
    # Risk matrix: Current Ratio vs Debt-to-Equity
    if all(col in df.columns for col in ['current_ratio', 'debt_to_equity']):
        df_risk = df[(df['current_ratio'] > 0) & (df['debt_to_equity'] > 0) & 
                    (pd.notnull(df['current_ratio'])) & (pd.notnull(df['debt_to_equity']))]
        
        if not df_risk.empty:
            fig_risk = px.scatter(
                df_risk,
                x="debt_to_equity",
                y="current_ratio",
                size="total_assets",
                color="net_profit_margin",
                hover_name="company_name",
                title="Financial Risk Matrix: Liquidity vs Leverage",
                labels={"debt_to_equity": "Debt-to-Equity Ratio", "current_ratio": "Current Ratio"},
                color_continuous_scale="RdYlGn"
            )
            
            # Add quadrant lines
            fig_risk.add_hline(y=1.0, line_dash="dash", line_color="gray", opacity=0.5)
            fig_risk.add_vline(x=1.0, line_dash="dash", line_color="gray", opacity=0.5)
            
            # Add quadrant annotations
            fig_risk.add_annotation(x=0.5, y=2.0, text="Low Risk", showarrow=False, 
                                  bgcolor="lightgreen", opacity=0.7)
            fig_risk.add_annotation(x=2.0, y=2.0, text="Moderate Risk", showarrow=False, 
                                  bgcolor="yellow", opacity=0.7)
            fig_risk.add_annotation(x=0.5, y=0.5, text="Liquidity Risk", showarrow=False, 
                                  bgcolor="orange", opacity=0.7)
            fig_risk.add_annotation(x=2.0, y=0.5, text="High Risk", showarrow=False, 
                                  bgcolor="lightcoral", opacity=0.7)
            
            fig_risk.update_layout(height=500)
            st.plotly_chart(fig_risk, use_container_width=True)

def create_performance_metrics(df: pd.DataFrame):
    """Create performance metrics dashboard."""
    st.subheader("📈 Performance Metrics Dashboard")
    
    # Create performance score
    df_perf = calculate_performance_scores(df)
    
    if not df_perf.empty:
        # Performance ranking
        fig_performance = px.bar(
            df_perf.head(15),
            x="performance_score",
            y="company_name",
            title="Overall Financial Performance Ranking",
            orientation='h',
            color="performance_score",
            color_continuous_scale="viridis"
        )
        fig_performance.update_layout(height=600)
        st.plotly_chart(fig_performance, use_container_width=True)
        
        # Performance components radar chart
        if len(df_perf) > 0:
            top_companies = df_perf.head(5)
            
            fig_radar = go.Figure()
            
            metrics = ['profitability_score', 'liquidity_score', 'leverage_score', 'efficiency_score']
            available_metrics = [m for m in metrics if m in top_companies.columns]
            
            for _, company in top_companies.iterrows():
                fig_radar.add_trace(go.Scatterpolar(
                    r=[company[m] for m in available_metrics],
                    theta=[m.replace('_score', '').title() for m in available_metrics],
                    fill='toself',
                    name=company['company_name']
                ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )),
                title="Top 5 Companies - Performance Breakdown",
                height=500
            )
            st.plotly_chart(fig_radar, use_container_width=True)
    
    # Cash flow analysis
    cf_cols = ['operating_cash_flow', 'investing_cash_flow', 'financing_cash_flow']
    if all(col in df.columns for col in cf_cols):
        df_cf = df[(abs(df[cf_cols]).sum(axis=1) > 0)].copy()
        
        if not df_cf.empty:
            # Cash flow waterfall for top companies
            top_cf_companies = df_cf.nlargest(10, 'operating_cash_flow')
            
            if not top_cf_companies.empty:
                fig_cf = go.Figure()
                
                fig_cf.add_trace(go.Bar(
                    name='Operating CF',
                    x=top_cf_companies['company_name'],
                    y=top_cf_companies['operating_cash_flow'],
                    marker_color='green'
                ))
                
                fig_cf.add_trace(go.Bar(
                    name='Investing CF',
                    x=top_cf_companies['company_name'],
                    y=top_cf_companies['investing_cash_flow'],
                    marker_color='blue'
                ))
                
                fig_cf.add_trace(go.Bar(
                    name='Financing CF',
                    x=top_cf_companies['company_name'],
                    y=top_cf_companies['financing_cash_flow'],
                    marker_color='orange'
                ))
                
                fig_cf.update_layout(
                    title='Cash Flow Analysis by Activity',
                    barmode='relative',
                    xaxis_tickangle=-45,
                    height=500
                )
                st.plotly_chart(fig_cf, use_container_width=True)

def calculate_performance_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate comprehensive performance scores for companies."""
    df_score = df.copy()
    
    # Initialize scores
    df_score['profitability_score'] = 0.0
    df_score['liquidity_score'] = 0.0
    df_score['leverage_score'] = 0.0
    df_score['efficiency_score'] = 0.0
    df_score['performance_score'] = 0.0
    
    # Profitability Score (0-100)
    profitability_metrics = ['net_profit_margin', 'roa', 'roe']
    prof_weights = [0.4, 0.3, 0.3]
    
    for metric, weight in zip(profitability_metrics, prof_weights):
        if metric in df_score.columns:
            # Handle NaN values and normalize to 0-100 scale
            metric_values = pd.to_numeric(df_score[metric], errors='coerce').fillna(0)
            max_vals = {'net_profit_margin': 50, 'roa': 25, 'roe': 30}
            normalized = np.clip(metric_values / max_vals.get(metric, 20) * 100, 0, 100)
            df_score['profitability_score'] += normalized * weight
    
    # Liquidity Score (0-100)
    if 'current_ratio' in df_score.columns:
        current_ratio_values = pd.to_numeric(df_score['current_ratio'], errors='coerce').fillna(0)
        # Optimal current ratio is around 2.0
        optimal_current_ratio = 2.0
        df_score['liquidity_score'] = np.where(
            current_ratio_values <= optimal_current_ratio,
            (current_ratio_values / optimal_current_ratio) * 100,
            100 - np.minimum((current_ratio_values - optimal_current_ratio) * 20, 80)
        )
        df_score['liquidity_score'] = np.clip(df_score['liquidity_score'], 0, 100)
    
    # Leverage Score (0-100) - Lower debt is better
    if 'debt_to_equity' in df_score.columns:
        debt_values = pd.to_numeric(df_score['debt_to_equity'], errors='coerce').fillna(0)
        df_score['leverage_score'] = np.clip(100 - (debt_values * 25), 0, 100)
    
    # Efficiency Score (0-100)
    if 'asset_turnover' in df_score.columns:
        turnover_values = pd.to_numeric(df_score['asset_turnover'], errors='coerce').fillna(0)
        df_score['efficiency_score'] = np.clip(turnover_values * 50, 0, 100)
    
    # Overall Performance Score (weighted average)
    weights = {'profitability': 0.4, 'liquidity': 0.2, 'leverage': 0.2, 'efficiency': 0.2}
    
    df_score['performance_score'] = (
        df_score['profitability_score'] * weights['profitability'] +
        df_score['liquidity_score'] * weights['liquidity'] +
        df_score['leverage_score'] * weights['leverage'] +
        df_score['efficiency_score'] * weights['efficiency']
    )
    
    # Only include companies with meaningful data
    df_score = df_score[
        (df_score['revenue'] > 0) | (df_score['total_assets'] > 0)
    ].sort_values('performance_score', ascending=False)
    
    return df_score

def create_risk_dashboard(df: pd.DataFrame):
    """Create risk analysis dashboard - FIXED VERSION."""
    st.subheader("⚠️ Risk Analysis Dashboard")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # High risk companies identification
        risk_flags = identify_risk_flags(df)
        if risk_flags:
            st.markdown("**🚨 Risk Alerts:**")
            for flag in risk_flags:
                st.warning(f"• {flag}")
        else:
            st.success("✅ No major risk flags identified")
    
    with col2:
        # Risk distribution pie chart
        risk_categories = categorize_companies_by_risk(df)
        if risk_categories and sum(risk_categories.values()) > 0:
            fig_risk_pie = px.pie(
                values=list(risk_categories.values()),
                names=list(risk_categories.keys()),
                title="Companies by Risk Category",
                color_discrete_map={
                    'Low Risk': 'green',
                    'Moderate Risk': 'yellow',
                    'High Risk': 'orange',
                    'Critical Risk': 'red'
                }
            )
            st.plotly_chart(fig_risk_pie, use_container_width=True)

def identify_risk_flags(df: pd.DataFrame) -> List[str]:
    """Identify risk flags in the dataset - FIXED VERSION."""
    flags = []
    
    # Liquidity risks
    if 'current_ratio' in df.columns:
        low_liquidity = df[(pd.notnull(df['current_ratio'])) & 
                          (df['current_ratio'] > 0) & 
                          (df['current_ratio'] < 1.0)]
        if len(low_liquidity) > 0:
            flags.append(f"{len(low_liquidity)} companies with current ratio < 1.0 (liquidity risk)")
    
    # High leverage
    if 'debt_to_equity' in df.columns:
        high_leverage = df[(pd.notnull(df['debt_to_equity'])) & 
                          (df['debt_to_equity'] > 0) & 
                          (df['debt_to_equity'] > 3.0)]
        if len(high_leverage) > 0:
            flags.append(f"{len(high_leverage)} companies with debt-to-equity > 3.0 (leverage risk)")
    
    # Negative profitability
    if 'net_profit_margin' in df.columns:
        negative_margins = df[(pd.notnull(df['net_profit_margin'])) & 
                             (df['net_profit_margin'] < -10)]
        if len(negative_margins) > 0:
            flags.append(f"{len(negative_margins)} companies with profit margin < -10% (profitability risk)")
    
    # Going concern issues - FIXED to handle non-string values
    if 'going_concern_issues' in df.columns:
        try:
            # Convert to string and filter out null/empty values safely
            going_concern_series = df['going_concern_issues'].astype(str)
            going_concern = df[(going_concern_series != 'nan') & 
                              (going_concern_series != '') & 
                              (going_concern_series != '0') & 
                              (going_concern_series != 'None') &
                              (going_concern_series.str.len() > 0)]
            if len(going_concern) > 0:
                flags.append(f"{len(going_concern)} companies with going concern issues noted")
        except Exception as e:
            # Skip going concern analysis if there are issues
            pass
    
    return flags

def categorize_companies_by_risk(df: pd.DataFrame) -> Dict[str, int]:
    """Categorize companies by overall risk level - FIXED VERSION."""
    risk_categories = {'Low Risk': 0, 'Moderate Risk': 0, 'High Risk': 0, 'Critical Risk': 0}
    
def categorize_companies_by_risk(df: pd.DataFrame) -> Dict[str, int]:
    """Categorize companies by overall risk level - FIXED VERSION."""
    risk_categories = {'Low Risk': 0, 'Moderate Risk': 0, 'High Risk': 0, 'Critical Risk': 0}
    
    for _, company in df.iterrows():
        risk_score = 0
        
        # Liquidity risk - handle NaN values safely
        try:
            current_ratio = pd.to_numeric(company.get('current_ratio', 0), errors='coerce')
            if pd.notnull(current_ratio) and current_ratio > 0:
                if current_ratio < 0.5:
                    risk_score += 3
                elif current_ratio < 1.0:
                    risk_score += 2
                elif current_ratio < 1.5:
                    risk_score += 1
        except:
            pass
        
        # Leverage risk - handle NaN values safely
        try:
            debt_to_equity = pd.to_numeric(company.get('debt_to_equity', 0), errors='coerce')
            if pd.notnull(debt_to_equity) and debt_to_equity > 0:
                if debt_to_equity > 5.0:
                    risk_score += 3
                elif debt_to_equity > 3.0:
                    risk_score += 2
                elif debt_to_equity > 2.0:
                    risk_score += 1
        except:
            pass
        
        # Profitability risk - handle NaN values safely
        try:
            profit_margin = pd.to_numeric(company.get('net_profit_margin', 0), errors='coerce')
            if pd.notnull(profit_margin):
                if profit_margin < -20:
                    risk_score += 3
                elif profit_margin < -5:
                    risk_score += 2
                elif profit_margin < 0:
                    risk_score += 1
        except:
            pass
        
        # Categorize based on total risk score
        if risk_score >= 6:
            risk_categories['Critical Risk'] += 1
        elif risk_score >= 4:
            risk_categories['High Risk'] += 1
        elif risk_score >= 2:
            risk_categories['Moderate Risk'] += 1
        else:
            risk_categories['Low Risk'] += 1
    
    return risk_categories

def format_currency(amount: float, currency: str = "SAR") -> str:
    """Format currency amounts with proper formatting."""
    if pd.isna(amount) or amount == 0:
        return f"0 {currency}"
    
    try:
        amount = float(amount)
        if abs(amount) >= 1_000_000_000:
            return f"{amount/1_000_000_000:.1f}B {currency}"
        elif abs(amount) >= 1_000_000:
            return f"{amount/1_000_000:.1f}M {currency}"
        elif abs(amount) >= 1_000:
            return f"{amount/1_000:.1f}K {currency}"
        else:
            return f"{amount:,.0f} {currency}"
    except (ValueError, TypeError):
        return f"0 {currency}"

def safe_numeric_operation(series: pd.Series, operation: str = 'mean'):
    """Safely perform numeric operations on pandas series."""
    try:
        numeric_series = pd.to_numeric(series, errors='coerce')
        numeric_series = numeric_series.dropna()
        
        if len(numeric_series) == 0:
            return 0
        
        if operation == 'mean':
            return numeric_series.mean()
        elif operation == 'sum':
            return numeric_series.sum()
        elif operation == 'max':
            return numeric_series.max()
        elif operation == 'min':
            return numeric_series.min()
        else:
            return numeric_series.mean()
    except:
        return 0
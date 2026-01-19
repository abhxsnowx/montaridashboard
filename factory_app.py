import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# Page config
st.set_page_config(page_title="Montari Production Dashboard", layout="wide", initial_sidebar_state="expanded")

# IMPROVED CSS - Better contrast for dark mode
st.markdown("""
<style>
    .main-header {
        font-size: 3rem; 
        color: white; 
        font-weight: bold;
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .metric-card {
        background: #1e1e1e;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_production_models():
    model = joblib.load(r'C:\Users\Abhilash\Documents\python\motor\production_xgboost_model.pkl')
    scaler = joblib.load(r'C:\Users\Abhilash\Documents\python\motor\production_scaler.pkl')
    feature_cols = joblib.load(r'C:\Users\Abhilash\Documents\python\motor\feature_names.pkl')
    return model, scaler, feature_cols

model, scaler, feature_cols = load_production_models()

# Header
st.markdown('<div class="main-header">MONTARI PRODUCTION LINE - ADVANCED ANALYTICS DASHBOARD</div>', unsafe_allow_html=True)

# Top KPI Row
kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)
with kpi_col1:
    st.metric("Model Accuracy", "99.5%", "+2.1%")
with kpi_col2:
    st.metric("R-Squared Score", "0.995", "Excellent")
with kpi_col3:
    st.metric("MAE Error", "±3.9 units", "-1.2")
with kpi_col4:
    st.metric("Training Samples", "200", "Optimal")
with kpi_col5:
    st.metric("Last Updated", datetime.now().strftime("%d-%b-%Y"))

st.markdown("---")

# SIDEBAR
st.sidebar.header("PRODUCTION PARAMETERS")
st.sidebar.markdown("#### Daily Production Plan")

plan_cap = st.sidebar.number_input("Planned Capacity (units)", 10, 500, 250, 5)
man_power = st.sidebar.slider("Manpower Available", 0, 20, 14, 1)
shift = st.sidebar.radio("Shift Type", [0.5, 1.0], 
                         format_func=lambda x: "Half Shift (4 hours)" if x==0.5 else "Full Shift (8 hours)",
                         horizontal=True)

st.sidebar.markdown("#### Operational Constraints")
downtime = st.sidebar.slider("Expected Downtime (minutes)", 0, 480, 0, 15)
material_loss = st.sidebar.slider("Material Loss Time (minutes)", 0, 480, 0, 15)

st.sidebar.markdown("#### Historical Context")
prev_completion = st.sidebar.slider("Previous Day Completion %", 70.0, 100.0, 95.0, 0.5)
prev_downtime = st.sidebar.slider("Previous Day Downtime", 0, 480, 0, 15)

with st.sidebar.expander("Advanced Settings"):
    forecast_days = st.slider("Forecast Period (days)", 3, 14, 7)
    confidence_interval = st.checkbox("Show Confidence Intervals", value=True)
    show_raw_data = st.checkbox("Display Raw Predictions", value=False)

has_downtime = 1 if downtime > 0 else 0
material_issue = 1 if material_loss > 0 else 0

st.sidebar.markdown("---")
predict_button = st.sidebar.button("GENERATE FORECAST", type="primary", use_container_width=True)

# MAIN DASHBOARD
if predict_button:
    input_data = pd.DataFrame({
        'Plan_Capacity': [plan_cap],
        'Man_power': [man_power],
        'SHIFT': [shift],
        'Total_Down_time': [downtime],
        'Material_Loss': [material_loss],
        'Has_downtime': [has_downtime],
        'Material_Issue': [material_issue],
        'Prev_Day_Completion': [prev_completion],
        'Prev_Day_Downtime': [prev_downtime]
    })[feature_cols]
    
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    completion_pct = (prediction/plan_cap)*100
    variance = prediction - plan_cap
    
    # RESULTS
    st.subheader("PRODUCTION FORECAST RESULTS")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric("Predicted Output", f"{prediction:.0f} units", f"{variance:+.0f}")
    with col2:
        st.metric("Completion Rate", f"{completion_pct:.1f}%", f"{completion_pct-100:.1f}%")
    with col3:
        st.metric("Plan Target", f"{plan_cap} units")
    with col4:
        st.metric("Expected Loss", f"{abs(variance):.0f} units" if variance < 0 else "0 units")
    with col5:
        st.metric("Downtime Impact", f"{downtime} min")
    with col6:
        st.metric("Efficiency", f"{(man_power/14)*100:.0f}%")
    
    st.markdown("---")
    
    # TABS
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Scenario Analysis", "Trend Forecast", "Factor Analysis", "Quality Metrics"])
    
    with tab1:
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.markdown("### Plan vs Predicted vs Historical")
            comparison_df = pd.DataFrame({
                'Metric': ['Plan Target', 'AI Prediction', 'Yesterday Actual', 'Best Case', 'Worst Case'],
                'Units': [
                    plan_cap, 
                    prediction, 
                    (prev_completion/100)*plan_cap,
                    prediction + 3.9,
                    prediction - 3.9
                ]
            })
            
            # FIXED: High contrast colors with white text
            fig1 = go.Figure(data=[
                go.Bar(x=comparison_df['Metric'], 
                       y=comparison_df['Units'],
                       text=comparison_df['Units'].round(0),
                       textposition='outside',
                       textfont=dict(size=14, color='white'),  # White text
                       marker=dict(
                           color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'],
                           line=dict(color='white', width=2)
                       ))
            ])
            
            fig1.update_layout(
                height=450,
                plot_bgcolor='#1e1e1e',
                paper_bgcolor='#1e1e1e',
                font=dict(color='white', size=12),
                xaxis=dict(gridcolor='#444', color='white'),
                yaxis=dict(gridcolor='#444', color='white', title='Production Units'),
                showlegend=False
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with chart_col2:
            st.markdown("### Completion Rate Indicator")
            fig2 = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=completion_pct,
                delta={'reference': 100, 'increasing': {'color': "#4ECDC4"}, 'decreasing': {'color': '#FF6B6B'}},
                title={'text': "Completion Percentage", 'font': {'size': 20, 'color': 'white'}},
                number={'font': {'size': 40, 'color': 'white'}},  # Large white numbers
                gauge={
                    'axis': {'range': [None, 120], 'tickwidth': 2, 'tickcolor': "white", 'tickfont': {'color': 'white', 'size': 12}},
                    'bar': {'color': "#4ECDC4", 'thickness': 0.75},
                    'bgcolor': "#1e1e1e",
                    'borderwidth': 3,
                    'bordercolor': "white",
                    'steps': [
                        {'range': [0, 70], 'color': '#FF6B6B'},
                        {'range': [70, 85], 'color': '#FFA07A'},
                        {'range': [85, 95], 'color': '#FFEAA7'},
                        {'range': [95, 120], 'color': '#4ECDC4'}
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.75,
                        'value': 100
                    }
                }
            ))
            fig2.update_layout(
                height=450,
                paper_bgcolor='#1e1e1e',
                font={'color': 'white', 'size': 14}
            )
            st.plotly_chart(fig2, use_container_width=True)
    
    with tab2:
        st.markdown("### Interactive Scenario Comparison")
        
        scenarios = {
            "Current Plan": [plan_cap, man_power, shift, downtime, material_loss, has_downtime, material_issue, prev_completion, prev_downtime],
            "Optimal Conditions": [plan_cap, 14, 1.0, 0, 0, 0, 0, 100, 0],
            "No Downtime": [plan_cap, man_power, shift, 0, 0, 0, 0, prev_completion, 0],
            "High Downtime (+90min)": [plan_cap, man_power, shift, downtime+90, material_loss, 1, material_issue, prev_completion, downtime],
            "Material Shortage": [plan_cap, man_power, shift, downtime, 120, has_downtime, 1, prev_completion, prev_downtime],
            "Low Manpower (-5)": [plan_cap, max(0, man_power-5), shift, downtime, material_loss, has_downtime, material_issue, prev_completion, prev_downtime],
            "High Manpower (+5)": [plan_cap, min(20, man_power+5), shift, downtime, material_loss, has_downtime, material_issue, prev_completion, prev_downtime],
            "Half Shift": [plan_cap, man_power, 0.5, downtime, material_loss, has_downtime, material_issue, prev_completion, prev_downtime],
            "Poor Yesterday": [plan_cap, man_power, shift, downtime, material_loss, has_downtime, material_issue, 80, prev_downtime]
        }
        
        scenario_results = []
        for name, params in scenarios.items():
            temp_df = pd.DataFrame([params], columns=feature_cols)
            pred = model.predict(scaler.transform(temp_df))[0]
            scenario_results.append({
                'Scenario': name, 
                'Predicted': pred,
                'Completion': (pred/plan_cap)*100
            })
        
        scenario_df = pd.DataFrame(scenario_results).sort_values('Predicted', ascending=False)
        
        # FIXED: Readable bar chart
        colors = ['#4ECDC4' if x >= 95 else '#FFEAA7' if x >= 85 else '#FF6B6B' 
                  for x in scenario_df['Completion']]
        
        fig3 = go.Figure(data=[
            go.Bar(x=scenario_df['Scenario'],
                   y=scenario_df['Predicted'],
                   text=scenario_df['Predicted'].round(0),
                   textposition='outside',
                   textfont=dict(size=14, color='white', family='Arial Black'),
                   marker=dict(color=colors, line=dict(color='white', width=2)))
        ])
        
        fig3.update_layout(
            height=500,
            plot_bgcolor='#1e1e1e',
            paper_bgcolor='#1e1e1e',
            font=dict(color='white', size=12),
            xaxis=dict(tickangle=-45, gridcolor='#444', color='white'),
            yaxis=dict(gridcolor='#444', color='white', title='Predicted Units'),
            showlegend=False
        )
        st.plotly_chart(fig3, use_container_width=True)
        
        st.dataframe(
            scenario_df.style.format({
                'Predicted': '{:.0f}',
                'Completion': '{:.1f}%'
            }).background_gradient(subset=['Completion'], cmap='RdYlGn', vmin=80, vmax=110),
            use_container_width=True,
            height=350
        )
    
    with tab3:
        st.markdown("### Multi-Day Production Forecast")
        
        dates = [datetime.now() + timedelta(days=i) for i in range(1, forecast_days+1)]
        daily_predictions = []
        daily_upper = []
        daily_lower = []
        
        current_completion = prev_completion
        current_downtime = prev_downtime
        
        for i, date in enumerate(dates):
            temp_completion = current_completion + np.random.uniform(-3, 3)
            temp_completion = np.clip(temp_completion, 75, 100)
            temp_downtime = max(0, downtime + np.random.randint(-20, 30))
            
            temp_data = pd.DataFrame([[
                plan_cap, man_power, shift, temp_downtime, material_loss,
                1 if temp_downtime>0 else 0, material_issue, 
                temp_completion, current_downtime
            ]], columns=feature_cols)
            
            pred = model.predict(scaler.transform(temp_data))[0]
            daily_predictions.append(pred)
            daily_upper.append(pred + 3.9)
            daily_lower.append(pred - 3.9)
            
            current_completion = temp_completion
            current_downtime = temp_downtime
        
        trend_df = pd.DataFrame({
            'Date': [d.strftime('%d-%b') for d in dates],
            'Predicted': daily_predictions,
            'Upper Bound': daily_upper,
            'Lower Bound': daily_lower,
            'Plan': [plan_cap] * len(dates)
        })
        
        # FIXED: High contrast line chart
        fig5 = go.Figure()
        
        if confidence_interval:
            fig5.add_trace(go.Scatter(
                x=trend_df['Date'], 
                y=trend_df['Upper Bound'],
                fill=None,
                mode='lines',
                line=dict(width=0),
                showlegend=False
            ))
            fig5.add_trace(go.Scatter(
                x=trend_df['Date'],
                y=trend_df['Lower Bound'],
                fill='tonexty',
                mode='lines',
                line=dict(width=0),
                name='Confidence Interval',
                fillcolor='rgba(78,205,196,0.3)'
            ))
        
        fig5.add_trace(go.Scatter(
            x=trend_df['Date'], 
            y=trend_df['Predicted'],
            mode='lines+markers+text',
            name='AI Prediction',
            line=dict(color='#4ECDC4', width=4),
            marker=dict(size=12, symbol='circle', color='white', line=dict(color='#4ECDC4', width=3)),
            text=trend_df['Predicted'].round(0),
            textposition='top center',
            textfont=dict(size=12, color='white', family='Arial Black')
        ))
        
        fig5.add_trace(go.Scatter(
            x=trend_df['Date'], 
            y=trend_df['Plan'],
            mode='lines+markers',
            name='Plan Target',
            line=dict(color='#FF6B6B', width=3, dash='dash'),
            marker=dict(size=10, symbol='diamond', color='white', line=dict(color='#FF6B6B', width=2))
        ))
        
        fig5.update_layout(
            height=500,
            plot_bgcolor='#1e1e1e',
            paper_bgcolor='#1e1e1e',
            font=dict(color='white', size=13),
            xaxis=dict(gridcolor='#444', color='white', title='Date'),
            yaxis=dict(gridcolor='#444', color='white', title='Production Units'),
            legend=dict(font=dict(size=12, color='white'), bgcolor='rgba(30,30,30,0.8)'),
            hovermode='x unified'
        )
        st.plotly_chart(fig5, use_container_width=True)
    
    with tab4:
        col_factor1, col_factor2 = st.columns(2)
        
        with col_factor1:
            st.markdown("### Production Impact Factors")
            impact_factors = {
                'Manpower Utilization': (man_power/14) * 100,
                'Shift Efficiency': shift * 100,
                'Downtime Impact': max(0, 100 - (downtime/480)*100),
                'Material Availability': max(0, 100 - (material_loss/480)*100),
                'Historical Momentum': prev_completion,
                'Overall Readiness': (completion_pct/100) * 95
            }
            
            impact_df = pd.DataFrame(list(impact_factors.items()), columns=['Factor', 'Impact Score'])
            
            # FIXED: Readable horizontal bar
            fig4 = go.Figure(data=[
                go.Bar(y=impact_df['Factor'],
                       x=impact_df['Impact Score'],
                       orientation='h',
                       text=impact_df['Impact Score'].round(1),
                       texttemplate='%{text:.1f}%',
                       textposition='outside',
                       textfont=dict(size=14, color='white', family='Arial Black'),
                       marker=dict(
                           color=impact_df['Impact Score'],
                           colorscale='Viridis',
                           line=dict(color='white', width=2),
                           showscale=False
                       ))
            ])
            
            fig4.update_layout(
                height=450,
                plot_bgcolor='#1e1e1e',
                paper_bgcolor='#1e1e1e',
                font=dict(color='white', size=12),
                xaxis=dict(range=[0, 120], gridcolor='#444', color='white'),
                yaxis=dict(gridcolor='#444', color='white')
            )
            st.plotly_chart(fig4, use_container_width=True)
        
        with col_factor2:
            st.markdown("### Efficiency Breakdown")
            
            efficiency_data = pd.DataFrame({
                'Category': ['Labor', 'Process', 'Materials'],
                'Value': [
                    (man_power/14)*35,
                    max(0, 40 - (downtime/480)*40),
                    max(0, 25 - (material_loss/480)*25)
                ]
            })
            
            # FIXED: Readable pie chart
            fig_pie = go.Figure(data=[go.Pie(
                labels=efficiency_data['Category'],
                values=efficiency_data['Value'],
                textinfo='label+percent+value',
                textfont=dict(size=16, color='white', family='Arial Black'),
                marker=dict(
                    colors=['#4ECDC4', '#FF6B6B', '#FFEAA7'],
                    line=dict(color='white', width=3)
                ),
                pull=[0.1, 0, 0]
            )])
            
            fig_pie.update_layout(
                height=450,
                paper_bgcolor='#1e1e1e',
                font=dict(color='white', size=14),
                showlegend=True,
                legend=dict(font=dict(size=14, color='white'))
            )
            st.plotly_chart(fig_pie, use_container_width=True)
    
    with tab5:
        st.markdown("### Quality & Performance Metrics")
        
        # FIXED: Readable heatmap
        st.markdown("### Parameter Correlation Matrix")
        correlation_data = pd.DataFrame({
            'Plan Capacity': [1.0, 0.27, 0.15, -0.12, -0.10],
            'Manpower': [0.27, 1.0, 0.19, -0.08, -0.05],
            'Shift': [0.15, 0.19, 1.0, -0.06, -0.04],
            'Downtime': [-0.12, -0.08, -0.06, 1.0, 0.35],
            'Material Loss': [-0.10, -0.05, -0.04, 0.35, 1.0]
        }, index=['Plan Capacity', 'Manpower', 'Shift', 'Downtime', 'Material Loss'])
        
        fig_heat = go.Figure(data=go.Heatmap(
            z=correlation_data.values,
            x=correlation_data.columns,
            y=correlation_data.index,
            colorscale='RdBu_r',
            text=correlation_data.values,
            texttemplate='%{text:.2f}',
            textfont=dict(size=16, color='white', family='Arial Black'),
            colorbar=dict(tickfont=dict(color='white'))
        ))
        
        fig_heat.update_layout(
            height=400,
            paper_bgcolor='#1e1e1e',
            plot_bgcolor='#1e1e1e',
            font=dict(color='white', size=12),
            xaxis=dict(color='white'),
            yaxis=dict(color='white')
        )
        st.plotly_chart(fig_heat, use_container_width=True)
    
    # RECOMMENDATIONS
    st.markdown("---")
    st.subheader("ACTIONABLE INSIGHTS & RECOMMENDATIONS")
    
    rec_col1, rec_col2, rec_col3, rec_col4 = st.columns(4)
    
    with rec_col1:
        if completion_pct >= 100:
            st.success(f"""
            **Production Outlook: POSITIVE**
            - Expected: {prediction:.0f} units
            - Status: On Target
            - Confidence: 99.5%
            """)
        elif completion_pct >= 90:
            st.warning(f"""
            **Production Outlook: ACCEPTABLE**
            - Expected: {prediction:.0f} units
            - Status: Minor Shortfall
            - Gap: {abs(variance):.0f} units
            """)
        else:
            st.error(f"""
            **Production Outlook: CRITICAL**
            - Expected: {prediction:.0f} units
            - Status: Below Target
            - Gap: {abs(variance):.0f} units
            """)
    
    with rec_col2:
        if downtime > 60:
            st.warning(f"""
            **Downtime Alert: HIGH**
            - Duration: {downtime} minutes
            - Impact: -{(downtime/480)*plan_cap:.0f} units
            - Action: Preventive maintenance
            """)
        elif downtime > 0:
            st.info(f"""
            **Downtime Alert: MODERATE**
            - Duration: {downtime} minutes
            - Impact: Minimal
            - Action: Monitor closely
            """)
        else:
            st.success("""
            **Downtime Status: NONE**
            - Optimal conditions
            - Maximum efficiency
            """)
    
    with rec_col3:
        if man_power < 12:
            st.error(f"""
            **Manpower: INSUFFICIENT**
            - Current: {man_power} workers
            - Optimal: 14 workers
            - Add: {14-man_power} workers
            """)
        elif man_power < 14:
            st.warning(f"""
            **Manpower: BELOW OPTIMAL**
            - Current: {man_power} workers
            - Add: {14-man_power} workers
            """)
        else:
            st.success(f"""
            **Manpower: OPTIMAL**
            - Current: {man_power} workers
            - Utilization: {(man_power/14)*100:.0f}%
            """)
    
    with rec_col4:
        if material_loss > 60:
            st.error(f"""
            **Material: CRITICAL**
            - Loss Time: {material_loss} min
            - Action: Emergency procurement
            """)
        elif material_loss > 0:
            st.warning(f"""
            **Material: CONCERN**
            - Loss Time: {material_loss} min
            - Action: Check inventory
            """)
        else:
            st.success("""
            **Material: ADEQUATE**
            - No shortages expected
            """)

else:
    # LANDING PAGE
    st.info("Configure production parameters in the sidebar and click 'GENERATE FORECAST' to view comprehensive analytics")
    
    col_landing1, col_landing2 = st.columns(2)
    
    with col_landing1:
        st.markdown("### Model Feature Importance")
        importance_data = pd.DataFrame({
            'Feature': ['Previous Day Completion', 'Plan Capacity', 'Manpower', 'Total Downtime', 
                       'Material Loss', 'Shift Type', 'Has Downtime', 'Material Issue', 'Previous Downtime'],
            'Importance': [31, 27, 19, 14, 4, 2, 1.5, 1, 0.5]
        })
        
        # FIXED: Readable importance chart
        fig_imp = go.Figure(data=[
            go.Bar(y=importance_data['Feature'],
                   x=importance_data['Importance'],
                   orientation='h',
                   text=importance_data['Importance'],
                   texttemplate='%{text:.1f}%',
                   textposition='outside',
                   textfont=dict(size=14, color='white', family='Arial Black'),
                   marker=dict(
                       color=importance_data['Importance'],
                       colorscale='Blues',
                       line=dict(color='white', width=2),
                       showscale=False
                   ))
        ])
        
        fig_imp.update_layout(
            height=500,
            plot_bgcolor='#1e1e1e',
            paper_bgcolor='#1e1e1e',
            font=dict(color='white', size=12),
            xaxis=dict(title="Importance Score (%)", gridcolor='#444', color='white'),
            yaxis=dict(gridcolor='#444', color='white')
        )
        st.plotly_chart(fig_imp, use_container_width=True)
    
    with col_landing2:
        st.markdown("### Model Performance Statistics")
        
        performance_metrics = pd.DataFrame({
            'Metric': ['R-Squared', 'MAE', 'RMSE', 'Train Acc', 'Test Acc', 'CV Score'],
            'Actual': [99.5, 3.9, 7.05, 99.7, 99.5, 98.5],
            'Target': [95.0, 10.0, 15.0, 95.0, 95.0, 95.0]
        })
        
        # FIXED: Readable grouped bar
        fig_perf = go.Figure(data=[
            go.Bar(name='Actual', 
                   x=performance_metrics['Metric'], 
                   y=performance_metrics['Actual'],
                   text=performance_metrics['Actual'].round(1),
                   textposition='outside',
                   textfont=dict(size=13, color='white', family='Arial Black'),
                   marker=dict(color='#4ECDC4', line=dict(color='white', width=2))),
            go.Bar(name='Target',
                   x=performance_metrics['Metric'],
                   y=performance_metrics['Target'],
                   text=performance_metrics['Target'].round(1),
                   textposition='outside',
                   textfont=dict(size=13, color='white', family='Arial Black'),
                   marker=dict(color='#FF6B6B', line=dict(color='white', width=2)))
        ])
        
        fig_perf.update_layout(
            barmode='group',
            height=500,
            plot_bgcolor='#1e1e1e',
            paper_bgcolor='#1e1e1e',
            font=dict(color='white', size=12),
            xaxis=dict(gridcolor='#444', color='white'),
            yaxis=dict(title="Score", gridcolor='#444', color='white'),
            legend=dict(font=dict(size=12, color='white'))
        )
        st.plotly_chart(fig_perf, use_container_width=True)

# FOOTER
st.markdown("---")
footer_col1, footer_col2, footer_col3, footer_col4 = st.columns(4)
with footer_col1:
    st.caption("**Production Line:** Montari Assembly")
with footer_col2:
    st.caption("**AI Engine:** XGBoost Regression | R²=0.995")
with footer_col3:
    st.caption("**Accuracy:** MAE ±3.9 units")
with footer_col4:
    st.caption(f"**Timestamp:** {datetime.now().strftime('%d %B %Y, %H:%M IST')}")

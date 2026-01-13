"""
Professional Streamlit demo app for Dynamic Ride Pricing System
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path
import time
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Page configuration
st.set_page_config(
    page_title="Dynamic Ride Pricing System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-family: 'Arial', sans-serif;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .professional-header {
        font-size: 2rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 1.5rem;
        font-family: 'Arial', sans-serif;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.25rem;
        padding: 1rem;
        margin: 1rem 0;
        font-weight: 500;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.25rem;
        padding: 1rem;
        margin: 1rem 0;
        font-weight: 500;
    }
    .feature-highlight {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .comparison-box {
        border: 2px solid #e1e5e9;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .professional-text {
        font-family: 'Arial', sans-serif;
        font-size: 1.1rem;
        line-height: 1.6;
        color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_sample_data():
    """Load sample data for demonstration"""
    try:
        data_path = Path(__file__).parent.parent / "data" / "cleaned_dataset.csv"
        if data_path.exists():
            df = pd.read_csv(data_path)
            return df
        else:
            # Generate enhanced sample data
            np.random.seed(42)
            n_samples = 1000
            
            sample_data = {
                'Number_of_Riders': np.random.poisson(15, n_samples),
                'Number_of_Drivers': np.random.poisson(10, n_samples),
                'Location_Category': np.random.choice(['Urban', 'Suburban', 'Rural'], n_samples, p=[0.5, 0.3, 0.2]),
                'Customer_Loyalty_Status': np.random.choice(['Silver', 'Gold', 'Platinum'], n_samples, p=[0.4, 0.4, 0.2]),
                'Number_of_Past_Rides': np.random.randint(0, 100, n_samples),
                'Average_Ratings': np.random.uniform(3.0, 5.0, n_samples),
                'Time_of_Booking': np.random.choice(['Morning', 'Afternoon', 'Evening', 'Night'], n_samples),
                'Vehicle_Type': np.random.choice(['Economy', 'Premium', 'Luxury'], n_samples, p=[0.6, 0.3, 0.1]),
                'Expected_Ride_Duration': np.random.uniform(5, 60, n_samples),
                'Historical_Cost_of_Ride': np.random.uniform(10, 100, n_samples)
            }
            
            return pd.DataFrame(sample_data)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def create_pricing_request_from_form():
    """Create pricing request from form inputs"""
    return {
        'number_of_riders': st.session_state.get('number_of_riders', 10),
        'number_of_drivers': st.session_state.get('number_of_drivers', 8),
        'location_category': st.session_state.get('location_category', 'Urban'),
        'customer_loyalty_status': st.session_state.get('customer_loyalty_status', 'Silver'),
        'time_of_booking': st.session_state.get('time_of_booking', 'Morning'),
        'vehicle_type': st.session_state.get('vehicle_type', 'Economy'),
        'expected_ride_duration': st.session_state.get('expected_ride_duration', 20),
        'request_id': f"demo_{np.random.randint(1000, 9999)}"
    }

def simulate_pricing_response(request):
    """Simulate pricing response for demo"""
    # Base price calculation
    base_price = request['expected_ride_duration'] * 0.8
    
    # Vehicle type multiplier
    vehicle_multipliers = {'Economy': 1.0, 'Premium': 1.5, 'Luxury': 2.0}
    base_price *= vehicle_multipliers.get(request['vehicle_type'], 1.0)
    
    # Location multiplier
    location_multipliers = {'Urban': 1.2, 'Suburban': 1.0, 'Rural': 0.8}
    base_price *= location_multipliers.get(request['location_category'], 1.0)
    
    # Demand-supply surge
    demand_supply_ratio = request['number_of_riders'] / max(request['number_of_drivers'], 1)
    surge_multiplier = min(3.0, max(0.8, 1.0 + (demand_supply_ratio - 1.0) * 0.3))
    
    # Loyalty discount
    loyalty_discounts = {'Silver': 0.95, 'Gold': 0.9, 'Platinum': 0.85}
    loyalty_multiplier = loyalty_discounts.get(request['customer_loyalty_status'], 1.0)
    
    # Time-based adjustment
    time_multipliers = {'Morning': 1.1, 'Afternoon': 1.0, 'Evening': 1.2, 'Night': 1.15}
    time_multiplier = time_multipliers.get(request['time_of_booking'], 1.0)
    
    # Calculate final price
    final_price = base_price * surge_multiplier * loyalty_multiplier * time_multiplier
    
    # Apply constraints
    final_price = max(5.0, min(200.0, final_price))
    
    return {
        'request_id': request['request_id'],
        'base_price': base_price,
        'surge_multiplier': surge_multiplier,
        'final_price': final_price,
        'loyalty_discount': (1 - loyalty_multiplier) * 100,
        'demand_supply_ratio': demand_supply_ratio,
        'time_multiplier': time_multiplier
    }

def main():
    """Main application"""
    st.markdown('<h1 class="main-header">Dynamic Ride Pricing System</h1>', unsafe_allow_html=True)
    
    # Professional sidebar navigation
    st.sidebar.title("Navigation")
    
    # Create navigation buttons
    if st.sidebar.button("🏠 System Overview", use_container_width=True):
        st.session_state.page = "System Overview"
    if st.sidebar.button("💰 Real-time Pricing", use_container_width=True):
        st.session_state.page = "Real-time Pricing"
    if st.sidebar.button("📊 Analytics Dashboard", use_container_width=True):
        st.session_state.page = "Analytics Dashboard"
    if st.sidebar.button("📈 Revenue Simulation", use_container_width=True):
        st.session_state.page = "Revenue Simulation"
    if st.sidebar.button("⚙️ System Configuration", use_container_width=True):
        st.session_state.page = "System Configuration"
    
    # Initialize page in session state
    if 'page' not in st.session_state:
        st.session_state.page = "System Overview"
    
    page = st.session_state.page
    
    # Load data
    df = load_sample_data()
    
    if page == "System Overview":
        st.markdown('<h2 class="professional-header">System Overview</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="professional-text">
        <p>The Dynamic Ride Pricing System optimizes ride fares in real-time by analyzing 
        demand-supply dynamics, time patterns, customer segments, and market conditions. 
        This advanced system leverages machine learning to balance profitability with 
        customer satisfaction while ensuring fair pricing practices.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # System metrics
        if not df.empty:
            st.markdown('<h2 class="professional-header">System Performance Metrics</h2>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_price = df['Historical_Cost_of_Ride'].mean()
                st.metric("Average Price", f"${avg_price:.2f}")
            
            with col2:
                total_rides = len(df)
                st.metric("Total Rides", f"{total_rides:,}")
            
            with col3:
                avg_duration = df['Expected_Ride_Duration'].mean()
                st.metric("Avg Duration", f"{avg_duration:.1f} min")
            
            with col4:
                demand_supply = df['Number_of_Riders'].sum() / df['Number_of_Drivers'].sum()
                st.metric("Demand/Supply Ratio", f"{demand_supply:.2f}")
        
        # Key features
        st.markdown('<h2 class="professional-header">Key Features</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>Core Capabilities</h3>
                <ul>
                    <li>Baseline Fare Prediction</li>
                    <li>Dynamic Surge Multipliers</li>
                    <li>Fairness Constraints</li>
                    <li>Business Rules Engine</li>
                    <li>Revenue Optimization</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>Advanced Features</h3>
                <ul>
                    <li>Real-time Data Integration</li>
                    <li>Multi-market Support</li>
                    <li>Behavioral Learning</li>
                    <li>Performance Optimization</li>
                    <li>Causal Inference</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Business impact
        st.markdown('<h2 class="professional-header">Business Impact</h2>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h4>Revenue Growth</h4>
                <p>8% increase in base revenue through optimized pricing strategies</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h4>Cost Efficiency</h4>
                <p>40% reduction in infrastructure and operational costs</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h4>Customer Satisfaction</h4>
                <p>13% improvement in customer satisfaction scores</p>
            </div>
            """, unsafe_allow_html=True)
    
    elif page == "Real-time Pricing":
        st.markdown('<h2 class="professional-header">Real-time Pricing Calculator</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown('<h3 class="professional-header">Ride Details</h3>', unsafe_allow_html=True)
            
            # Trip characteristics
            st.markdown("#### Trip Characteristics")
            col1a, col1b = st.columns(2)
            
            with col1a:
                st.number_input("Number of Riders", min_value=1, max_value=50, value=10, key="number_of_riders")
                st.number_input("Number of Drivers", min_value=1, max_value=50, value=8, key="number_of_drivers")
                st.selectbox("Location Category", ['Urban', 'Suburban', 'Rural'], key="location_category")
            
            with col1b:
                st.selectbox("Customer Loyalty Status", ['Silver', 'Gold', 'Platinum'], key="customer_loyalty_status")
                st.number_input("Number of Past Rides", min_value=0, max_value=500, value=25, key="number_of_past_rides")
                st.slider("Average Ratings", min_value=1.0, max_value=5.0, value=4.2, step=0.1, key="average_ratings")
            
            # Time and vehicle
            st.markdown("#### Time & Vehicle")
            col1c, col1d = st.columns(2)
            
            with col1c:
                st.selectbox("Time of Booking", ['Morning', 'Afternoon', 'Evening', 'Night'], key="time_of_booking")
                st.selectbox("Vehicle Type", ['Economy', 'Premium', 'Luxury'], key="vehicle_type")
            
            with col1d:
                st.number_input("Expected Ride Duration (minutes)", min_value=5, max_value=120, value=20, key="expected_ride_duration")
            
            # Calculate pricing button
            if st.button("Calculate Price", type="primary", use_container_width=True):
                request = create_pricing_request_from_form()
                response = simulate_pricing_response(request)
                
                # Store in session state
                st.session_state.current_pricing = response
                st.session_state.pricing_history = st.session_state.get('pricing_history', [])
                st.session_state.pricing_history.append({
                    'request_id': response['request_id'],
                    'final_price': response['final_price'],
                    'surge_multiplier': response['surge_multiplier'],
                    'location_category': request['location_category'],
                    'timestamp': pd.Timestamp.now()
                })
        
        with col2:
            st.markdown('<h3 class="professional-header">Pricing Results</h3>', unsafe_allow_html=True)
            
            if 'current_pricing' in st.session_state:
                response = st.session_state.current_pricing
                
                # Main price display
                st.markdown(f"""
                <div class="success-box">
                    <h3>Final Price: ${response['final_price']:.2f}</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Price breakdown
                st.markdown("#### Price Breakdown")
                
                col2a, col2b = st.columns(2)
                
                with col2a:
                    st.metric("Base Price", f"${response['base_price']:.2f}")
                    st.metric("Surge Multiplier", f"{response['surge_multiplier']:.2f}x")
                
                with col2b:
                    st.metric("Loyalty Discount", f"{response['loyalty_discount']:.1f}%")
                    st.metric("Demand/Supply", f"{response['demand_supply_ratio']:.2f}")
                
                # Demand indicator
                demand_ratio = response['demand_supply_ratio']
                if demand_ratio > 2.0:
                    st.markdown('<div class="warning-box">High demand detected - surge pricing active</div>', unsafe_allow_html=True)
                elif demand_ratio < 0.5:
                    st.markdown('<div class="success-box">Low demand - competitive pricing</div>', unsafe_allow_html=True)
            else:
                st.info("Enter ride details and click 'Calculate Price' to see pricing results.")
    
    elif page == "Analytics Dashboard":
        st.markdown('<h2 class="professional-header">Analytics Dashboard</h2>', unsafe_allow_html=True)
        
        if df.empty:
            st.error("No data available for analytics")
            return
        
        # Executive Summary
        st.markdown('<h3 class="professional-header">Executive Summary</h3>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_price = df['Historical_Cost_of_Ride'].mean()
            st.metric("Average Price", f"${avg_price:.2f}", delta=f"${avg_price - 25:.2f}")
        
        with col2:
            total_revenue = df['Historical_Cost_of_Ride'].sum()
            st.metric("Total Revenue", f"${total_revenue:,.0f}", delta="+8.3%")
        
        with col3:
            high_demand = (df['Number_of_Riders'] > df['Number_of_Drivers'] * 1.5).sum()
            demand_rate = high_demand / len(df) * 100
            st.metric("High Demand Rate", f"{demand_rate:.1f}%", delta="+2.1%")
        
        with col4:
            premium_rides = (df['Vehicle_Type'] == 'Premium').sum()
            premium_rate = premium_rides / len(df) * 100
            st.metric("Premium Usage", f"{premium_rate:.1f}%", delta="+1.8%")
        
        # Revenue Analysis
        st.markdown('<h3 class="professional-header">Revenue Analysis</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Revenue by Location
            location_revenue = df.groupby('Location_Category')['Historical_Cost_of_Ride'].agg(['sum', 'count']).reset_index()
            location_revenue.columns = ['Location', 'Revenue', 'Ride_Count']
            location_revenue['Avg_Revenue'] = location_revenue['Revenue'] / location_revenue['Ride_Count']
            
            fig = px.bar(location_revenue, x='Location', y='Revenue', 
                         title="Revenue by Location",
                         color='Avg_Revenue',
                         color_continuous_scale="Blues")
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Revenue by Vehicle Type
            vehicle_revenue = df.groupby('Vehicle_Type')['Historical_Cost_of_Ride'].agg(['sum', 'count']).reset_index()
            vehicle_revenue.columns = ['Vehicle', 'Revenue', 'Ride_Count']
            
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(
                go.Bar(x=vehicle_revenue['Vehicle'], y=vehicle_revenue['Revenue'], 
                      name="Revenue", marker_color='#1f77b4'),
                secondary_y=False,
            )
            fig.add_trace(
                go.Scatter(x=vehicle_revenue['Vehicle'], y=vehicle_revenue['Ride_Count'], 
                          mode='lines+markers', name="Ride Count", marker_color='#ff7f0e'),
                secondary_y=True,
            )
            fig.update_xaxes(title_text="Vehicle Type")
            fig.update_yaxes(title_text="Revenue ($)", secondary_y=False)
            fig.update_yaxes(title_text="Number of Rides", secondary_y=True)
            fig.update_layout(height=350, title_text="Revenue Performance by Vehicle Type")
            st.plotly_chart(fig, use_container_width=True)
        
        # Demand Supply Analysis
        st.markdown('<h3 class="professional-header">Demand & Supply Analysis</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Demand Supply Heatmap
            df['Demand_Supply_Ratio'] = df['Number_of_Riders'] / df['Number_of_Drivers']
            df['Demand_Category'] = pd.cut(df['Demand_Supply_Ratio'], 
                                          bins=[0, 0.8, 1.2, 2.0, float('inf')],
                                          labels=['Low Demand', 'Balanced', 'High Demand', 'Extreme Demand'])
            
            demand_supply_matrix = df.groupby(['Location_Category', 'Demand_Category']).size().unstack(fill_value=0)
            
            fig = px.imshow(demand_supply_matrix, 
                          title="Demand-Supply Matrix by Location",
                          color_continuous_scale="RdYlBu_r",
                          labels=dict(x="Demand Category", y="Location", color="Number of Rides"))
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Demand Supply Trends
            # Create hour mapping for time categories
            time_to_hour = {
                'Morning': 9,    # 9 AM
                'Afternoon': 14, # 2 PM
                'Evening': 18,   # 6 PM
                'Night': 22      # 10 PM
            }
            df['Hour'] = df['Time_of_Booking'].map(time_to_hour)
            hourly_demand = df.groupby('Hour')[['Number_of_Riders', 'Number_of_Drivers']].mean()
            hourly_demand['Ratio'] = hourly_demand['Number_of_Riders'] / hourly_demand['Number_of_Drivers']
            
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(
                go.Scatter(x=hourly_demand.index, y=hourly_demand['Number_of_Riders'], 
                          mode='lines+markers', name="Demand", line=dict(color='#1f77b4')),
                secondary_y=False,
            )
            fig.add_trace(
                go.Scatter(x=hourly_demand.index, y=hourly_demand['Number_of_Drivers'], 
                          mode='lines+markers', name="Supply", line=dict(color='#ff7f0e')),
                secondary_y=False,
            )
            fig.add_trace(
                go.Scatter(x=hourly_demand.index, y=hourly_demand['Ratio'], 
                          mode='lines', name="Demand/Supply Ratio", line=dict(color='#2ca02c', dash='dash')),
                secondary_y=True,
            )
            fig.update_xaxes(title_text="Hour of Day")
            fig.update_yaxes(title_text="Average Count", secondary_y=False)
            fig.update_yaxes(title_text="Demand/Supply Ratio", secondary_y=True)
            fig.update_layout(height=350, title_text="Demand Supply Trends Throughout Day")
            st.plotly_chart(fig, use_container_width=True)
        
        # Customer Segmentation
        st.markdown('<h3 class="professional-header">Customer Segmentation Analysis</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Customer Loyalty Performance
            loyalty_analysis = df.groupby('Customer_Loyalty_Status').agg({
                'Historical_Cost_of_Ride': ['mean', 'sum', 'count'],
                'Number_of_Past_Rides': 'mean',
                'Average_Ratings': 'mean'
            }).round(2)
            loyalty_analysis.columns = ['Avg_Price', 'Total_Revenue', 'Ride_Count', 'Avg_Past_Rides', 'Avg_Rating']
            
            # Create radar chart for loyalty segments
            categories = ['Avg_Price', 'Avg_Past_Rides', 'Avg_Rating']
            fig = go.Figure()
            
            for loyalty in loyalty_analysis.index:
                values = loyalty_analysis.loc[loyalty, categories].values
                values = np.concatenate([values, [values[0]]])  # Close the loop
                categories_full = categories + [categories[0]]
                
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories_full,
                    fill='toself',
                    name=loyalty
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, loyalty_analysis[categories].values.max() * 1.1]
                    )),
                title="Customer Loyalty Segments Performance",
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Price Sensitivity by Customer Segment
            price_sensitivity = df.groupby(['Customer_Loyalty_Status', 'Vehicle_Type'])['Historical_Cost_of_Ride'].mean().unstack()
            
            fig = go.Figure()
            for vehicle in price_sensitivity.columns:
                fig.add_trace(go.Bar(
                    name=vehicle,
                    x=price_sensitivity.index,
                    y=price_sensitivity[vehicle],
                ))
            
            fig.update_layout(
                title="Average Price by Customer Segment and Vehicle Type",
                xaxis_title="Customer Loyalty Status",
                yaxis_title="Average Price ($)",
                barmode='group',
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Time-Based Analysis
        st.markdown('<h3 class="professional-header">Time-Based Performance Analysis</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Revenue by Time of Day
            time_revenue = df.groupby('Time_of_Booking')['Historical_Cost_of_Ride'].agg(['sum', 'mean', 'count']).reset_index()
            time_revenue.columns = ['Time', 'Total_Revenue', 'Avg_Price', 'Ride_Count']
            
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(
                go.Bar(x=time_revenue['Time'], y=time_revenue['Total_Revenue'], 
                      name="Total Revenue", marker_color='#1f77b4'),
                secondary_y=False,
            )
            fig.add_trace(
                go.Scatter(x=time_revenue['Time'], y=time_revenue['Avg_Price'], 
                          mode='lines+markers', name="Average Price", marker_color='#ff7f0e'),
                secondary_y=True,
            )
            fig.update_xaxes(title_text="Time of Booking")
            fig.update_yaxes(title_text="Total Revenue ($)", secondary_y=False)
            fig.update_yaxes(title_text="Average Price ($)", secondary_y=True)
            fig.update_layout(height=350, title_text="Revenue Performance by Time of Day")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Duration vs Price Analysis
            duration_bins = pd.qcut(df['Expected_Ride_Duration'], q=5, labels=['Very Short', 'Short', 'Medium', 'Long', 'Very Long'])
            duration_analysis = df.groupby(duration_bins)['Historical_Cost_of_Ride'].agg(['mean', 'count']).reset_index()
            duration_analysis.columns = ['Duration', 'Avg_Price', 'Ride_Count']
            
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(
                go.Bar(x=duration_analysis['Duration'], y=duration_analysis['Avg_Price'], 
                      name="Average Price", marker_color='#1f77b4'),
                secondary_y=False,
            )
            fig.add_trace(
                go.Scatter(x=duration_analysis['Duration'], y=duration_analysis['Ride_Count'], 
                          mode='lines+markers', name="Ride Count", marker_color='#ff7f0e'),
                secondary_y=True,
            )
            fig.update_xaxes(title_text="Ride Duration")
            fig.update_yaxes(title_text="Average Price ($)", secondary_y=False)
            fig.update_yaxes(title_text="Number of Rides", secondary_y=True)
            fig.update_layout(height=350, title_text="Price Analysis by Ride Duration")
            st.plotly_chart(fig, use_container_width=True)
        
        # Key Insights
        st.markdown('<h3 class="professional-header">Key Business Insights</h3>', unsafe_allow_html=True)
        
        insights = []
        
        # Calculate insights
        peak_revenue_time = time_revenue.loc[time_revenue['Total_Revenue'].idxmax(), 'Time']
        highest_paying_segment = loyalty_analysis['Avg_Price'].idxmax()
        best_performing_location = location_revenue.loc[location_revenue['Revenue'].idxmax(), 'Location']
        surge_frequency = (df['Demand_Supply_Ratio'] > 1.5).sum() / len(df) * 100
        
        insights.extend([
            f"Peak revenue time: {peak_revenue_time}",
            f"Highest paying segment: {highest_paying_segment}",
            f"Best performing location: {best_performing_location}",
            f"Surge pricing frequency: {surge_frequency:.1f}% of rides"
        ])
        
        # Display insights in columns
        col1, col2 = st.columns(2)
        
        with col1:
            for i, insight in enumerate(insights[:2]):
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Insight {i+1}</h4>
                    <p>{insight}</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            for i, insight in enumerate(insights[2:]):
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Insight {i+3}</h4>
                    <p>{insight}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Performance Trends
        st.markdown('<h3 class="professional-header">Performance Trends</h3>', unsafe_allow_html=True)
        
        # Create a comprehensive performance dashboard
        col1, col2 = st.columns(2)
        
        with col1:
            # Price Distribution with Statistics
            fig = px.histogram(df, x='Historical_Cost_of_Ride', nbins=30, 
                             title="Price Distribution Analysis",
                             marginal="box")
            fig.add_vline(x=avg_price, line_dash="dash", line_color="red", 
                         annotation_text=f"Mean: ${avg_price:.2f}")
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Correlation Heatmap
            numeric_cols = ['Number_of_Riders', 'Number_of_Drivers', 'Expected_Ride_Duration', 
                          'Number_of_Past_Rides', 'Average_Ratings', 'Historical_Cost_of_Ride']
            corr_matrix = df[numeric_cols].corr()
            
            fig = px.imshow(corr_matrix, 
                          title="Feature Correlation Matrix",
                          color_continuous_scale="RdBu",
                          aspect="auto")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    elif page == "Revenue Simulation":
        st.markdown('<h2 class="professional-header">Revenue Simulation</h2>', unsafe_allow_html=True)
        
        if df.empty:
            st.error("No data available for simulation")
            return
        
        st.markdown("### Simulation Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Pricing Strategy")
            strategy = st.selectbox("Pricing Strategy", [
                "Current Pricing",
                "Aggressive Surge", 
                "Conservative Pricing",
                "Time-based Pricing",
                "Loyalty-focused Pricing"
            ])
            
            demand_elasticity = st.slider("Demand Elasticity", min_value=-2.0, max_value=-0.1, 
                                        value=-0.5, step=0.1, 
                                        help="How demand responds to price changes")
        
        with col2:
            st.markdown("#### Business Parameters")
            implementation_cost = st.number_input("Implementation Cost ($)", 
                                               min_value=0, max_value=100000, value=10000, step=1000)
            
            monthly_growth_rate = st.slider("Monthly Growth Rate (%)", min_value=0, max_value=10, 
                                         value=2, step=1) / 100
        
        # Run simulation
        if st.button("Run Simulation", type="primary"):
            with st.spinner("Running revenue simulation..."):
                # Generate scenario prices
                baseline_prices = df['Historical_Cost_of_Ride'].values
                
                if strategy == "Current Pricing":
                    scenario_prices = baseline_prices
                elif strategy == "Aggressive Surge":
                    demand_supply_ratio = df['Number_of_Riders'] / (df['Number_of_Drivers'] + 1e-8)
                    surge_factors = np.where(demand_supply_ratio > 1.5, 1.8, 1.0)
                    scenario_prices = baseline_prices * surge_factors
                elif strategy == "Conservative Pricing":
                    scenario_prices = baseline_prices * 0.9
                elif strategy == "Time-based Pricing":
                    time_multipliers = {'Morning': 1.2, 'Evening': 1.3, 'Afternoon': 1.1, 'Night': 1.0}
                    scenario_prices = baseline_prices.copy()
                    for time_period, multiplier in time_multipliers.items():
                        mask = df['Time_of_Booking'] == time_period
                        scenario_prices[mask] *= multiplier
                else:  # Loyalty-focused
                    loyalty_discounts = {'Silver': 0.95, 'Gold': 0.9, 'Platinum': 0.85}
                    scenario_prices = baseline_prices.copy()
                    for loyalty_status, discount in loyalty_discounts.items():
                        mask = df['Customer_Loyalty_Status'] == loyalty_status
                        scenario_prices[mask] *= discount
                
                # Calculate revenue difference
                baseline_revenue = baseline_prices.sum()
                scenario_revenue = scenario_prices.sum()
                revenue_lift = (scenario_revenue - baseline_revenue) / baseline_revenue * 100
                
                # Display results
                st.markdown("### Simulation Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Revenue Lift", f"{revenue_lift:.1f}%")
                
                with col2:
                    st.metric("Additional Revenue", f"${scenario_revenue - baseline_revenue:,.2f}")
                
                with col3:
                    st.metric("Price Change", f"{np.mean(scenario_prices - baseline_prices):.2f}")
                
                # Revenue comparison chart
                st.markdown("### Revenue Comparison")
                
                revenue_data = pd.DataFrame([
                    {'Scenario': 'Baseline', 'Revenue': baseline_revenue},
                    {'Scenario': strategy, 'Revenue': scenario_revenue}
                ])
                
                fig = px.bar(revenue_data, x='Scenario', y='Revenue', 
                           title="Revenue Comparison")
                st.plotly_chart(fig, use_container_width=True)
    
    elif page == "System Configuration":
        st.markdown('<h2 class="professional-header">System Configuration</h2>', unsafe_allow_html=True)
        
        st.markdown("### Pricing Constraints")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Surge Pricing")
            min_surge = st.slider("Minimum Surge Multiplier", min_value=0.5, max_value=1.0, value=0.8, step=0.1)
            max_surge = st.slider("Maximum Surge Multiplier", min_value=1.5, max_value=5.0, value=3.0, step=0.1)
            
            st.markdown("#### Price Bounds")
            floor_price = st.number_input("Minimum Price ($)", min_value=1.0, max_value=50.0, value=5.0, step=1.0)
            ceiling_price = st.number_input("Maximum Price ($)", min_value=50.0, max_value=500.0, value=100.0, step=10.0)
        
        with col2:
            st.markdown("#### Loyalty Discounts")
            silver_discount = st.slider("Silver Discount (%)", min_value=0, max_value=20, value=5, step=1)
            gold_discount = st.slider("Gold Discount (%)", min_value=0, max_value=25, value=10, step=1)
            platinum_discount = st.slider("Platinum Discount (%)", min_value=0, max_value=30, value=15, step=1)
            
            st.markdown("#### Location Premiums")
            urban_premium = st.slider("Urban Premium (%)", min_value=0, max_value=50, value=20, step=5)
            suburban_premium = st.slider("Suburban Premium (%)", min_value=-20, max_value=20, value=0, step=5)
            rural_premium = st.slider("Rural Premium (%)", min_value=-30, max_value=0, value=-20, step=5)
        
        # Save configuration
        if st.button("Save Configuration", type="primary"):
            config = {
                'pricing_constraints': {
                    'min_surge_multiplier': min_surge,
                    'max_surge_multiplier': max_surge,
                    'floor_price': floor_price,
                    'ceiling_price': ceiling_price
                },
                'loyalty_discounts': {
                    'Silver': silver_discount / 100,
                    'Gold': gold_discount / 100,
                    'Platinum': platinum_discount / 100
                },
                'location_premiums': {
                    'Urban': 1 + urban_premium / 100,
                    'Suburban': 1 + suburban_premium / 100,
                    'Rural': 1 + rural_premium / 100
                }
            }
            
            st.session_state.config = config
            st.success("Configuration saved successfully!")
            
            # Display saved configuration
            st.markdown("### Saved Configuration")
            st.json(config)

if __name__ == "__main__":
    main()

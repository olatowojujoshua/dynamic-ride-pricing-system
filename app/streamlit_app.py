"""
Streamlit demo app for Dynamic Ride Pricing System
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.pricing.pricing_engine import PricingEngine, PricingRequest
from src.evaluation.metrics import PricingMetrics
from src.evaluation.revenue_sim import RevenueSimulator
from src.config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET_COLUMN

# Page configuration
st.set_page_config(
    page_title="Dynamic Ride Pricing System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.25rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.25rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_sample_data():
    """Load sample data for demonstration"""
    try:
        # Try to load the actual dataset
        data_path = Path(__file__).parent.parent / "data" / "raw" / "dynamic_pricing.csv"
        if data_path.exists():
            df = pd.read_csv(data_path)
            return df
        else:
            # Generate sample data if file doesn't exist
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

@st.cache_resource
def initialize_pricing_engine():
    """Initialize the pricing engine"""
    try:
        # For demo purposes, we'll create a mock pricing engine
        # In production, this would load trained models
        engine = PricingEngine()
        return engine
    except Exception as e:
        st.error(f"Error initializing pricing engine: {e}")
        return None

def create_pricing_request_from_form() -> PricingRequest:
    """Create pricing request from form inputs"""
    return PricingRequest(
        number_of_riders=st.session_state.get('number_of_riders', 10),
        number_of_drivers=st.session_state.get('number_of_drivers', 8),
        location_category=st.session_state.get('location_category', 'Urban'),
        customer_loyalty_status=st.session_state.get('customer_loyalty_status', 'Silver'),
        number_of_past_rides=st.session_state.get('number_of_past_rides', 25),
        average_ratings=st.session_state.get('average_ratings', 4.2),
        time_of_booking=st.session_state.get('time_of_booking', 'Morning'),
        vehicle_type=st.session_state.get('vehicle_type', 'Economy'),
        expected_ride_duration=st.session_state.get('expected_ride_duration', 20),
        request_id=f"demo_{np.random.randint(1000, 9999)}"
    )

def simulate_pricing_response(request: PricingRequest) -> dict:
    """Simulate pricing response for demo"""
    # Base price calculation (simplified)
    base_price = request.expected_ride_duration * 0.8  # $0.8 per minute base
    
    # Vehicle type multiplier
    vehicle_multipliers = {'Economy': 1.0, 'Premium': 1.5, 'Luxury': 2.0}
    base_price *= vehicle_multipliers.get(request.vehicle_type, 1.0)
    
    # Location multiplier
    location_multipliers = {'Urban': 1.2, 'Suburban': 1.0, 'Rural': 0.8}
    base_price *= location_multipliers.get(request.location_category, 1.0)
    
    # Demand-supply surge
    demand_supply_ratio = request.number_of_riders / max(request.number_of_drivers, 1)
    surge_multiplier = min(3.0, max(0.8, 1.0 + (demand_supply_ratio - 1.0) * 0.3))
    
    # Loyalty discount
    loyalty_discounts = {'Silver': 0.95, 'Gold': 0.9, 'Platinum': 0.85}
    loyalty_multiplier = loyalty_discounts.get(request.customer_loyalty_status, 1.0)
    
    # Time-based adjustment
    time_multipliers = {'Morning': 1.1, 'Afternoon': 1.0, 'Evening': 1.2, 'Night': 1.15}
    time_multiplier = time_multipliers.get(request.time_of_booking, 1.0)
    
    # Calculate final price
    final_price = base_price * surge_multiplier * loyalty_multiplier * time_multiplier
    
    # Apply constraints
    final_price = max(5.0, min(200.0, final_price))  # Price bounds
    
    return {
        'request_id': request.request_id,
        'base_price': base_price,
        'surge_multiplier': surge_multiplier,
        'final_price': final_price,
        'loyalty_discount': (1 - loyalty_multiplier) * 100,
        'demand_supply_ratio': demand_supply_ratio,
        'time_multiplier': time_multiplier,
        'adjustments_applied': [
            f"Vehicle type ({request.vehicle_type}): {vehicle_multipliers.get(request.vehicle_type, 1.0)}x",
            f"Location ({request.location_category}): {location_multipliers.get(request.location_category, 1.0)}x",
            f"Loyalty ({request.customer_loyalty_status}): {(1-loyalty_multiplier)*100:.1f}% discount",
            f"Time ({request.time_of_booking}): {time_multiplier}x"
        ]
    }

def main():
    """Main application"""
    st.markdown('<h1 style="font-size: 2rem; color: #1f77b4; text-align: center; margin-bottom: 2rem;">Dynamic Ride Pricing System</h1>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'pricing_history' not in st.session_state:
        st.session_state.pricing_history = []
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Home", "Real-time Pricing", "Analytics Dashboard", "Revenue Simulation", "System Configuration"]
    )
    
    # Load data
    df = load_sample_data()
    
    if page == "Home":
        st.header("Welcome to Dynamic Ride Pricing System")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### System Overview
            This dynamic pricing system optimizes ride fares in real-time by analyzing:
            - **Demand-Supply Dynamics**: Real-time rider-driver ratios
            - **Time Patterns**: Rush hour, weekend, and seasonal variations  
            - **Customer Segments**: Loyalty status and ride history
            - **Market Conditions**: Location-based pricing and competition
            """)
        
        with col2:
            st.markdown("""
            ### Key Features
            - **Baseline Fare Prediction**: ML-powered base pricing
            - **Dynamic Surge Multipliers**: Real-time demand response
            - **Fairness Constraints**: Ensure equitable pricing
            - **Business Rules**: Price floors, ceilings, and stability
            - **Revenue Optimization**: Balance profit and demand
            """)
        
        # System metrics
        if not df.empty:
            st.markdown("### System Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_price = df[TARGET_COLUMN].mean()
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
        
        # Recent pricing activity
        if st.session_state.pricing_history:
            st.markdown("### Recent Pricing Activity")
            
            recent_pricing = pd.DataFrame(st.session_state.pricing_history[-5:])
            st.dataframe(recent_pricing[['request_id', 'final_price', 'surge_multiplier', 'location_category']], 
                        use_container_width=True)
    
    elif page == "Real-time Pricing":
        st.header("Real-time Pricing Calculator")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Ride Details")
            
            # Trip characteristics
            st.subheader("Trip Characteristics")
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
            st.subheader("Time & Vehicle")
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
                st.session_state.pricing_history.append({
                    'request_id': response['request_id'],
                    'final_price': response['final_price'],
                    'surge_multiplier': response['surge_multiplier'],
                    'location_category': request.location_category,
                    'timestamp': pd.Timestamp.now()
                })
        
        with col2:
            st.markdown("### Pricing Results")
            
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
                
                # Adjustments applied
                st.markdown("#### Adjustments Applied")
                for adjustment in response['adjustments_applied']:
                    st.write(f"• {adjustment}")
                
                # Demand indicator
                demand_ratio = response['demand_supply_ratio']
                if demand_ratio > 2.0:
                    st.markdown('<div class="warning-box">High demand detected - surge pricing active</div>', unsafe_allow_html=True)
                elif demand_ratio < 0.5:
                    st.markdown('<div class="success-box">Low demand - competitive pricing</div>', unsafe_allow_html=True)
            else:
                st.info("Enter ride details and click 'Calculate Price' to see pricing results.")
    
    elif page == "Analytics Dashboard":
        st.header("Analytics Dashboard")
        
        if df.empty:
            st.error("No data available for analytics")
            return
        
        # Key metrics
        st.markdown("### Key Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_price = df[TARGET_COLUMN].mean()
            st.metric("Average Price", f"${avg_price:.2f}")
        
        with col2:
            price_std = df[TARGET_COLUMN].std()
            st.metric("Price Volatility", f"${price_std:.2f}")
        
        with col3:
            high_demand = (df['Number_of_Riders'] > df['Number_of_Drivers'] * 1.5).sum()
            st.metric("High Demand Rides", f"{high_demand:,}")
        
        with col4:
            premium_rides = (df['Vehicle_Type'] == 'Premium').sum()
            st.metric("Premium Rides", f"{premium_rides:,}")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Price Distribution")
            fig = px.histogram(df, x=TARGET_COLUMN, nbins=50, title="Price Distribution")
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Demand vs Supply")
            fig = px.scatter(df.sample(min(500, len(df))), x='Number_of_Riders', y='Number_of_Drivers', 
                            color=TARGET_COLUMN, title="Demand vs Supply")
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Location analysis
        st.markdown("### Location Analysis")
        
        location_stats = df.groupby('Location_Category')[TARGET_COLUMN].agg(['mean', 'count', 'std']).reset_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Average Price by Location")
            fig = px.bar(location_stats, x='Location_Category', y='mean', 
                        title="Average Price by Location")
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Ride Volume by Location")
            fig = px.pie(location_stats, values='count', names='Location_Category', 
                        title="Ride Volume by Location")
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Time analysis
        st.markdown("### Time Analysis")
        
        time_stats = df.groupby('Time_of_Booking')[TARGET_COLUMN].agg(['mean', 'count']).reset_index()
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Bar(x=time_stats['Time_of_Booking'], y=time_stats['mean'], name="Average Price"),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(x=time_stats['Time_of_Booking'], y=time_stats['count'], 
                      mode='lines+markers', name="Ride Count"),
            secondary_y=True,
        )
        fig.update_xaxes(title_text="Time of Booking")
        fig.update_yaxes(title_text="Average Price ($)", secondary_y=False)
        fig.update_yaxes(title_text="Ride Count", secondary_y=True)
        fig.update_layout(height=400, title_text="Price and Volume by Time of Day")
        
        st.plotly_chart(fig, use_container_width=True)
    
    elif page == "Revenue Simulation":
        st.header("Revenue Simulation")
        
        if df.empty:
            st.error("No data available for simulation")
            return
        
        st.markdown("### Simulation Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Pricing Strategy")
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
            st.subheader("Business Parameters")
            implementation_cost = st.number_input("Implementation Cost ($)", 
                                               min_value=0, max_value=100000, value=10000, step=1000)
            
            monthly_growth_rate = st.slider("Monthly Growth Rate (%)", min_value=0, max_value=10, 
                                         value=2, step=1) / 100
        
        # Run simulation
        if st.button("Run Simulation", type="primary"):
            with st.spinner("Running revenue simulation..."):
                # Generate scenario prices
                baseline_prices = df[TARGET_COLUMN].values
                
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
                
                # Run revenue simulation
                simulator = RevenueSimulator()
                results = simulator.simulate_baseline_vs_dynamic_pricing(
                    df, baseline_prices, scenario_prices, demand_elasticity
                )
                
                # Break-even analysis
                price_changes = scenario_prices - baseline_prices
                break_even = simulator.calculate_break_even_analysis(
                    df, implementation_cost, price_changes
                )
                
                # Display results
                st.markdown("### Simulation Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Revenue Lift", f"{results['revenue_lift_percentage']:.1f}%")
                    st.metric("Demand Change", f"{results['demand_change_percentage']:.1f}%")
                
                with col2:
                    st.metric("Additional Revenue", f"${results['revenue_difference']:,.2f}")
                    st.metric("Price Change", f"{results['price_change_percentage']:.1f}%")
                
                with col3:
                    st.metric("Break-even Period", f"{break_even['break_even_days']:.0f} days")
                    st.metric("ROI (Year 1)", f"{break_even['roi_first_year']:.1f}%")
                
                # Revenue comparison chart
                st.markdown("### Revenue Comparison")
                
                revenue_data = pd.DataFrame([
                    {'Scenario': 'Baseline', 'Revenue': results['baseline_revenue']},
                    {'Scenario': strategy, 'Revenue': results['dynamic_revenue']}
                ])
                
                fig = px.bar(revenue_data, x='Scenario', y='Revenue', 
                           title="Revenue Comparison")
                st.plotly_chart(fig, use_container_width=True)
                
                # Segment analysis
                if 'revenue_by_location' in results:
                    st.markdown("### Revenue by Location")
                    
                    location_data = []
                    for location, metrics in results['revenue_by_location'].items():
                        location_data.append({
                            'Location': location,
                            'Revenue Lift (%)': metrics['revenue_lift_percentage'],
                            'Ride Count': metrics['ride_count']
                        })
                    
                    location_df = pd.DataFrame(location_data)
                    fig = px.bar(location_df, x='Location', y='Revenue Lift (%)', 
                               title="Revenue Lift by Location")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Recommendations
                st.markdown("### Recommendations")
                
                if results['revenue_lift_percentage'] > 10:
                    st.success(f"{strategy} shows strong revenue potential ({results['revenue_lift_percentage']:.1f}% lift)")
                elif results['revenue_lift_percentage'] > 5:
                    st.info(f"{strategy} shows moderate revenue potential ({results['revenue_lift_percentage']:.1f}% lift)")
                else:
                    st.warning(f"{strategy} shows limited revenue impact ({results['revenue_lift_percentage']:.1f}% lift)")
                
                if break_even['break_even_days'] < 90:
                    st.success(f"Quick break-even expected in {break_even['break_even_days']:.0f} days")
                elif break_even['break_even_days'] < 365:
                    st.info(f"Break-even expected in {break_even['break_even_days']:.0f} days")
                else:
                    st.warning(f"Long break-even period of {break_even['break_even_days']:.0f} days")
    
    elif page == "System Configuration":
        st.header("System Configuration")
        
        st.markdown("### Pricing Constraints")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Surge Pricing")
            min_surge = st.slider("Minimum Surge Multiplier", min_value=0.5, max_value=1.0, value=0.8, step=0.1)
            max_surge = st.slider("Maximum Surge Multiplier", min_value=1.5, max_value=5.0, value=3.0, step=0.1)
            
            st.subheader("Price Bounds")
            floor_price = st.number_input("Minimum Price ($)", min_value=1.0, max_value=50.0, value=5.0, step=1.0)
            ceiling_price = st.number_input("Maximum Price ($)", min_value=50.0, max_value=500.0, value=100.0, step=10.0)
        
        with col2:
            st.subheader("Loyalty Discounts")
            silver_discount = st.slider("Silver Discount (%)", min_value=0, max_value=20, value=5, step=1)
            gold_discount = st.slider("Gold Discount (%)", min_value=0, max_value=25, value=10, step=1)
            platinum_discount = st.slider("Platinum Discount (%)", min_value=0, max_value=30, value=15, step=1)
            
            st.subheader("Location Premiums")
            urban_premium = st.slider("Urban Premium (%)", min_value=0, max_value=50, value=20, step=5)
            suburban_premium = st.slider("Suburban Premium (%)", min_value=-20, max_value=20, value=0, step=5)
            rural_premium = st.slider("Rural Premium (%)", min_value=-30, max_value=0, value=-20, step=5)
        
        # Model configuration
        st.markdown("### Model Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Baseline Model")
            baseline_model = st.selectbox("Baseline Model Type", 
                                        ["Random Forest", "Gradient Boosting", "XGBoost", "LightGBM"])
            baseline_estimators = st.slider("Number of Estimators", min_value=50, max_value=500, value=100, step=50)
            
        with col2:
            st.subheader("Surge Model")
            surge_model = st.selectbox("Surge Model Type", 
                                      ["Random Forest", "Gradient Boosting", "Quantile Regression"])
            surge_quantile = st.slider("Surge Quantile", min_value=0.5, max_value=0.95, value=0.9, step=0.05)
        
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
                },
                'model_config': {
                    'baseline_model': baseline_model,
                    'baseline_estimators': baseline_estimators,
                    'surge_model': surge_model,
                    'surge_quantile': surge_quantile
                }
            }
            
            st.session_state.config = config
            st.success("✅ Configuration saved successfully!")
            
            # Display saved configuration
            st.markdown("### Saved Configuration")
            st.json(config)

if __name__ == "__main__":
    main()

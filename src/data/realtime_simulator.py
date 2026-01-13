"""
Real-time data simulation and external data integration
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import random
import json
from dataclasses import dataclass
from abc import ABC, abstractmethod
import requests
from ..config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES
from ..utils.logger import logger

@dataclass
class WeatherData:
    """Weather information for pricing adjustments"""
    temperature: float
    humidity: float
    precipitation: float
    wind_speed: float
    condition: str  # sunny, rainy, snowy, foggy

@dataclass
class TrafficData:
    """Traffic information for pricing adjustments"""
    congestion_level: float  # 0-1 scale
    average_speed: float
    incidents_count: int
    road_condition: str  # clear, congested, blocked

@dataclass
class EventData:
    """Event information affecting demand"""
    event_type: str  # concert, sports, conference, holiday
    attendance: int
    proximity_km: float
    impact_score: float  # 0-1 scale

class ExternalDataSource(ABC):
    """Abstract base class for external data sources"""
    
    @abstractmethod
    def fetch_data(self, location: str, timestamp: datetime) -> Dict[str, Any]:
        """Fetch external data for given location and time"""
        pass

class WeatherAPI(ExternalDataSource):
    """Simulated weather API integration"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.weather_conditions = ['sunny', 'cloudy', 'rainy', 'snowy', 'foggy']
        
    def fetch_data(self, location: str, timestamp: datetime) -> WeatherData:
        """Fetch or simulate weather data"""
        # Simulate weather patterns based on time and location
        hour = timestamp.hour
        month = timestamp.month
        
        # Seasonal temperature patterns
        if month in [12, 1, 2]:  # Winter
            base_temp = np.random.normal(5, 10)
            precip_prob = 0.3
        elif month in [6, 7, 8]:  # Summer
            base_temp = np.random.normal(25, 8)
            precip_prob = 0.2
        else:  # Spring/Fall
            base_temp = np.random.normal(15, 9)
            precip_prob = 0.25
        
        # Time-based variations
        if 6 <= hour <= 18:  # Daytime
            temp = base_temp + np.random.normal(5, 3)
        else:  # Nighttime
            temp = base_temp - np.random.normal(3, 2)
        
        # Weather condition
        if np.random.random() < precip_prob:
            condition = np.random.choice(['rainy', 'snowy'] if month in [12, 1, 2] else ['rainy'])
            precipitation = np.random.exponential(2)
        else:
            condition = np.random.choice(['sunny', 'cloudy', 'foggy'])
            precipitation = 0.0
        
        return WeatherData(
            temperature=round(temp, 1),
            humidity=round(np.random.beta(2, 2) * 100, 1),
            precipitation=round(precipitation, 1),
            wind_speed=round(np.random.exponential(5), 1),
            condition=condition
        )

class TrafficAPI(ExternalDataSource):
    """Simulated traffic API integration"""
    
    def __init__(self):
        self.rush_hours = [(7, 9), (17, 19)]  # Morning and evening rush
        
    def fetch_data(self, location: str, timestamp: datetime) -> TrafficData:
        """Fetch or simulate traffic data"""
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        
        # Base congestion by location type
        location_congestion = {
            'Urban': 0.7,
            'Suburban': 0.4,
            'Rural': 0.2
        }.get(location, 0.5)
        
        # Rush hour multiplier
        rush_multiplier = 1.0
        for start, end in self.rush_hours:
            if start <= hour <= end:
                rush_multiplier = 2.0 if day_of_week < 5 else 1.3  # Weekday vs weekend
                break
        
        # Weekend reduction
        weekend_multiplier = 0.7 if day_of_week >= 5 else 1.0
        
        # Calculate congestion
        base_congestion = location_congestion * rush_multiplier * weekend_multiplier
        congestion = min(1.0, base_congestion + np.random.normal(0, 0.1))
        
        # Average speed inversely related to congestion
        avg_speed = max(10, 60 * (1 - congestion) + np.random.normal(0, 5))
        
        # Incidents more likely during high congestion
        incidents = np.random.poisson(congestion * 2)
        
        # Road condition
        if congestion > 0.8:
            road_condition = 'blocked'
        elif congestion > 0.5:
            road_condition = 'congested'
        else:
            road_condition = 'clear'
        
        return TrafficData(
            congestion_level=round(congestion, 3),
            average_speed=round(avg_speed, 1),
            incidents_count=incidents,
            road_condition=road_condition
        )

class EventAPI(ExternalDataSource):
    """Simulated event API integration"""
    
    def __init__(self):
        self.event_types = ['concert', 'sports', 'conference', 'festival', 'holiday']
        self.major_venues = {
            'Urban': 3,    # More venues in urban areas
            'Suburban': 1,
            'Rural': 0.5
        }
        
    def fetch_data(self, location: str, timestamp: datetime) -> Optional[EventData]:
        """Fetch or simulate event data"""
        day_of_week = timestamp.weekday()
        hour = timestamp.hour
        
        # Events more likely on weekends and evenings
        event_probability = 0.1 if day_of_week < 5 else 0.3
        if 18 <= hour <= 23:
            event_probability *= 1.5
        
        # Venue availability by location
        venue_multiplier = self.major_venues.get(location, 1)
        event_probability *= venue_multiplier
        
        if np.random.random() > event_probability:
            return None
        
        # Generate event
        event_type = np.random.choice(self.event_types)
        
        # Attendance based on event type and location
        base_attendance = {
            'concert': 5000,
            'sports': 8000,
            'conference': 500,
            'festival': 10000,
            'holiday': 15000
        }.get(event_type, 1000)
        
        attendance = int(base_attendance * venue_multiplier * np.random.uniform(0.5, 2.0))
        
        # Proximity and impact
        proximity = np.random.exponential(2)  # Most events within 2km
        impact = min(1.0, attendance / 10000 * np.exp(-proximity / 5))
        
        return EventData(
            event_type=event_type,
            attendance=attendance,
            proximity_km=round(proximity, 2),
            impact_score=round(impact, 3)
        )

class RealTimeDataSimulator:
    """Real-time data simulation and integration"""
    
    def __init__(self):
        self.weather_api = WeatherAPI()
        self.traffic_api = TrafficAPI()
        self.event_api = EventAPI()
        self.base_data = None
        self.current_time = datetime.now()
        
    def load_base_data(self, df: pd.DataFrame):
        """Load base historical data for simulation"""
        self.base_data = df.copy()
        logger.info(f"Loaded base data with {len(df)} records")
        
    def simulate_realtime_scenario(self, 
                                 location: str = 'Urban',
                                 time_offset_minutes: int = 0,
                                 custom_events: Optional[List[EventData]] = None) -> Dict[str, Any]:
        """
        Simulate a real-time pricing scenario
        
        Args:
            location: Location category
            time_offset_minutes: Minutes from current time
            custom_events: Optional custom events to include
        
        Returns:
            Complete scenario data with external factors
        """
        timestamp = self.current_time + timedelta(minutes=time_offset_minutes)
        
        # Fetch external data
        weather = self.weather_api.fetch_data(location, timestamp)
        traffic = self.traffic_api.fetch_data(location, timestamp)
        event = self.event_api.fetch_data(location, timestamp)
        
        # Use custom event if provided
        if custom_events:
            event = custom_events[0] if custom_events else None
        
        # Generate base ride scenario
        scenario = self._generate_base_scenario(location, timestamp, weather, traffic, event)
        
        # Add external data
        scenario['external_factors'] = {
            'weather': weather,
            'traffic': traffic,
            'event': event,
            'timestamp': timestamp.isoformat()
        }
        
        # Calculate dynamic multipliers
        scenario['dynamic_multipliers'] = self._calculate_dynamic_multipliers(
            weather, traffic, event
        )
        
        return scenario
    
    def _generate_base_scenario(self, location: str, timestamp: datetime, 
                              weather: WeatherData, traffic: TrafficData, 
                              event: Optional[EventData]) -> Dict[str, Any]:
        """Generate base ride scenario with external influences"""
        
        # Base demand influenced by time and external factors
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        
        # Time-based demand patterns
        if 7 <= hour <= 9 or 17 <= hour <= 19:  # Rush hours
            base_demand = np.random.poisson(25)
        elif 10 <= hour <= 16:  # Daytime
            base_demand = np.random.poisson(15)
        else:  # Night/early morning
            base_demand = np.random.poisson(8)
        
        # Weather impact on demand
        weather_impact = 1.0
        if weather.condition in ['rainy', 'snowy']:
            weather_impact = 1.3  # More demand in bad weather
        elif weather.condition == 'sunny' and 10 <= hour <= 16:
            weather_impact = 1.1  # Slightly more demand in good weather
        
        # Traffic impact on supply
        traffic_impact = 1.0
        if traffic.road_condition == 'blocked':
            traffic_impact = 0.7  # Fewer drivers in heavy traffic
        elif traffic.road_condition == 'congested':
            traffic_impact = 0.85
        
        # Event impact
        event_demand_boost = 1.0
        if event:
            event_demand_boost = 1.0 + event.impact_score * 0.5
        
        # Calculate final numbers
        num_riders = int(base_demand * weather_impact * event_demand_boost)
        num_drivers = int(np.random.poisson(12) * traffic_impact)
        
        # Generate other features
        customer_loyalty = np.random.choice(['Silver', 'Gold', 'Platinum'], 
                                         p=[0.5, 0.35, 0.15])
        num_past_rides = np.random.randint(0, 100)
        avg_ratings = np.random.uniform(3.5, 5.0)
        time_booking = self._get_time_bucket(hour)
        vehicle_type = np.random.choice(['Economy', 'Premium', 'Luxury'], 
                                       p=[0.6, 0.3, 0.1])
        expected_duration = np.random.uniform(10, 60)
        
        return {
            'Number_of_Riders': max(1, num_riders),
            'Number_of_Drivers': max(1, num_drivers),
            'Location_Category': location,
            'Customer_Loyalty_Status': customer_loyalty,
            'Number_of_Past_Rides': num_past_rides,
            'Average_Ratings': round(avg_ratings, 2),
            'Time_of_Booking': time_booking,
            'Vehicle_Type': vehicle_type,
            'Expected_Ride_Duration': round(expected_duration, 1)
        }
    
    def _calculate_dynamic_multipliers(self, weather: WeatherData, 
                                     traffic: TrafficData, 
                                     event: Optional[EventData]) -> Dict[str, float]:
        """Calculate dynamic pricing multipliers based on external factors"""
        
        multipliers = {
            'weather_multiplier': 1.0,
            'traffic_multiplier': 1.0,
            'event_multiplier': 1.0,
            'combined_multiplier': 1.0
        }
        
        # Weather multiplier
        if weather.condition in ['rainy', 'snowy']:
            multipliers['weather_multiplier'] = 1.2
        elif weather.condition == 'foggy':
            multipliers['weather_multiplier'] = 1.1
        
        # Traffic multiplier
        if traffic.road_condition == 'blocked':
            multipliers['traffic_multiplier'] = 1.3
        elif traffic.road_condition == 'congested':
            multipliers['traffic_multiplier'] = 1.15
        
        # Event multiplier
        if event:
            multipliers['event_multiplier'] = 1.0 + event.impact_score * 0.3
        
        # Combined multiplier (capped to prevent extreme prices)
        combined = (multipliers['weather_multiplier'] * 
                   multipliers['traffic_multiplier'] * 
                   multipliers['event_multiplier'])
        multipliers['combined_multiplier'] = min(2.5, max(0.7, combined))
        
        return multipliers
    
    def _get_time_bucket(self, hour: int) -> str:
        """Convert hour to time bucket"""
        if 6 <= hour < 12:
            return 'Morning'
        elif 12 <= hour < 17:
            return 'Afternoon'
        elif 17 <= hour < 21:
            return 'Evening'
        else:
            return 'Night'
    
    def generate_time_series(self, start_time: datetime, duration_hours: int = 24,
                           interval_minutes: int = 30) -> pd.DataFrame:
        """
        Generate time series of scenarios for analysis
        
        Args:
            start_time: Starting timestamp
            duration_hours: Duration in hours
            interval_minutes: Interval between scenarios
        
        Returns:
            DataFrame with time series scenarios
        """
        scenarios = []
        current_time = start_time
        
        while current_time < start_time + timedelta(hours=duration_hours):
            # Cycle through locations
            location = np.random.choice(['Urban', 'Suburban', 'Rural'])
            
            scenario = self.simulate_realtime_scenario(
                location=location,
                time_offset_minutes=int((current_time - self.current_time).total_seconds() / 60)
            )
            
            # Flatten scenario for DataFrame
            flat_scenario = {
                'timestamp': current_time,
                'location': location,
                **scenario,
                **scenario['external_factors'],
                **scenario['dynamic_multipliers']
            }
            
            scenarios.append(flat_scenario)
            current_time += timedelta(minutes=interval_minutes)
        
        return pd.DataFrame(scenarios)
    
    def integrate_external_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Integrate external data into existing dataset
        
        Args:
            df: Original dataset
        
        Returns:
            Enhanced dataset with external factors
        """
        enhanced_df = df.copy()
        
        # Add external factor columns
        enhanced_df['weather_condition'] = np.random.choice(
            ['sunny', 'cloudy', 'rainy', 'snowy', 'foggy'], 
            size=len(df), 
            p=[0.4, 0.3, 0.15, 0.1, 0.05]
        )
        
        enhanced_df['traffic_level'] = np.random.beta(2, 2, size=len(df))
        enhanced_df['event_impact'] = np.random.exponential(0.1, size=len(df))
        enhanced_df['temperature'] = np.random.normal(20, 10, size=len(df))
        
        # Calculate combined external multiplier
        weather_mult = np.where(enhanced_df['weather_condition'].isin(['rainy', 'snowy']), 1.2, 1.0)
        traffic_mult = 1.0 + enhanced_df['traffic_level'] * 0.3
        event_mult = 1.0 + enhanced_df['event_impact'] * 0.2
        
        enhanced_df['external_multiplier'] = np.minimum(2.0, weather_mult * traffic_mult * event_mult)
        
        logger.info(f"Enhanced dataset with external factors: {enhanced_df.shape}")
        
        return enhanced_df

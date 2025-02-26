import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric
import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error
import holidays

# Set random seed for reproducibility
np.random.seed(42)

# Function to generate realistic retail sales data
def generate_retail_data(start_date='2020-01-01', end_date='2023-12-31', store_count=5):
    """
    Generate synthetic retail sales data with realistic patterns
    """
    # Create date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Initialize dataframe
    data = []
    
    # US holidays for holiday effect
    us_holidays = holidays.US()
    
    # Create store IDs
    store_ids = [f"store_{i}" for i in range(1, store_count + 1)]
    
    # Generate data for each store
    for store_id in store_ids:
        # Base sales for this store (varies by store)
        base_sales = np.random.randint(5000, 15000)
        
        for date in date_range:
            # Day of week effect (weekends have higher sales)
            dow_effect = 1.0
            if date.dayofweek >= 5:  # Weekend
                dow_effect = 1.3
            
            # Monthly seasonality (higher in certain months)
            month_effect = 1.0 + 0.2 * np.sin(2 * np.pi * date.month / 12)
            
            # Yearly trend (gradually increasing)
            yearly_trend = 1.0 + 0.1 * (date.year - 2020)
            
            # Holiday effect
            holiday_effect = 1.0
            if date in us_holidays:
                holiday_effect = 1.5
            # Black Friday effect
                if date.month == 11 and 20 <= date.day <= 30:
                    if date.dayofweek == 4:  # Friday
                        holiday_effect = 2.5
            
            # Promotion effect (random promotions)
            promotion = 0
            if np.random.random() < 0.05:  # 5% chance of promotion
                promotion = 1
                promotion_effect = 1.4
            else:
                promotion_effect = 1.0
            
            # Combine effects
            sales = base_sales * dow_effect * month_effect * yearly_trend * holiday_effect * promotion_effect
            
            # Add noise
            sales = int(sales * np.random.normal(1, 0.05))
            
            # Add to data
            data.append({
                'date': date,
                'store_id': store_id,
                'sales': sales,
                'promotion': promotion
            })
    
    # Convert to dataframe
    df = pd.DataFrame(data)
    return df

# Generate the dataset
retail_data = generate_retail_data()

# Save to CSV
retail_data.to_csv('retail_sales_data.csv', index=False)
print(f"Data generated with {len(retail_data)} records")

# Exploratory Data Analysis
def perform_eda(df):
    """
    Perform exploratory data analysis on the retail dataset
    """
    print("Dataset Overview:")
    print(f"Time range: {df['date'].min()} to {df['date'].max()}")
    print(f"Number of stores: {df['store_id'].nunique()}")
    print(f"Total records: {len(df)}")
    
    # Basic statistics
    print("\nSales Statistics by Store:")
    store_stats = df.groupby('store_id')['sales'].agg(['mean', 'std', 'min', 'max'])
    print(store_stats)
    
    # Visualize time series for each store
    plt.figure(figsize=(15, 10))
    
    # Plot time series for each store
    for store in df['store_id'].unique():
        store_data = df[df['store_id'] == store]
        plt.plot(store_data['date'], store_data['sales'], label=store)
    
    plt.title('Daily Sales by Store')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.savefig('sales_by_store.png')
    
    # Plot average sales by day of week
    plt.figure(figsize=(10, 6))
    df['day_of_week'] = df['date'].dt.day_name()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    sns.boxplot(x='day_of_week', y='sales', data=df, order=day_order)
    plt.title('Sales Distribution by Day of Week')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('sales_by_dow.png')
    
    # Plot average sales by month
    plt.figure(figsize=(10, 6))
    df['month'] = df['date'].dt.month_name()
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                   'July', 'August', 'September', 'October', 'November', 'December']
    
    sns.boxplot(x='month', y='sales', data=df, order=month_order)
    plt.title('Sales Distribution by Month')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('sales_by_month.png')
    
    # Promotion effect
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='promotion', y='sales', data=df)
    plt.title('Effect of Promotions on Sales')
    plt.xticks([0, 1], ['No Promotion', 'Promotion'])
    plt.tight_layout()
    plt.savefig('promotion_effect.png')
    
    return df

# Run EDA
print("Performing Exploratory Data Analysis...")
retail_data = perform_eda(retail_data)

# Time Series Forecasting with Prophet
def forecast_store_sales(df, store_id, periods=90, cv_periods=365):
    """
    Forecast sales for a specific store using Prophet
    
    Parameters:
    df (DataFrame): Input dataframe
    store_id (str): Store ID to forecast
    periods (int): Number of days to forecast
    cv_periods (int): Number of days to use for cross-validation
    
    Returns:
    tuple: (forecast DataFrame, model, metrics DataFrame)
    """
    # Filter data for this store
    store_data = df[df['store_id'] == store_id]
    
    # Prepare dataframe for Prophet (needs 'ds' and 'y' columns)
    prophet_df = store_data[['date', 'sales']].rename(columns={'date': 'ds', 'sales': 'y'})
    
    # Add promotion as a regressor
    prophet_df['promotion'] = store_data['promotion']
    
    # Initialize and fit the model
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='multiplicative',  # Multiplicative seasonality often works better for retail
        interval_width=0.95  # 95% confidence interval
    )
    
    # Add regressor
    model.add_regressor('promotion')
    
    # Add US holidays
    model.add_country_holidays(country_name='US')
    
    # Fit the model
    model.fit(prophet_df)
    
    # Create future dataframe for prediction
    future = model.make_future_dataframe(periods=periods, freq='D')
    
    # Add promotion to future (assuming no promotions in the future for simplicity)
    future['promotion'] = 0
    
    # Make forecast
    forecast = model.predict(future)
    
    # Perform cross-validation
    if len(prophet_df) > cv_periods:
        cv_results = cross_validation(
            model=model,
            initial=365*2,  # Use 2 years for initial training
            period=30,  # Test on 30 days each fold
            horizon=90,  # Forecast 90 days
            parallel='processes'
        )
        
        # Calculate performance metrics
        cv_metrics = performance_metrics(cv_results)
    else:
        cv_metrics = None
    
    return forecast, model, cv_metrics

# Run forecasting for all stores
def forecast_all_stores(df, output_dir='forecasts'):
    """Forecast sales for all stores and generate visualizations"""
    import os
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    all_metrics = []
    
    # Process each store
    for store_id in df['store_id'].unique():
        print(f"Forecasting for {store_id}...")
        
        # Generate forecast
        forecast, model, cv_metrics = forecast_store_sales(df, store_id)
        
        # Save forecast data
        forecast.to_csv(f"{output_dir}/{store_id}_forecast.csv", index=False)
        
        # Visualize forecast components
        fig1 = model.plot_components(forecast)
        fig1.savefig(f"{output_dir}/{store_id}_components.png")
        plt.close(fig1)
        
        # Visualize forecast
        fig2 = plt.figure(figsize=(15, 8))
        model.plot(forecast, xlabel='Date', ylabel='Sales')
        plt.title(f'Sales Forecast for {store_id}')
        plt.savefig(f"{output_dir}/{store_id}_forecast.png")
        plt.close(fig2)
        
        # If cv metrics available
        if cv_metrics is not None:
            # Save metrics
            cv_metrics['store_id'] = store_id
            all_metrics.append(cv_metrics)
            
            # Visualize cross-validation results
            fig3 = plt.figure(figsize=(10, 6))
            plot_cross_validation_metric(cv_results=cv_metrics, metric='mape')
            plt.title(f'Cross-Validation MAPE for {store_id}')
            plt.savefig(f"{output_dir}/{store_id}_cv_mape.png")
            plt.close(fig3)
    
    # Combine all metrics
    if all_metrics:
        all_metrics_df = pd.concat(all_metrics)
        all_metrics_df.to_csv(f"{output_dir}/all_stores_metrics.csv", index=False)
        
        # Compare performance across stores
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='store_id', y='mape', data=all_metrics_df)
        plt.title('Forecast MAPE by Store')
        plt.ylabel('Mean Absolute Percentage Error (%)')
        plt.savefig(f"{output_dir}/mape_comparison.png")
    
    return

# Main function to run the entire analysis
def main():
    # Load data (or use previously generated data)
    try:
        retail_data = pd.read_csv('retail_sales_data.csv')
        retail_data['date'] = pd.to_datetime(retail_data['date'])
        print("Loaded existing data.")
    except FileNotFoundError:
        retail_data = generate_retail_data()
        retail_data.to_csv('retail_sales_data.csv', index=False)
        print("Generated new data.")
    
    # Run EDA
    retail_data = perform_eda(retail_data)
    
    # Generate forecasts for all stores
    forecast_all_stores(retail_data)
    
    print("Analysis complete! Check the 'forecasts' directory for results.")
    
    # Generate a summary report
    generate_summary_report(retail_data)

def generate_summary_report(df):
    """Generate a summary report with key findings"""
    # Calculate key metrics
    total_sales = df['sales'].sum()
    avg_daily_sales = df.groupby('date')['sales'].sum().mean()
    best_performing_store = df.groupby('store_id')['sales'].sum().idxmax()
    best_store_sales = df.groupby('store_id')['sales'].sum().max()
    
    # Calculate promotion effectiveness
    promo_effect = df.groupby('promotion')['sales'].mean()
    promo_lift = (promo_effect[1] / promo_effect[0] - 1) * 100
    
    # Calculate day of week effect
    df['day_of_week'] = df['date'].dt.day_name()
    dow_effect = df.groupby('day_of_week')['sales'].mean()
    best_day = dow_effect.idxmax()
    worst_day = dow_effect.idxmin()
    
    # Write report
    with open('sales_analysis_report.md', 'w') as f:
        f.write(f"# Retail Sales Analysis Summary\n\n")
        f.write(f"## Key Metrics\n")
        f.write(f"- Total Sales: ${total_sales:,.2f}\n")
        f.write(f"- Average Daily Sales: ${avg_daily_sales:,.2f}\n")
        f.write(f"- Best Performing Store: {best_performing_store} (${best_store_sales:,.2f})\n\n")
        
        f.write(f"## Promotion Effectiveness\n")
        f.write(f"- Average Sales without Promotion: ${promo_effect[0]:,.2f}\n")
        f.write(f"- Average Sales with Promotion: ${promo_effect[1]:,.2f}\n")
        f.write(f"- Promotion Lift: {promo_lift:.2f}%\n\n")
        
        f.write(f"## Day of Week Pattern\n")
        f.write(f"- Best Day: {best_day} (${dow_effect[best_day]:,.2f})\n")
        f.write(f"- Worst Day: {worst_day} (${dow_effect[worst_day]:,.2f})\n\n")
        
        f.write(f"## Forecasting Results\n")
        f.write(f"Forecasts have been generated for all stores. See the 'forecasts' directory for detailed results.")

if __name__ == "__main__":
    main()

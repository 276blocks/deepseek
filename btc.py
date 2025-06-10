import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
from io import StringIO
import logging
from flask_caching import Cache
from dash.exceptions import PreventUpdate
from dash_extensions import Lottie
import statsmodels.api as sm
from scipy.stats import linregress
import time

# ----------------
# Constants & Themes
# ----------------
GENESIS_DATE = pd.Timestamp('2009-01-03')
DEFAULT_START = pd.Timestamp('2010-07-17')
DEFAULT_END = pd.Timestamp.today().normalize()
DATA_URL = (
    "http://data.bitcoinity.org/export_data.csv?c=e&currency=USD&data_type=price&"
    "r=day&t=l&timespan=all"
)
HALVING_DATES = [
    pd.Timestamp('2012-11-28'),
    pd.Timestamp('2016-07-09'),
    pd.Timestamp('2020-05-11'),
    pd.Timestamp('2024-04-19')
]

# Modern color schemes
COLOR_SCHEME = {
    'light': {
        'bg': '#f8f9fa', 'card': '#ffffff', 'text': '#212529', 
        'primary': '#0d6efd', 'secondary': '#6c757d', 'success': '#198754',
        'warning': '#ffc107', 'danger': '#dc3545', 'info': '#0dcaf0',
        'chart_bg': '#ffffff'
    },
    'dark': {
        'bg': '#121212', 'card': '#1e1e1e', 'text': '#e9ecef',
        'primary': '#0d6efd', 'secondary': '#6c757d', 'success': '#198754',
        'warning': '#ffc107', 'danger': '#dc3545', 'info': '#0dcaf0',
        'chart_bg': '#1e1e1e'
    }
}

# Modern glassmorphism effect
GLASS_STYLE = {
    'backdropFilter': 'blur(10px)',
    'backgroundColor': 'rgba(255, 255, 255, 0.1)',
    'borderRadius': '12px',
    'border': '1px solid rgba(255, 255, 255, 0.18)',
    'boxShadow': '0 8px 32px 0 rgba(31, 38, 135, 0.37)'
}

external_stylesheets = [
    dbc.themes.CYBORG,
    'https://cdnjs.cloudflare.com/ajax/libs/animate.css/4.1.1/animate.min.css',
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css'
]
LOTTIE_URL = 'https://assets3.lottiefiles.com/packages/lf20_jcikwtux.json'

# Initialize app with exception suppression
app = dash.Dash(
    __name__,
    external_stylesheets=external_stylesheets,
    suppress_callback_exceptions=True
)
server = app.server
cache = Cache(app.server, config={'CACHE_TYPE': 'simple'})

# ----------------
# Data Loading & Processing
# ----------------
def fetch_data(url):
    """Fetch data from a URL with retries (no caching on this helper)."""
    for attempt in range(3):
        try:
            response = requests.get(url)
            response.raise_for_status()
            if len(response.content) < 100:
                raise ValueError("Incomplete data received.")
            return response.text
        except (requests.exceptions.RequestException, ValueError) as e:
            logging.error(f"Attempt {attempt + 1} to fetch data failed: {e}")
            time.sleep(2)  # Wait before retrying
    raise ValueError("Failed to retrieve data after multiple attempts.")

def fetch_latest_price():
    """Fetch the latest Bitcoin price from CoinGecko API."""
    latest_price_url = 'https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd&include_24hr_change=true'
    try:
        response = requests.get(latest_price_url, timeout=10)
        response.raise_for_status()
        price_data = response.json()
        latest_price = price_data['bitcoin']['usd']
        change_24h = price_data['bitcoin'].get('usd_24h_change', 0)
        latest_time = pd.Timestamp.utcnow().replace(tzinfo=None)
        return latest_time, latest_price, change_24h
    except Exception as e:
        logging.error(f"Failed to fetch latest price: {e}")
        return None, None, None

@cache.memoize(timeout=600)
def load_data():
    """Load and clean Bitcoin CSV data and compute Days_Since_Genesis."""
    try:
        data_text = fetch_data(DATA_URL)
        data = pd.read_csv(StringIO(data_text))
        data.columns = data.columns.str.strip()
        
        if 'Time' not in data.columns or 'coinbase' not in data.columns:
            raise ValueError("CSV missing required columns: 'Time' or 'coinbase'")
        
        data['Time'] = pd.to_datetime(data['Time'], format='%Y-%m-%d %H:%M:%S UTC', errors='coerce')
        data['Price'] = data['coinbase']
        
        # Handle missing values
        if data['Price'].isna().any() and 'others' in data.columns:
            data.loc[data['Price'].isna(), 'Price'] = data.loc[data['Price'].isna(), 'others']
            
        data.dropna(subset=['Time', 'Price'], inplace=True)
        data.drop_duplicates(subset='Time', inplace=True)
        data.sort_values('Time', inplace=True)
        data.reset_index(drop=True, inplace=True)

        data['Days_Since_Genesis'] = (data['Time'] - GENESIS_DATE).dt.days
        data = data[data['Days_Since_Genesis'] > 0].copy()
        data['Date_Str'] = data['Time'].dt.strftime('%Y-%m-%d')
        
        # Fetch and append latest price
        latest_time, latest_price, _ = fetch_latest_price()
        if latest_time and latest_price:
            if data['Time'].max() < latest_time:
                latest_data = pd.DataFrame({
                    'Time': [latest_time],
                    'Price': [latest_price],
                    'Days_Since_Genesis': [(latest_time - GENESIS_DATE).days],
                    'Date_Str': [latest_time.strftime('%Y-%m-%d')]
                })
                data = pd.concat([data, latest_data], ignore_index=True)
                data = data.sort_values('Time').reset_index(drop=True)
                
        return data
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Calculate future values summary
def calculate_future_values(data, num_bitcoins, years_future):
    slope, intercept, r2 = perform_regression(data)
    now = pd.Timestamp.today()
    days_now = (now - GENESIS_DATE).days
    days_future = days_now + years_future*365
    current_price = data['Price'].iloc[-1]
    current_value = num_bitcoins * current_price
    future_price = 10**(intercept + slope*np.log10(days_future))
    # prior periods
    p1 = 10**(intercept + slope*np.log10(max(days_future-1,1)))
    pm = 10**(intercept + slope*np.log10(max(days_future-30,1)))
    py = 10**(intercept + slope*np.log10(max(days_future-365,1)))
    return {
        'current_value_str': f"${current_value:,.2f}",
        'future_value_str': f"${num_bitcoins*future_price:,.2f}",
        'per_year_increase_str': f"${num_bitcoins*(future_price - py):,.2f}",
        'per_month_increase_str': f"${num_bitcoins*(future_price - pm):,.2f}",
        'per_day_increase_str': f"${num_bitcoins*(future_price - p1):,.2f}",
        'slope': slope, 'intercept': intercept, 'r_squared': r2
    }

def perform_regression(data):
    """Perform Ordinary Least Squares regression on log-transformed data and return slope, intercept, and R-squared."""
    # Filter out zero or negative days to avoid log(0)
    data = data[data['Days_Since_Genesis'] > 0].copy()

    # Log-transform the data using log10
    log_days = np.log10(data['Days_Since_Genesis'])
    log_prices = np.log10(data['Price'])

    # Perform OLS regression
    X = sm.add_constant(log_days)
    y = log_prices
    model = sm.OLS(y, X)
    results = model.fit()

    intercept = results.params[0]
    slope = results.params[1]
    r_squared = results.rsquared

    return slope, intercept, r_squared

def predict_prices(slope, intercept, days_since_genesis):
    """Predict prices using the regression slope and intercept."""
    log_prices = intercept + slope * np.log10(days_since_genesis)
    prices = 10 ** log_prices
    return prices

# --------------------------
# GRAPH CREATION FUNCTIONS
# --------------------------

def create_log_log_plot(data, is_light_mode, start_date, end_date, axis_scale):
    """
    Create an interactive, modern plot with an OLS regression line, standard deviation bands,
    enhanced hover info showing deviation percentages, and a price equation annotation.

    Parameters:
      - data: DataFrame containing at least 'Days_Since_Genesis', 'Price', 'Time', and 'Date_Str'
      - is_light_mode: Boolean for light/dark mode color scheme.
      - start_date, end_date: Strings or timestamps defining the date range.
      - axis_scale: One of 'linear', 'log-linear', or 'log-log'

    Returns:
      - Plotly Figure object.
    """
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import statsmodels.api as sm

    # Filter out non-positive days to avoid issues with log transforms.
    data = data[data['Days_Since_Genesis'] > 0].copy()

    # -----------------------------
    # OLS Regression in log-log space
    # -----------------------------
    log_days = np.log10(data['Days_Since_Genesis'])
    log_prices = np.log10(data['Price'])
    X = sm.add_constant(log_days)
    y = log_prices

    # Perform OLS regression.
    results_ols = sm.OLS(y, X).fit()
    A = results_ols.params[1]  # Slope
    C = results_ols.params[0]  # Intercept
    r_squared = results_ols.rsquared

    # Compute predicted prices and deviation percentage for each data point.
    data['Predicted_Price'] = 10 ** (A * np.log10(data['Days_Since_Genesis']) + C)
    data['Deviation_Percent'] = ((data['Price'] - data['Predicted_Price']) / data['Predicted_Price']) * 100

    # -----------------------------------------
    # Prepare extended date range for predictions.
    # -----------------------------------------
    start_days = max((pd.to_datetime(start_date) - GENESIS_DATE).days, 1)
    end_days = (pd.to_datetime(end_date) - GENESIS_DATE).days
    extended_days_since_genesis = np.linspace(start_days, end_days, num=1000)
    extended_dates = GENESIS_DATE + pd.to_timedelta(extended_days_since_genesis, unit='D')
    extended_date_str = extended_dates.strftime('%Y-%m-%d')

    # Regression predictions over the extended range.
    predicted_log_prices = A * np.log10(extended_days_since_genesis) + C
    predicted_prices = 10 ** predicted_log_prices

    # Standard deviation bands based on OLS residuals.
    std_dev = np.std(results_ols.resid)
    predicted_prices_upper = 10 ** (predicted_log_prices + std_dev)   # +1 Std Dev
    predicted_prices_lower = 10 ** (predicted_log_prices - std_dev)   # -1 Std Dev
    predicted_prices_upper_2 = 10 ** (predicted_log_prices + 2 * std_dev)  # +2 Std Dev

    # -----------------------------
    # Determine x-axis settings.
    # -----------------------------
    if axis_scale in ['linear', 'log-linear']:
        x_data = data['Time']
        x_pred = extended_dates
        x_title = 'Date'
        xaxis_type = 'date'
    else:  # 'log-log'
        x_data = data['Days_Since_Genesis']
        x_pred = extended_days_since_genesis
        x_title = 'Days Since Genesis'
        xaxis_type = 'log'

    yaxis_type = 'linear' if axis_scale == 'linear' else 'log'

    # -----------------------------
    # Define Color Palette for Light and Dark Modes
    # -----------------------------
    if is_light_mode:
        colors = {
            'background': '#ffffff',
            'text': '#2c3e50',
            'plot_bgcolor': '#f8f9fa',
            'gridcolor': '#bdc3c7',
            # Trace colors:
            'actual_price': '#3498db',      # Modern blue
            'regression': '#9b59b6',        # Refined purple
            'std_fill': 'rgba(155, 89, 182, 0.2)',  # Translucent purple for std region
            'std_plus2': '#e67e22',         # Vibrant orange
            'halving': '#f1c40f',           # Bright yellow
        }
    else:
        # Awesome dark mode palette:
        colors = {
            'background': '#121212',   # Near-black background
            'text': '#e0e0e0',         # Soft white text
            'plot_bgcolor': '#1e1e1e', # Dark charcoal for plot background
            'gridcolor': '#424242',    # Medium gray grid lines
            # Trace colors:
            'actual_price': '#00e676',  # Neon green for actual prices
            'regression': '#2979ff',    # Vibrant blue for the regression line
            'std_fill': 'rgba(41, 121, 255, 0.2)',  # Translucent vibrant blue fill for ±1 Std Dev
            'std_plus2': '#ff9100',     # Bright vivid orange for +2 Std Dev
            'halving': '#ffea00',       # Bright yellow for halving events
        }

    # -----------------------------
    # Create Plotly Figure
    # -----------------------------
    fig = go.Figure()

    # 1. Actual Price Trace with Enhanced Hover Info.
    actual_customdata = np.stack((
        data['Date_Str'],
        data['Predicted_Price'],
        data['Deviation_Percent']
    ), axis=-1)
    fig.add_trace(go.Scatter(
        x=x_data,
        y=data['Price'],
        mode='lines+markers',
        name='Actual Price',
        line=dict(color=colors['actual_price'], width=2),
        marker=dict(size=1),
        customdata=actual_customdata,
        hovertemplate=(
            '<b>Date:</b> %{customdata[0]}<br>'
            '<b>Actual Price:</b> $%{y:,.2f}<br>'
            '<b>Deviation:</b> %{customdata[2]:.2f}%<extra></extra>'
        )
    ))

    # 2. OLS Regression Line Trace.
    fig.add_trace(go.Scatter(
        x=x_pred,
        y=predicted_prices,
        mode='lines',
        name='OLS Regression',
        line=dict(color=colors['regression'], width=3),
        customdata=np.stack((extended_date_str,), axis=-1),
        hovertemplate=(
            '<b>Predicted Price (OLS):</b> $%{y:,.2f}<extra></extra>'
        )
    ))

    # 3. +1 Standard Deviation Trace
    fig.add_trace(go.Scatter(
        x=x_pred,
        y=predicted_prices_upper,
        mode='lines',
        name='+1 Std Dev',
        line=dict(color=colors['regression'], width=2, dash='dash'),
        customdata=np.stack((extended_date_str,), axis=-1),
        hovertemplate=(
            '<b>+1 Std Dev Price:</b> $%{y:,.2f}<extra></extra>'
        )
    ))

    # 4. -1 Standard Deviation Trace
    fig.add_trace(go.Scatter(
        x=x_pred,
        y=predicted_prices_lower,
        mode='lines',
        name='-1 Std Dev',
        line=dict(color=colors['regression'], width=2, dash='dash'),
        customdata=np.stack((extended_date_str,), axis=-1),
        hovertemplate=(
            '<b>-1 Std Dev Price:</b> $%{y:,.2f}<extra></extra>'
        )
    ))

    # 5. +2 Standard Deviation Trace
    fig.add_trace(go.Scatter(
        x=x_pred,
        y=predicted_prices_upper_2,
        mode='lines',
        name='+2 Std Dev',
        line=dict(color=colors['std_plus2'], width=2, dash='dashdot'),
        customdata=np.stack((extended_date_str,), axis=-1),
        hovertemplate=(
            '<b>+2 Std Dev Price:</b> $%{y:,.2f}<extra></extra>'
        )
    ))

    # 6. Halving Points (Special Events)
    for idx, halving_date in enumerate(HALVING_DATES):
        if axis_scale in ['linear', 'log-linear']:
            halving_x = halving_date
            x_data_key = 'Time'
        else:
            halving_x = (halving_date - GENESIS_DATE).days
            x_data_key = 'Days_Since_Genesis'
        if halving_x in data[x_data_key].values:
            halving_price = data.loc[data[x_data_key] == halving_x, 'Price'].values[0]
            halving_date_str = halving_date.strftime('%Y-%m-%d')
            fig.add_trace(go.Scatter(
                x=[halving_x],
                y=[halving_price],
                mode='markers',
                name='Halving' if idx == 0 else None,
                marker=dict(color=colors['halving'], size=12, symbol='star'),
                customdata=[[halving_date_str]],
                hovertemplate=(
                    '<b>Halving Price:</b> $%{y:,.2f}<extra></extra>'
                ),
                showlegend=idx == 0
            ))

    # -----------------------------
    # Configure Axes and Layout
    # -----------------------------
    xaxis_config = dict(
        title=x_title,
        type=xaxis_type,
        showgrid=True,
        gridcolor=colors['gridcolor'],
        zeroline=False,
    )
    if axis_scale in ['linear', 'log-linear']:
        xaxis_config['tickformat'] = '%Y'
        xaxis_config['tickmode'] = 'auto'
        xaxis_config['nticks'] = 20
        # Add a range slider for interactivity.
        xaxis_config['rangeslider'] = dict(visible=True)
    elif axis_scale == 'log-log':
        xaxis_config['tickvals'] = [
            (pd.Timestamp(f'{year}-01-01') - GENESIS_DATE).days for year in range(2010, 2030)
        ]
        xaxis_config['ticktext'] = [str(year) for year in range(2010, 2030)]

    yaxis_config = dict(
        title='Price in USD',
        type=yaxis_type,
        showgrid=True,
        gridcolor=colors['gridcolor'],
        zeroline=False,
    )

    fig.update_layout(
        title='Bitcoin Price Analysis with OLS Regression',
        xaxis=xaxis_config,
        yaxis=yaxis_config,
        template='plotly_white' if is_light_mode else 'plotly_dark',
        hovermode='x unified',
        legend=dict(
            title="Legend",
            orientation="h",
            x=0.5,
            y=-0.2,
            xanchor='center',
            yanchor='top',
            bgcolor='rgba(0,0,0,0)',
            bordercolor='rgba(0,0,0,0)'
        ),
        margin=dict(l=60, r=60, t=120, b=100),
        plot_bgcolor=colors['plot_bgcolor'],
        paper_bgcolor=colors['background'],
        font=dict(color=colors['text'], size=12),
        transition=dict(duration=600, easing='cubic-in-out')
    )

    # -----------------------------
    equation_text = f"log10(Price) = {A:.2f} * log10(Days Since Genesis) + {C:.2f}\nR² = {r_squared:.3f}"
    fig.add_annotation(
        text=equation_text,
        xref='paper', yref='paper',
        x=0.50, y=1.02, showarrow=False,
        font=dict(color='red' if is_light_mode else 'orange', size=12),
        xanchor='center', yanchor='bottom',
        bgcolor='rgba(0, 0, 0, 0.5)' if not is_light_mode else 'rgba(255, 255, 255, 0.5)',
        bordercolor='red' if is_light_mode else 'orange'
    )

    return fig

def create_4yr_average_power_law_plot(data, is_light_mode=True):
    """
    Create a figure with actual Bitcoin prices, 4-year SMA, and a power law regression prediction.
    """
    data['Time'] = pd.to_datetime(data['Time'])
    data = data.sort_values('Time').reset_index(drop=True)
    genesis_date = pd.Timestamp('2009-01-03')
    data['Days_Since_Genesis'] = (data['Time'] - genesis_date).dt.days
    data = data[data['Days_Since_Genesis'] > 0].reset_index(drop=True)
    window_days = 4 * 365
    data['4yr_SMA'] = data['Price'].rolling(window=window_days, min_periods=1).mean()
    
    positive_mask = (data['4yr_SMA'] > 0) & (data['Days_Since_Genesis'] > 0)
    log_days = np.log10(data.loc[positive_mask, 'Days_Since_Genesis'])
    log_sma = np.log10(data.loc[positive_mask, '4yr_SMA'])
    slope, intercept, r_value, p_value, std_err = linregress(log_days, log_sma)
    A = slope
    C = intercept
    r_squared = r_value**2
    
    def price_scaling_law(x):
        return 10 ** (A * np.log10(x) + C)
    
    data['Predicted_Price'] = price_scaling_law(data['Days_Since_Genesis'])
    future_days = 5 * 365
    last_day = data['Days_Since_Genesis'].max()
    future_days_since_genesis = np.arange(last_day + 1, last_day + future_days + 1)
    future_predicted_prices = price_scaling_law(future_days_since_genesis)
    future_dates = [data['Time'].max() + pd.Timedelta(days=int(i)) for i in range(1, future_days + 1)]
    
    future_data = pd.DataFrame({
        'Time': future_dates,
        'Days_Since_Genesis': future_days_since_genesis,
        'Price': np.nan,
        '4yr_SMA': np.nan,
        'Predicted_Price': future_predicted_prices
    })
    
    combined_data = pd.concat([data, future_data], ignore_index=True)
    combined_data['Date_Str'] = combined_data['Time'].dt.strftime('%Y-%m-%d')
    
    def generate_yearly_ticks(genesis_date, end_date):
        years = pd.date_range(start=genesis_date, end=end_date, freq='YS')
        tickvals = (years - genesis_date).days
        tickvals = tickvals[tickvals > 0]
        ticktext = [year.strftime('%Y') for year in years if year >= genesis_date]
        return tickvals.tolist(), ticktext
    
    start_date_plot = data['Time'].min()
    end_date_plot = combined_data['Time'].max()
    tickvals, ticktext = generate_yearly_ticks(genesis_date, end_date_plot)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=combined_data['Days_Since_Genesis'],
        y=combined_data['Price'],
        mode='lines',
        name='Actual Price',
        line=dict(color='#1f77b4'),
        hovertemplate=('<b>Date:</b> %{customdata|%Y-%m-%d}<br>'
                       '<b>Price:</b> %{y:$,.2f}<extra></extra>'),
        customdata=combined_data['Time']
    ))
    fig.add_trace(go.Scatter(
        x=combined_data['Days_Since_Genesis'],
        y=combined_data['4yr_SMA'],
        mode='lines',
        name='4-Year SMA',
        line=dict(color='#ff7f0e', dash='dash'),
        hovertemplate=('<b>Date:</b> %{customdata|%Y-%m-%d}<br>'
                       '<b>4-Year SMA:</b> %{y:$,.2f}<extra></extra>'),
        customdata=combined_data['Time']
    ))
    fig.add_trace(go.Scatter(
        x=combined_data['Days_Since_Genesis'],
        y=combined_data['Predicted_Price'],
        mode='lines',
        name='Power Law Regression',
        line=dict(color='#2ca02c', dash='dot'),
        hovertemplate=('<b>Date:</b> %{customdata|%Y-%m-%d}<br>'
                       '<b>Predicted Price:</b> %{y:$,.2f}<extra></extra>'),
        customdata=combined_data['Time']
    ))
    
    fig.update_layout(
        title='Bitcoin Price with 4-Year SMA and Power Law Prediction',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        template='plotly_white' if is_light_mode else 'plotly_dark',
        legend=dict(x=0.01, y=0.99),
        hovermode='x unified',
        plot_bgcolor='#ffffff' if is_light_mode else '#1e1e1e',
        paper_bgcolor='#ffffff' if is_light_mode else '#1e1e1e',
        font=dict(color='black' if is_light_mode else 'white'),
        yaxis=dict(
            type='log',
            tickformat="$,.0f",
            title_font=dict(size=14),
            tickfont=dict(size=12),
            gridcolor='#e0e0e0' if is_light_mode else '#2a2a2a',
        ),
        xaxis=dict(
            type='log',
            tickmode='array',
            tickvals=tickvals,
            ticktext=ticktext,
            title_font=dict(size=14),
            tickfont=dict(size=12),
            showgrid=True,
            gridcolor='#e0e0e0' if is_light_mode else '#2a2a2a',
        )
    )
    
    equation_text = f"log10(Price) = {A:.2f} * log10(Days Since Genesis) + {C:.2f}\nR² = {r_squared:.3f}"
    fig.add_annotation(
        text=equation_text,
        xref='paper', yref='paper',
        x=0.50, y=1.02, showarrow=False,
        font=dict(color='red' if is_light_mode else 'orange', size=12),
        xanchor='center', yanchor='bottom',
        bgcolor='rgba(0, 0, 0, 0.5)' if not is_light_mode else 'rgba(255, 255, 255, 0.5)',
        bordercolor='red' if is_light_mode else 'orange'
    )
    
    return fig


def create_weinstein_stage_plot(
    data,
    is_light_mode=True,
    consolidation_bars=20,
    min_range_pct=0.02,
    volume_factor=1.75,
):
    """
    Create a plot showing Stan Weinstein's market stages using a 200-day moving average.
    Stages:
      1: Accumulation (basing)
      2: Advancing (markup)
      3: Topping (distribution)
      4: Declining (markdown)
    """
    df = data.copy()
    df['Time'] = pd.to_datetime(df['Time'])

    # Ensure correct DataFrame
    if isinstance(df, pd.Series):
        df = df.to_frame().T

    # 200-day SMA on daily data
    df['SMA_200'] = df['Price'].rolling(window=200, min_periods=1).mean()
    # 50-day SMA on daily data
    df['SMA_50'] = df['Price'].rolling(window=50, min_periods=1).mean()

    # SMA slope detection
    df['SMA_Slope'] = df['SMA_200'].pct_change()
    df['SMA_Rising'] = df['SMA_Slope'].rolling(3, min_periods=1).mean() >= 0.0005
    df['SMA_Falling'] = df['SMA_Slope'].rolling(3, min_periods=1).mean() <= -0.0005
    df['SMA_Flat'] = ~(df['SMA_Rising'] | df['SMA_Falling'])

    # Consolidation detection
    df['Consolidation_High'] = df['Price'].rolling(consolidation_bars, min_periods=1).max()
    df['Consolidation_Low'] = df['Price'].rolling(consolidation_bars, min_periods=1).min()
    df['Avg_Price'] = df['Price'].rolling(consolidation_bars, min_periods=1).mean()
    df['Range_Size'] = df['Consolidation_High'] - df['Consolidation_Low']
    df['Valid_Consolidation'] = df['Range_Size'] >= df['Avg_Price'] * min_range_pct

    # Volume analysis
    if 'Volume' in df.columns:
        df['Vol_Avg'] = df['Volume'].rolling(20, min_periods=1).mean()
        df['Volume_Spike'] = df['Volume'] > df['Vol_Avg'] * volume_factor
    else:
        df['Vol_Avg'] = 0
        df['Volume_Spike'] = False

    # Stage determination using SMA_200
    df['Stage'] = 0
    df.loc[
        (df['Price'] > df['SMA_200'])
        & df['SMA_Flat']
        & (df['Price'] < df['Consolidation_High'])
        & df['Valid_Consolidation'],
        'Stage',
    ] = 1
    df.loc[
        (df['Price'] > df['SMA_200']) & df['SMA_Rising'],
        'Stage',
    ] = 2
    df.loc[
        (df['Price'] < df['SMA_50'])
        & (df['SMA_Flat'] | df['SMA_Falling'])
        & (df['Price'] > df['Consolidation_Low'])
        & df['Valid_Consolidation'],
        'Stage',
    ] = 3
    df.loc[
        (df['Price'] < df['SMA_200']) & df['SMA_Falling'] & df['SMA_Flat'],
        'Stage',
    ] = 4

    # Stage transitions
    df['Stage1_to_Stage2'] = (df['Stage'] == 1) & (df['Stage'].shift(1) != 1)
    df['Stage2_to_Stage3'] = (df['Stage'] == 2) & (df['Stage'].shift(1) != 2)
    df['Stage3_to_Stage4'] = (df['Stage'] == 3) & (df['Stage'].shift(1) != 3)
    df['Stage4_to_Stage1'] = (df['Stage'] == 4) & (df['Stage'].shift(1) != 4)

    # Build figure
    fig = go.Figure()

    # Colored backgrounds per stage
    stage_colors = {
        1: 'rgba(41,98,255,0.15)',
        2: 'rgba(0,200,83,0.15)',
        3: 'rgba(255,109,0,0.15)',
        4: 'rgba(66,66,66,0.15)',
    }
    for s, c in stage_colors.items():
        seg = df[df['Stage'] == s]
        if not seg.empty:
            fig.add_trace(
                go.Scatter(
                    x=seg['Time'],
                    y=seg['Price'],
                    fill='tozeroy',
                    mode='none',
                    fillcolor=c,
                    name=f'Stage {s}',
                    showlegend=False,
                )
            )

    # Price
    fig.add_trace(
        go.Scatter(
            x=df['Time'], y=df['Price'], mode='lines', name='Price',
            hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>Price:</b> $%{y:,.2f}<extra></extra>'
        )
    )
    # 200-day SMA
    fig.add_trace(
        go.Scatter(
            x=df['Time'], y=df['SMA_200'], mode='lines', name='200-Day SMA',
            line=dict(dash='dash'),
            hovertemplate='<b>200-Day SMA:</b> $%{y:,.2f}<extra></extra>'
        )
    )
        # 50-day SMA
    fig.add_trace(
        go.Scatter(
            x=df['Time'], y=df['SMA_50'], mode='lines', name='50-Day SMA',
            line=dict(dash='dash'),
            hovertemplate='<b>50-Day SMA:</b> $%{y:,.2f}<extra></extra>'
        )
    )
   

    # Transition markers
    for cond, name, symbol in [
        ('Stage1_to_Stage2', 'Entry S2', 'triangle-up'),
        ('Stage2_to_Stage3', 'Warn S3', 'triangle-down'),
        ('Stage3_to_Stage4', 'Avoid S4', 'x'),
    ]:
        pts = df[df[cond]]
        if not pts.empty:
            fig.add_trace(
                go.Scatter(
                    x=pts['Time'], y=pts['Price'] * (0.97 if 'Entry' in name else 1.03),
                    mode='markers', name=name, marker=dict(symbol=symbol, size=10)
                )
            )

    # Current stage annotation
    current = int(df['Stage'].iloc[-1])
    labels = {1:'Accumulation',2:'Advancing',3:'Topping',4:'Declining'}
    fig.add_annotation(
        xref='paper', yref='paper', x=0.02, y=0.95,
        text=f"<b>Current Stage:</b> {labels.get(current,'N/A')}", showarrow=False,
        font=dict(color=('black' if is_light_mode else 'white')),
        bgcolor=stage_colors.get(current,'rgba(0,0,0,0.3)').replace('0.15','0.7')
    )

    # Layout
    fig.update_layout(
        title='Stan Weinstein Stage Analysis',
        xaxis_title='Date', yaxis_title='Price (USD)',
        template='plotly_white' if is_light_mode else 'plotly_dark',
        hovermode='x unified', yaxis=dict(type='linear')
    )

    return fig


def create_spiral_plot(data, is_light_mode):
    """
    Create an enhanced spiral plot for Bitcoin prices with a clockwise rotation.
    Note: This version uses the passed data (do not re-load the data here).
    """
    # Use the provided data (assumed already cleaned)
    # Perform regression on the log-transformed data
    slope, intercept, r_value = perform_regression(data)
    
    end_date_2030 = pd.Timestamp('2030-01-01')
    end_days_2030 = (end_date_2030 - GENESIS_DATE).days
    extended_dates = np.linspace(data['Days_Since_Genesis'].min(), end_days_2030, num=1000)
    predicted_prices_extended = predict_prices(slope, intercept, extended_dates)
    
    days_in_4_years = 4 * 365.25
    theta_actual = (data['Days_Since_Genesis'] % days_in_4_years) / days_in_4_years * 360
    r_actual = data['Price']
    
    theta_predicted = (extended_dates % days_in_4_years) / days_in_4_years * 360
    r_predicted = predicted_prices_extended
    predicted_dates = [GENESIS_DATE + pd.Timedelta(days=int(day)) for day in extended_dates]
    
    red_marked_dates = ['11/30/2013', '12/17/2017', '11/10/2021']
    green_marked_dates = ['01/14/2015', '12/15/2018', '11/21/2022']
    red_dates = pd.to_datetime(red_marked_dates, format='%m/%d/%Y')
    green_dates = pd.to_datetime(green_marked_dates, format='%m/%d/%Y')
    red_indices = data[data['Time'].isin(red_dates)].index
    green_indices = data[data['Time'].isin(green_dates)].index
    halving_indices = data[data['Time'].isin(HALVING_DATES)].index
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=r_actual,
        theta=theta_actual,
        mode='lines',
        name='Actual Price',
        line=dict(color='#2b8cbe', width=3),
        text=[f"📅 Date: {date.strftime('%Y-%m-%d')}<br>💰 Actual Price: ${price:,.2f}" 
              for date, price in zip(data['Time'], data['Price'])],
        hovertemplate='%{text}<extra></extra>'
    ))
    fig.add_trace(go.Scatterpolar(
        r=r_predicted,
        theta=theta_predicted,
        mode='lines',
        name='Predicted Price',
        line=dict(color='#fdae61', width=2, dash='dash'),
        text=[f"📅 Date: {date.strftime('%Y-%m-%d')}<br>🔮 Predicted Price: ${price:,.2f}" 
              for date, price in zip(predicted_dates, r_predicted)],
        hovertemplate='%{text}<extra></extra>'
    ))
    current_price = data.iloc[-1]['Price']
    current_days_since_genesis = data.iloc[-1]['Days_Since_Genesis']
    current_theta = (current_days_since_genesis % days_in_4_years) / days_in_4_years * 360
    current_date = (GENESIS_DATE + pd.to_timedelta(current_days_since_genesis, unit='D')).strftime('%Y-%m-%d')
    fig.add_trace(go.Scatterpolar(
        r=[0, current_price],
        theta=[0, current_theta],
        mode='lines+markers',
        line=dict(color='#d62728', width=3),
        marker=dict(color='#d62728', size=8),
        name='Current Price',
        hovertemplate=f"🎯 <b>Current Date:</b> {current_date}<br>💵 <b>Price:</b> ${current_price:,.2f}<extra></extra>"
    ))
    fig.add_trace(go.Scatterpolar(
        r=data.loc[green_indices, 'Price'],
        theta=theta_actual[green_indices],
        mode='markers',
        name='Bottoms',
        marker=dict(color='#2ca02c', size=12, symbol='triangle-up'),
        text=[f"📉 <b>Bottom</b><br>📅 Date: {data['Time'].iloc[idx].strftime('%Y-%m-%d')}<br>💵 Price: ${data['Price'].iloc[idx]:,.2f}" 
              for idx in green_indices],
        hovertemplate='%{text}<extra></extra>'
    ))
    fig.add_trace(go.Scatterpolar(
        r=data.loc[red_indices, 'Price'],
        theta=theta_actual[red_indices],
        mode='markers',
        name='Tops',
        marker=dict(color='#e41a1c', size=12, symbol='triangle-down'),
        text=[f"📈 <b>Top</b><br>📅 Date: {data['Time'].iloc[idx].strftime('%Y-%m-%d')}<br>💵 Price: ${data['Price'].iloc[idx]:,.2f}" 
              for idx in red_indices],
        hovertemplate='%{text}<extra></extra>'
    ))
    fig.add_trace(go.Scatterpolar(
        r=data.loc[halving_indices, 'Price'],
        theta=theta_actual[halving_indices],
        mode='markers',
        name='Halvings',
        marker=dict(color='#800026', size=14, symbol='star'),
        text=[f"⭐ <b>Halving</b><br>📅 Date: {data['Time'].iloc[idx].strftime('%Y-%m-%d')}<br>💵 Price: ${data['Price'].iloc[idx]:,.2f}" 
              for idx in halving_indices],
        hovertemplate='%{text}<extra></extra>'
    ))
    fig.update_layout(
        title={
            'text': '🚀 Bitcoin Price Spiral - Clockwise Rotation with Highlights',
            'y':0.97,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        polar=dict(
            radialaxis=dict(
                visible=True,
                type='log',
                showgrid=True,
                gridcolor='#e5e5e5',
                gridwidth=1,
                tickfont=dict(size=10)
            ),
            angularaxis=dict(
                direction='clockwise',
                rotation=180,
                tickfont=dict(size=10),
                showline=True,
                linewidth=1,
                linecolor='#636363'
            )
        ),
        template='plotly_white' if is_light_mode else 'plotly_dark',
        showlegend=True,
        legend=dict(
            title="Legend",
            orientation="h",
            x=0.5,
            y=-0.1,
            xanchor='center',
            yanchor='top',
            bgcolor='rgba(0,0,0,0)',
            bordercolor='rgba(0,0,0,0)'
        ),
        margin=dict(l=40, r=150, t=60, b=40),
        plot_bgcolor='#ffffff' if is_light_mode else '#1e1e1e',
        paper_bgcolor='#ffffff' if is_light_mode else '#1e1e1e',
        font=dict(color='black' if is_light_mode else 'white')
    )
    return fig

def create_oscillator_plot(data, is_light_mode):
    """Create an oscillator plot by normalizing data within decay bands."""
    a = 10**-17.351
    b = 5.836
    c = 10**1.836
    d = -0.0002323

    end_date_2040 = pd.Timestamp('2040-01-01')
    end_days_2040 = (end_date_2040 - GENESIS_DATE).days
    x_extended = np.linspace(data['Days_Since_Genesis'].min(), end_days_2040, num=1000)

    y_upper = a * data['Days_Since_Genesis']**b
    y_lower = a * data['Days_Since_Genesis']**b * (1 + c * 10**(d * data['Days_Since_Genesis']))

    normalized_price = 100 - (data['Price'] - y_lower) / (y_upper - y_lower) * 100

    oscillator_fig = go.Figure()
    oscillator_fig.add_trace(go.Scatter(
        x=data['Days_Since_Genesis'],
        y=normalized_price,
        mode='lines',
        name='Normalized BTC Price',
        line=dict(color='#FFA500', width=2),
        text=data['Time'].dt.strftime('%Y-%m-%d'),
        hovertemplate='<b>Date:</b> %{text}<br><b>Normalized Price:</b> %{y:.2f}%<extra></extra>'
    ))
    oscillator_fig.update_layout(
        title='BTC Normalized Oscillator Plot',
        xaxis=dict(
            title='Year',
            type='log',
            showgrid=True,
            gridcolor='#e0e0e0' if is_light_mode else '#2a2a2a'
        ),
        yaxis=dict(
            title='Normalized Price (%)',
            tickmode='array',
            tickvals=[i for i in range(0, 101, 10)],
            ticktext=[f'{i}%' for i in range(0, 101, 10)],
            showgrid=True,
            gridcolor='#e0e0e0' if is_light_mode else '#2a2a2a'
        ),
        template='plotly_white' if is_light_mode else 'plotly_dark',
        autosize=True,
        showlegend=True,
        margin=dict(l=40, r=40, t=100, b=60),
        plot_bgcolor='#ffffff' if is_light_mode else '#1e1e1e',
        paper_bgcolor='#ffffff' if is_light_mode else '#1e1e1e',
        font=dict(color='black' if is_light_mode else 'white')
    )
    return oscillator_fig

def create_deviation_analysis_plot(data, is_light_mode):
    """
    Create a plot showing the percentage deviations of actual prices from the predicted prices based on the power law regression.
    """
    slope, intercept, r_value = perform_regression(data)
    predicted_prices = predict_prices(slope, intercept, data['Days_Since_Genesis'])
    data['Deviation_Percentage'] = ((data['Price'] - predicted_prices) / predicted_prices) * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data['Time'],
        y=data['Deviation_Percentage'],
        mode='lines',
        name='Deviation (%)',
        line=dict(color='#1f77b4'),
        hovertemplate='<b>Date:</b> %{x}<br><b>Deviation:</b> %{y:.2f}%<extra></extra>'
    ))
    fig.add_hline(y=50, line_color='red', line_dash='dash', 
                  annotation_text='Overbought (>50%)', annotation_position="top left")
    fig.add_hline(y=-50, line_color='green', line_dash='dash', 
                  annotation_text='Oversold (<-50%)', annotation_position="bottom left")
    fig.update_layout(
        title='Percentage Deviation from Power Law Regression Line',
        xaxis_title='Date',
        yaxis_title='Deviation (%)',
        template='plotly_white' if is_light_mode else 'plotly_dark',
        showlegend=False,
        font=dict(color='black' if is_light_mode else 'white')
    )
    return fig

def create_power_law_residuals_plot(data, is_light_mode):
    """
    Create a plot showing the residuals between actual prices and predicted prices from the power law regression.
    """
    slope, intercept, r_value = perform_regression(data)
    predicted_prices = predict_prices(slope, intercept, data['Days_Since_Genesis'])
    residuals = np.log(data['Price']) - np.log(predicted_prices)
    std_dev = np.std(residuals)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data['Time'],
        y=residuals,
        mode='lines',
        name='Residuals',
        line=dict(color='#1f77b4'),
        hovertemplate='<b>Date:</b> %{x}<br><b>Residual:</b> %{y:.4f}<extra></extra>'
    ))
    fig.add_hline(y=0, line_color='black', line_dash='dash', 
                  annotation_text='Zero Line', annotation_position="bottom right")
    fig.add_hline(y=std_dev, line_color='orange', line_dash='dot', 
                  annotation_text='+1 Std Dev', annotation_position="top right")
    fig.add_hline(y=-std_dev, line_color='green', line_dash='dot', 
                  annotation_text='-1 Std Dev', annotation_position="bottom right")
    fig.add_hline(y=2 * std_dev, line_color='red', line_dash='dot', 
                  annotation_text='+2 Std Dev', annotation_position="top right")
    fig.update_layout(
        title='Power Law Residuals Over Time',
        xaxis_title='Date',
        yaxis_title='Log Price Residuals',
        template='plotly_white' if is_light_mode else 'plotly_dark',
        showlegend=False,
        plot_bgcolor='#ffffff' if is_light_mode else '#1e1e1e',
        paper_bgcolor='#ffffff' if is_light_mode else '#1e1e1e',
        font=dict(color='black' if is_light_mode else 'white')
    )
    return fig

def create_cycle_overlay_plot(data, is_light_mode, start_date, end_date, axis_scale):
    """Create a plot with regression lines and shifted graphs using OLS regression."""
    log_days = np.log10(data['Days_Since_Genesis'])
    log_prices = np.log10(data['Price'])
    X = sm.add_constant(log_days)
    y = log_prices
    model = sm.OLS(y, X)
    results = model.fit()
    A = results.params[1]
    C = results.params[0]
    start_days = max((pd.to_datetime(start_date) - GENESIS_DATE).days, 1)
    end_days = (pd.to_datetime(end_date) - GENESIS_DATE).days
    extended_days_since_genesis = np.linspace(start_days, end_days, num=1000)
    predicted_log_prices = A * np.log10(extended_days_since_genesis) + C
    predicted_prices = 10 ** predicted_log_prices

    fig = go.Figure()
    if axis_scale in ['linear', 'log-linear']:
        x_data = data['Time']
        x_pred = GENESIS_DATE + pd.to_timedelta(extended_days_since_genesis, unit='D')
        x_title = 'Date'
        xaxis_type = 'date'
    else:
        x_data = data['Days_Since_Genesis']
        x_pred = extended_days_since_genesis
        x_title = 'Days Since Genesis'
        xaxis_type = 'log'
    yaxis_type = 'linear' if axis_scale == 'linear' else 'log'
    fig.add_trace(go.Scatter(
        x=x_data,
        y=data['Price'],
        mode='lines',
        name='Original Price',
        line=dict(color='#1f77b4'),
        hovertemplate='<b>Date:</b> %{x}<br><b>Price:</b> $%{y:,.2f}<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=x_pred,
        y=predicted_prices,
        mode='lines',
        name='Original OLS Regression Line',
        line=dict(color='orange', width=2),
        hovertemplate='<b>Date:</b> %{x}<br><b>Predicted Price:</b> $%{y:,.2f}<extra></extra>'
    ))
    shift_days_list = [365 * 4, 365 * 8]
    shift_colors = {365 * 4: 'magenta', 365 * 8: 'lightcoral'}
    for shift_days in shift_days_list:
        shifted_data = data.copy()
        shifted_data['Time'] = shifted_data['Time'] + pd.Timedelta(days=shift_days)
        shifted_data['Days_Since_Genesis'] = (shifted_data['Time'] - GENESIS_DATE).dt.days
        if axis_scale in ['linear', 'log-linear']:
            shifted_data = shifted_data[(shifted_data['Time'] >= pd.to_datetime(start_date)) & (shifted_data['Time'] <= pd.to_datetime(end_date))]
        else:
            shifted_data = shifted_data[(shifted_data['Days_Since_Genesis'] >= start_days) & (shifted_data['Days_Since_Genesis'] <= end_days)]
        x_shifted_pred = extended_days_since_genesis
        predicted_shifted_log_prices = A * np.log10(x_shifted_pred) + C
        predicted_shifted_prices = 10 ** predicted_shifted_log_prices
        if axis_scale in ['linear', 'log-linear']:
            x_shifted_pred_line = GENESIS_DATE + pd.to_timedelta(x_shifted_pred + shift_days, unit='D')
            x_shifted_data = shifted_data['Time']
        else:
            x_shifted_pred_line = x_shifted_pred + shift_days
            x_shifted_data = shifted_data['Days_Since_Genesis']
        fig.add_trace(go.Scatter(
            x=x_shifted_data,
            y=shifted_data['Price'],
            mode='lines',
            name=f'Shifted +{shift_days//365} Years',
            line=dict(color=shift_colors[shift_days], width=2, dash='dot'),
            hovertemplate='<b>Date:</b> %{x}<br><b>Shifted Price:</b> $%{y:,.2f}<extra></extra>'
        ))
        fig.add_trace(go.Scatter(
            x=x_shifted_pred_line,
            y=predicted_shifted_prices,
            mode='lines',
            name=f'OLS Regression Line (Shifted +{shift_days//365} Years)',
            line=dict(color=shift_colors[shift_days], width=2, dash='solid'),
            hovertemplate='<b>Date:</b> %{x}<br><b>Predicted Price:</b> $%{y:,.2f}<extra></extra>'
        ))
    colors = {
        'background': '#ffffff' if is_light_mode else '#1e1e1e',
        'text': '#000000' if is_light_mode else '#ffffff',
        'plot_bgcolor': '#f9f9f9' if is_light_mode else '#1e1e1e',
        'gridcolor': '#e1e1e1' if is_light_mode else '#2a2a2a',
    }
    fig.update_layout(
        title='Bitcoin Power Law Fit with OLS Regression',
        xaxis=dict(
            title=x_title,
            type=xaxis_type,
            showgrid=True,
            gridcolor=colors['gridcolor'],
            zeroline=False,
        ),
        yaxis=dict(
            title='Price in USD',
            type=yaxis_type,
            showgrid=True,
            gridcolor=colors['gridcolor'],
            zeroline=False,
        ),
        template='plotly_white' if is_light_mode else 'plotly_dark',
        showlegend=True,
        legend=dict(
            title="Legend",
            orientation="h",
            x=0.5,
            y=-0.2,
            xanchor='center',
            yanchor='top',
            bgcolor='rgba(0,0,0,0)',
            bordercolor='rgba(0,0,0,0)'
        ),
        margin=dict(l=50, r=150, t=80, b=80),
        plot_bgcolor=colors['plot_bgcolor'],
        paper_bgcolor=colors['background'],
        font=dict(color=colors['text']),
        transition=dict(duration=500, easing='cubic-in-out')
    )
    return fig

def get_price_at(data, timestamp):
    """Interpolate the price at a specific timestamp."""
    if timestamp in data['Time'].values:
        return data.loc[data['Time'] == timestamp, 'Price'].iloc[0]
    else:
        before = data[data['Time'] < timestamp].tail(1)
        after = data[data['Time'] > timestamp].head(1)
        if not before.empty and not after.empty:
            time_diff = (after['Time'].iloc[0] - before['Time'].iloc[0]).total_seconds()
            price_diff = after['Price'].iloc[0] - before['Price'].iloc[0]
            time_ratio = (timestamp - before['Time'].iloc[0]).total_seconds() / time_diff
            interpolated_price = before['Price'].iloc[0] + price_diff * time_ratio
            return interpolated_price
        else:
            if not before.empty:
                return before['Price'].iloc[0]
            elif not after.empty:
                return after['Price'].iloc[0]
            else:
                raise ValueError(f"No data available for timestamp {timestamp}")

def create_normalized_overlay_plot(data, is_light_mode):
    """
    Create an overlay plot with selected periods starting at the same point and a logarithmic y-axis.
    """
    periods = [
        ('2015-01-14', '2018-12-15'),
        ('2018-12-15', '2022-11-21'),
        ('2022-11-21', pd.Timestamp.today().strftime('%Y-%m-%d'))
    ]
    current_period_start = periods[-1][0]
    current_start_price = get_price_at(data, pd.Timestamp(current_period_start))
    fig = go.Figure()
    max_days = max((pd.to_datetime(end) - pd.to_datetime(start)).days for start, end in periods)
    colors = ['blue', 'orange', 'green']
    for i, (start, end) in enumerate(periods):
        period_data = data[(data['Time'] >= pd.Timestamp(start)) & (data['Time'] <= pd.Timestamp(end))].copy()
        period_data = period_data.reset_index(drop=True)
        period_data['Days_Since_Start'] = (period_data['Time'] - pd.Timestamp(start)).dt.days
        period_data = period_data[period_data['Days_Since_Start'] <= max_days]
        start_price = get_price_at(data, pd.Timestamp(start))
        period_data['Adjusted_Price'] = (period_data['Price'] / start_price) * current_start_price
        period_data['Aligned_Time'] = pd.Timestamp(current_period_start) + pd.to_timedelta(period_data['Days_Since_Start'], unit='D')
        fig.add_trace(go.Scatter(
            x=period_data['Aligned_Time'],
            y=period_data['Adjusted_Price'],
            mode='lines',
            name=f'{start} to {end}',
            line=dict(color=colors[i]),
            hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>Adjusted Price:</b> $%{y:,.2f}<extra></extra>'
        ))
    colors_layout = {
        'background': '#ffffff' if is_light_mode else '#1e1e1e',
        'text': '#000000' if is_light_mode else '#ffffff',
        'plot_bgcolor': '#f9f9f9' if is_light_mode else '#1e1e1e',
        'gridcolor': '#e1e1e1' if is_light_mode else '#2a2a2a',
    }
    fig.update_layout(
        title='Overlay of Selected Periods with Prices Adjusted to Current Period Starting from 2022-11-21',
        xaxis_title='Date',
        yaxis_title='Adjusted Price (USD)',
        xaxis=dict(
            tickformat="%b %Y",
            dtick="M3",
            showgrid=True,
            gridcolor=colors_layout['gridcolor'],
            tickangle=45,
            title_font=dict(color=colors_layout['text']),
            tickfont=dict(color=colors_layout['text']),
        ),
        yaxis=dict(
            type='log',
            showgrid=True,
            tickprefix="$",
            title='Price in USD',
            gridcolor=colors_layout['gridcolor'],
            title_font=dict(color=colors_layout['text']),
            tickfont=dict(color=colors_layout['text']),
        ),
        template='plotly_white' if is_light_mode else 'plotly_dark',
        showlegend=True,
        legend=dict(
            title="Legend",
            orientation="h",
            x=0.5,
            y=-0.2,
            xanchor='center',
            yanchor='top',
            bgcolor='rgba(0,0,0,0)',
            bordercolor='rgba(0,0,0,0)'
        ),
        margin=dict(l=50, r=150, t=80, b=80),
        plot_bgcolor=colors_layout['plot_bgcolor'],
        paper_bgcolor=colors_layout['background'],
        font=dict(color=colors_layout['text']),
        transition=dict(duration=500, easing='cubic-in-out')
    )
    return fig

def create_seasonality_plot(data, is_light_mode=True):
    """
    Create a plot showing mean monthly returns over a 48-month cycle with repeating calendar month labels.
    """
    data = data.copy()
    if 'Time' not in data.columns:
        raise ValueError("The 'Time' column is missing from the data.")
    if not pd.api.types.is_datetime64_any_dtype(data['Time']):
        data['Time'] = pd.to_datetime(data['Time'])
    data.set_index('Time', inplace=True)
    monthly_data = data['Price'].resample('M').last()
    monthly_returns = monthly_data.pct_change().dropna() * 100
    monthly_returns = monthly_returns.reset_index()
    monthly_returns.rename(columns={'Price': 'Monthly_Return'}, inplace=True)
    monthly_returns['48_Month_Cycle_Month'] = (((monthly_returns['Time'].dt.year - monthly_returns['Time'].dt.year.min()) * 12
                                                + monthly_returns['Time'].dt.month - 1) % 48) + 1
    monthly_returns['Year_in_Cycle'] = ((monthly_returns['48_Month_Cycle_Month'] - 1) // 12) + 1
    monthly_returns['Month_in_Year'] = ((monthly_returns['48_Month_Cycle_Month'] - 1) % 12) + 1
    cycle_monthly_returns = (monthly_returns.groupby(['48_Month_Cycle_Month', 'Year_in_Cycle', 'Month_in_Year'])['Monthly_Return']
                             .mean().reset_index())
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    cycle_monthly_returns['Month_Label'] = cycle_monthly_returns['Month_in_Year'].apply(lambda x: months[x - 1])
    cycle_monthly_returns['X_Label'] = cycle_monthly_returns['Month_Label'] + ' (' + cycle_monthly_returns['48_Month_Cycle_Month'].astype(str) + ')'
    color_mapping = {1: 'purple', 2: 'blue', 3: 'orange', 4: 'green'}
    cycle_monthly_returns['Color'] = cycle_monthly_returns['Year_in_Cycle'].map(color_mapping)
    cycle_monthly_returns.sort_values(by='48_Month_Cycle_Month', inplace=True)
    fig = go.Figure()
    for year in range(1, 5):
        year_data = cycle_monthly_returns[cycle_monthly_returns['Year_in_Cycle'] == year]
        fig.add_trace(go.Bar(
            x=year_data['X_Label'],
            y=year_data['Monthly_Return'],
            name=f'Year {year}',
            marker_color=color_mapping[year]
        ))
    overall_avg_return = monthly_returns['Monthly_Return'].mean()
    fig.add_trace(go.Scatter(
        x=cycle_monthly_returns['X_Label'],
        y=[overall_avg_return] * len(cycle_monthly_returns),
        mode='lines',
        name='Overall Mean Monthly Return',
        line=dict(color='red', dash='dash')
    ))
    fig.update_layout(
        title='Mean Monthly Returns Over 48-Month Cycle for Bitcoin',
        xaxis_title='Month in 48-Month Cycle',
        yaxis_title='Mean Monthly Return (%)',
        template='plotly_white' if is_light_mode else 'plotly_dark',
        yaxis_tickformat='.2f',
        showlegend=True,
        plot_bgcolor='#ffffff' if is_light_mode else '#1e1e1e',
        paper_bgcolor='#ffffff' if is_light_mode else '#1e1e1e',
        font=dict(color='black' if is_light_mode else 'white'),
        barmode='group'
    )
    return fig

def create_mayer_multiple_plot(data, is_light_mode):
    """
    Create a Mayer Multiple plot for Bitcoin.
    """
    data['200DMA'] = data['Price'].rolling(window=200).mean()
    data['Mayer_Multiple'] = data['Price'] / data['200DMA']
    data = data.dropna(subset=['Mayer_Multiple'])
    if data.empty:
        fig = go.Figure()
        fig.update_layout(
            title='Mayer Multiple Plot',
            xaxis_title='Date',
            yaxis_title='Mayer Multiple',
            template='plotly_white' if is_light_mode else 'plotly_dark',
            showlegend=True,
            plot_bgcolor='#ffffff' if is_light_mode else '#1e1e1e',
            paper_bgcolor='#ffffff' if is_light_mode else '#1e1e1e',
            font=dict(color='black' if is_light_mode else 'white'),
            annotations=[dict(
                text='No data available for Mayer Multiple calculation.',
                xref='paper', yref='paper', showarrow=False,
                font=dict(size=16, color='red'))
            ]
        )
        return fig
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data['Time'],
        y=data['Mayer_Multiple'],
        mode='lines',
        name='Mayer Multiple',
        line=dict(color='#FFA500')
    ))
    fig.add_hline(y=2.4, line_color='red', line_dash='dash',
                  annotation_text='Overbought Threshold (2.4)', annotation_position="top left")
    fig.add_hline(y=1.0, line_color='green', line_dash='dot',
                  annotation_text='Fair Value (1.0)', annotation_position="top left")
    fig.update_layout(
        title='Bitcoin Mayer Multiple (Price / 200-Day Moving Average)',
        xaxis_title='Date',
        yaxis_title='Mayer Multiple',
        yaxis_tickformat='.1f',
        template='plotly_white' if is_light_mode else 'plotly_dark',
        showlegend=True,
        plot_bgcolor='#ffffff' if is_light_mode else '#1e1e1e',
        paper_bgcolor='#ffffff' if is_light_mode else '#1e1e1e',
        font=dict(color='black' if is_light_mode else 'white')
    )
    return fig

def create_daily_returns_avg_plot(data, is_light_mode, target_prices=[120000, 150000, 180000], target_date='2025-07-31'):
    """
    Create a plot showing average daily returns and required returns to reach target prices.
    """
    if not pd.api.types.is_datetime64_any_dtype(data['Time']):
        data['Time'] = pd.to_datetime(data['Time'])
    if not isinstance(target_date, pd.Timestamp):
        target_date = pd.to_datetime(target_date)
    current_date = data['Time'].iloc[-1]
    days_to_target = (target_date - current_date).days
    if days_to_target <= 0:
        raise ValueError("Target date must be after the last date in the data.")
    data['Daily_Returns'] = data['Price'].pct_change()
    data['Avg_Daily_Returns'] = data['Daily_Returns'].rolling(window=60).mean()
    current_price = data['Price'].iloc[-1]
    required_returns = {}
    for target_price in target_prices:
        required_return = (target_price / current_price) ** (1 / days_to_target) - 1
        required_returns[target_price] = required_return
    slope, intercept, r_value = perform_regression(data)
    data_non_zero = data[data['Days_Since_Genesis'] > 0].copy()
    data_non_zero['Predicted_Daily_Returns'] = ((data_non_zero['Days_Since_Genesis'] + 1) / data_non_zero['Days_Since_Genesis']) ** slope - 1
    data = data.merge(data_non_zero[['Time', 'Predicted_Daily_Returns']], on='Time', how='left')
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data['Time'],
        y=data['Avg_Daily_Returns'],
        mode='lines',
        name='60-Day Avg Daily Returns',
        line=dict(color='#1f77b4', width=2),
        hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>Avg Daily Return:</b> %{y:.4%}<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=data['Time'],
        y=data['Predicted_Daily_Returns'],
        mode='lines',
        name='Daily Returns (Power Law Regression)',
        line=dict(color='orange', width=2),
        hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>Predicted Daily Return:</b> %{y:.4%}<extra></extra>'
    ))
    colors = ['blue', 'red', 'green', 'purple', 'brown']
    for idx, (target_price, required_return) in enumerate(required_returns.items()):
        fig.add_trace(go.Scatter(
            x=[data['Time'].iloc[0], data['Time'].iloc[-1]],
            y=[required_return, required_return],
            mode='lines',
            name=f'Required Return for ${target_price:,.0f}',
            line=dict(color=colors[idx % len(colors)], dash='dash', width=2),
            hovertemplate=(f'<b>Required Return to reach ${target_price:,.0f} by {target_date.strftime("%Y-%m-%d")}:</b> '
                           '%{y:.4%}<extra></extra>')
        ))
    fig.update_layout(
        title=f'Daily Returns and Required Returns to Reach Targets by {target_date.strftime("%Y-%m-%d")}',
        xaxis_title='Date',
        yaxis_title='Daily Return',
        template='plotly_white' if is_light_mode else 'plotly_dark',
        yaxis_tickformat='.2%',
        font=dict(color='black' if is_light_mode else 'white'),
        legend_title='Legend',
        hovermode='x unified'
    )
    return fig

def create_required_return_plot(data, is_light_mode, target_prices=[200000, 300000, 500000], target_date='2025-11-30'):
    """
    Create a plot showing required daily returns to reach target prices.
    """
    if not pd.api.types.is_datetime64_any_dtype(data['Time']):
        data['Time'] = pd.to_datetime(data['Time'])
    if not isinstance(target_date, pd.Timestamp):
        target_date = pd.to_datetime(target_date)
    current_date = data['Time'].iloc[-1]
    days_to_target = (target_date - current_date).days
    if days_to_target <= 0:
        raise ValueError("Target date must be after the last date in the data.")
    data['Daily_Returns'] = data['Price'].pct_change()
    data['Avg_Daily_Returns'] = data['Daily_Returns'].rolling(window=365).mean()
    current_price = data['Price'].iloc[-1]
    required_returns = {}
    for target_price in target_prices:
        required_return = (target_price / current_price) ** (1 / days_to_target) - 1
        required_returns[target_price] = required_return
    slope, intercept, r_value = perform_regression(data)
    data_non_zero = data[data['Days_Since_Genesis'] > 0].copy()
    data_non_zero['Predicted_Daily_Returns'] = ((data_non_zero['Days_Since_Genesis'] + 1) / data_non_zero['Days_Since_Genesis']) ** slope - 1
    data = data.merge(data_non_zero[['Time', 'Predicted_Daily_Returns']], on='Time', how='left')
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data['Time'],
        y=data['Avg_Daily_Returns'],
        mode='lines',
        name='60-Day Avg Daily Returns',
        line=dict(color='#1f77b4', width=2),
        hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>Avg Daily Return:</b> %{y:.4%}<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=data['Time'],
        y=data['Predicted_Daily_Returns'],
        mode='lines',
        name='Daily Returns (Power Law Regression)',
        line=dict(color='orange', width=2),
        hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>Predicted Daily Return:</b> %{y:.4%}<extra></extra>'
    ))
    colors = ['blue', 'red', 'green', 'purple', 'brown']
    for idx, (target_price, required_return) in enumerate(required_returns.items()):
        fig.add_trace(go.Scatter(
            x=[data['Time'].iloc[0], data['Time'].iloc[-1]],
            y=[required_return, required_return],
            mode='lines',
            name=f'Required Return for ${target_price:,.0f}',
            line=dict(color=colors[idx % len(colors)], dash='dash', width=2),
            hovertemplate=(f'<b>Required Return to reach ${target_price:,.0f} by {target_date.strftime("%Y-%m-%d")}:</b> '
                           '%{y:.4%}<extra></extra>')
        ))
    fig.update_layout(
        title=f'Daily Returns and Required Returns to Reach Targets by {target_date.strftime("%Y-%m-%d")}',
        xaxis_title='Date',
        yaxis_title='Daily Return',
        template='plotly_white' if is_light_mode else 'plotly_dark',
        yaxis_tickformat='.2%',
        font=dict(color='black' if is_light_mode else 'white'),
        legend_title='Legend',
        hovermode='x unified'
    )
    return fig

def perform_rolling_regression_intercept(data, window=365):
    """
    Perform a rolling regression on log-transformed data.
    """
    slopes = []
    intercepts = []
    times = []
    for i in range(len(data) - window + 1):
        window_data = data.iloc[i:i + window]
        log_date = np.log(window_data['Days_Since_Genesis'])
        log_price = np.log(window_data['Price'])
        valid = ~(np.isnan(log_date) | np.isnan(log_price) | np.isinf(log_date) | np.isinf(log_price))
        log_date = log_date[valid]
        log_price = log_price[valid]
        if len(log_date) > 1:
            slope, intercept, _, _, _ = linregress(log_date, log_price)
            slopes.append(slope)
            intercepts.append(intercept)
            times.append(window_data['Time'].iloc[-1])
    return pd.DataFrame({'Time': times, 'Slope': slopes, 'Intercept': intercepts})

def create_regression_fit_quality_plot(data, is_light_mode, window=365):
    """
    Create a plot showing the rolling regression fit quality over time.
    """
    rolling_regression = perform_rolling_regression_intercept(data, window)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rolling_regression['Time'],
        y=rolling_regression['Slope'],
        mode='lines',
        name='Rolling Slope',
        line=dict(color='blue'),
        hovertemplate='<b>Date:</b> %{x}<br><b>Slope:</b> %{y:.4f}<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=rolling_regression['Time'],
        y=rolling_regression['Intercept'],
        mode='lines',
        name='Rolling Intercept',
        line=dict(color='orange'),
        hovertemplate='<b>Date:</b> %{x}<br><b>Intercept:</b> %{y:.4f}<extra></extra>'
    ))
    slope_mean = rolling_regression['Slope'].mean()
    slope_std = rolling_regression['Slope'].std()
    intercept_mean = rolling_regression['Intercept'].mean()
    intercept_std = rolling_regression['Intercept'].std()
    fig.add_hline(y=slope_mean, line_color='green', line_dash='dash', annotation_text='Mean Slope')
    fig.add_hline(y=slope_mean + slope_std, line_color='red', line_dash='dot', annotation_text='+1 Std Dev Slope')
    fig.add_hline(y=slope_mean - slope_std, line_color='red', line_dash='dot', annotation_text='-1 Std Dev Slope')
    fig.add_hline(y=slope_mean + 2 * slope_std, line_color='blue', line_dash='dot', annotation_text='+2 Std Dev Slope')
    fig.add_hline(y=slope_mean - 2 * slope_std, line_color='blue', line_dash='dot', annotation_text='-2 Std Dev Slope')
    fig.add_hline(y=intercept_mean, line_color='purple', line_dash='dash', annotation_text='Mean Intercept')
    fig.add_hline(y=intercept_mean + intercept_std, line_color='red', line_dash='dot', annotation_text='+1 Std Dev Intercept')
    fig.add_hline(y=intercept_mean - intercept_std, line_color='red', line_dash='dot', annotation_text='-1 Std Dev Intercept')
    fig.add_hline(y=intercept_mean + 2 * intercept_std, line_color='blue', line_dash='dot', annotation_text='+2 Std Dev Intercept')
    fig.add_hline(y=intercept_mean - 2 * intercept_std, line_color='blue', line_dash='dot', annotation_text='-2 Std Dev Intercept')
    fig.update_layout(
        title='Regression Fit Quality Over Time',
        xaxis_title='Date',
        yaxis_title='Value',
        template='plotly_white' if is_light_mode else 'plotly_dark',
        showlegend=True,
        plot_bgcolor='#ffffff' if is_light_mode else '#1e1e1e',
        paper_bgcolor='#ffffff' if is_light_mode else '#1e1e1e',
        font=dict(color='black' if is_light_mode else 'white')
    )
    return fig



# NEW: Monte Carlo simulation function
def monte_carlo_paths(data, years, n_paths=200):
    daily_ret = data['Price'].pct_change().dropna()
    mu, sigma = daily_ret.mean(), daily_ret.std()
    last = data['Price'].iloc[-1]
    days = int(years*365)
    sim = np.zeros((days, n_paths))
    for i in range(n_paths):
        shock = np.random.normal(mu, sigma, days)
        sim[:,i] = last * np.cumprod(1+shock)
    dates = pd.date_range(data['Time'].iloc[-1], periods=days, freq='D')
    return dates, sim

# ----------------
# Modern Layout
# ----------------
def create_metric_card(title, value, change=None, icon="fa-chart-line", color="primary"):
    change_element = html.Span([
        html.I(className=f"fas fa-arrow-{'up' if change and change >= 0 else 'down'} me-1"),
        f"{abs(change):.2f}%" if change is not None else ""
    ], className=f"text-{'success' if change and change >= 0 else 'danger'}") if change is not None else ""
    
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.I(className=f"fas {icon} fa-2x text-{color} mb-3"),
                html.H5(title, className="card-title mb-1"),
                html.Div([
                    html.H4(value, className="mb-0"),
                    html.Small(change_element, className="ms-2") if change is not None else ""
                ], className="d-flex align-items-center")
            ], className="text-center")
        ])
    ], className="h-100")

app.layout = dbc.Container(fluid=True, className="py-3", style={"minHeight": "100vh"}, children=[
    dcc.Store(id='theme-store', data='dark'),
    dcc.Interval(id='price-updater', interval=60*1000),  # Update every minute
    
    # Top Navigation Bar
    dbc.Navbar(
        dbc.Container([
            html.A(
                dbc.Row([
                    dbc.Col(html.Img(src="https://upload.wikimedia.org/wikipedia/commons/4/46/Bitcoin.svg", height="30px"), className="me-2"),
                    dbc.Col(dbc.NavbarBrand("Bitcoin Analytics Dashboard", className="ms-2")),
                ], align="center", className="g-0"),
                href="#",
                style={"textDecoration": "none"},
            ),
            dbc.Nav([
                dbc.Button(html.I(className="fas fa-moon"), id="theme-toggle", className="me-2"),
                dbc.DropdownMenu(
                    [
                        dbc.DropdownMenuItem("Documentation", href="#"),
                        dbc.DropdownMenuItem("GitHub", href="#"),
                        dbc.DropdownMenuItem("About", href="#"),
                    ],
                    nav=True,
                    in_navbar=True,
                    label="Help",
                ),
            ], className="ms-auto"),
        ]),
        color="primary",
        dark=True,
        className="mb-4 shadow"
    ),
    
    # Main Content
    dbc.Row([
        # Sidebar Controls
        dbc.Col(width=3, className="pe-2", children=[
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col(Lottie(options={'loop':True,'autoplay':True}, width='100%', height='80px', url=LOTTIE_URL)),
                    ], className="mb-3"),
                    
                    dbc.Row([
                        dbc.Col(html.H4("Controls", className="text-center mb-3")),
                    ]),
                    
                    # Date Range Picker
                    dbc.Row([
                        dbc.Col(html.Label("Date Range", className="mb-1")),
                        dbc.Col(dcc.DatePickerRange(
                            id='date-range',
                            start_date=DEFAULT_START,
                            end_date=DEFAULT_END,
                            className="mb-3"
                        ), width=12)
                    ]),
                    
                    # Theme and Scale Selectors
                    dbc.Row([
                        dbc.Col([
                            html.Label("Theme", className="mb-1"),
                            dbc.RadioItems(
                                id='theme-selector',
                                options=[
                                    {'label': [html.I(className="fas fa-sun me-2"), "Light"], 'value': 'light'},
                                    {'label': [html.I(className="fas fa-moon me-2"), "Dark"], 'value': 'dark'}
                                ],
                                value='dark',
                                inline=True,
                                className="btn-group",
                                inputClassName="btn-check",
                                labelClassName="btn btn-outline-primary",
                                labelCheckedClassName="active",
                            )
                        ], width=6),
                        
                        dbc.Col([
                            html.Label("Scale", className="mb-1"),
                            dbc.RadioItems(
                                id='axis-scale',
                                options=[
                                    {'label': "Log-Log", 'value': 'log-log'},
                                    {'label': "Log-Linear", 'value': 'log-linear'},
                                   
                                ],
                                value='log-log',
                                inline=True,
                                className="btn-group",
                                inputClassName="btn-check",
                                labelClassName="btn btn-outline-primary",
                                labelCheckedClassName="active",
                            )
                        ], width=6),
                    ], className="mb-3"),
                    
                    # Analysis Type Selector
                    dbc.Row([
                        dbc.Col([
                            html.Label("Analysis Type", className="mb-1"),
                            dcc.Dropdown(
                                id='plot-type',
                                options=[
                                    {'label':'Log-Log Regression', 'value':'log'}, 
                                    {'label':'4-Year Average', 'value':'4yr_average_power_law'}, 
                                    {'label': 'Weinstein Stage Analysis', 'value': 'weinstein'},
                                    {'label':'Deviation Analysis', 'value':'deviation_analysis'},
                                    {'label':'Price Spiral', 'value':'spiral'},
                                    {'label':'Mayer Multiple', 'value':'mayer_multiple'},
                                    {'label':'Daily Returns', 'value':'daily_returns_avg'},
                                    {'label':'Required Returns', 'value':'required_returns'},
                                    {'label':'Seasonality', 'value':'seasonality'},
                                    {'label':'Regression Quality', 'value':'regression_fit_quality'}
                                ],
                                value='log',
                                clearable=False,
                                className="mb-3"
                            )
                        ]),
                    ]),
                    
                    # Investment Calculator
                    dbc.Row([
                        dbc.Col(html.H5("Investment Calculator", className="text-center mb-3")),
                    ]),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.InputGroup([
                                dbc.InputGroupText("BTC Amount"),
                                dbc.Input(id='btc-amt', type='number', value=1, min=0, step=0.1)
                            ], className="mb-2"),
                        ]),
                        
                        dbc.Col([
                            dbc.InputGroup([
                                dbc.InputGroupText("Years Ahead"),
                                dbc.Input(id='years-future', type='number', value=1, min=0.1)
                            ], className="mb-2"),
                        ]),
                    ]),
                    
                    dbc.Row([
                        dbc.Col(
                            dbc.Button(
                                [html.I(className="fas fa-calculator me-2"), "Calculate Projection"],
                                id='run-sim',
                                color='primary',
                                className='w-100'
                            ),
                            width=12
                        )
                    ], className="mb-3"),
                    
                    # Prediction Results
                    dbc.Card(id='prediction-results', className="mt-3 shadow-sm"),
                    
                    # Current Price Ticker
                    dbc.Row([
                        dbc.Col(html.Div(id='current-price-ticker', className="text-center mt-3 p-2 rounded"))
                    ])
                ])
            ], className="h-100 shadow", style=GLASS_STYLE)
        ]),
        
        # Main Content Area
        dbc.Col(width=9, children=[
            dbc.Row([
                # Key Metrics Cards
                dbc.Col(create_metric_card("Current Price", "$--", 0, "fa-coins", "warning"), width=3, id='current-price-card'),
                dbc.Col(create_metric_card("Market Cap", "$--", 0, "fa-globe", "info"), width=3, id='market-cap-card'),
                dbc.Col(create_metric_card("24h Volume", "$--", 0, "fa-chart-bar", "success"), width=3, id='volume-card'),
                dbc.Col(create_metric_card("Fear & Greed", "--", 0, "fa-gauge-high", "danger"), width=3, id='fear-greed-card'),
            ], className="mb-4"),
            
            # Main Graph Area
            dbc.Row([
                dbc.Col(
                    dcc.Loading(
                        dcc.Graph(id='main-graph', animate=True, 
                                  config={'displayModeBar': True, 'modeBarButtonsToAdd': ['drawline', 'drawopenpath']}),
                        type="cube",
                        color="#0d6efd",
                        className="mb-4"
                    ), width=12
                )
            ]),
            
            # Additional Analysis Area
            dbc.Row([
                dbc.Col(
                    dcc.Loading(
                        dcc.Graph(id='secondary-graph', animate=True),
                        type="circle",
                        className="mb-4"
                    ), width=12
                )
            ]),
            
            # Monte Carlo Simulation
            dbc.Row([
                dbc.Col(html.Div(id='sim-output', className="mt-4"), width=12)
            ])
        ])
    ])
])

# ----------------
# Enhanced Callbacks
# ----------------
@app.callback(
    Output('current-price-ticker', 'children'),
    [Input('price-updater', 'n_intervals')]
)
def update_price_ticker(n):
    _, price, change = fetch_latest_price()
    if price is None:
        return html.Div("Loading price...", className="text-warning")
    
    change_color = "success" if change >= 0 else "danger"
    change_icon = "fa-arrow-up" if change >= 0 else "fa-arrow-down"
    
    return [
        html.H5("Current Bitcoin Price", className="mb-1"),
        html.H3(f"${price:,.2f}", className="mb-1"),
        html.Small([
            html.I(className=f"fas {change_icon} me-1"),
            f"{abs(change):.2f}% 24h"
        ], className=f"text-{change_color}")
    ]

@app.callback(
    [Output('main-graph', 'figure'),
     Output('secondary-graph', 'figure')],
    [Input('date-range', 'start_date'),
     Input('date-range', 'end_date'),
     Input('theme-selector', 'value'),
     Input('axis-scale', 'value'),
     Input('plot-type', 'value')],
    prevent_initial_call=False
)
def update_graphs(start, end, theme, axis_scale, ptype):
    data = load_data()
    if data.empty:
        return go.Figure(), go.Figure()
    
    start_dt, end_dt = pd.to_datetime(start), pd.to_datetime(end)
    is_light = (theme == 'light')
    
    # Main graph based on selected analysis type
    graph_funcs = {
        'log': lambda: create_log_log_plot(data, is_light, start_dt, end_dt, axis_scale),
        '4yr_average_power_law': lambda: create_4yr_average_power_law_plot(data, is_light),
        'weinstein': lambda: create_weinstein_stage_plot(data, is_light),
        'deviation_analysis': lambda: create_deviation_analysis_plot(data, is_light),
        'spiral': lambda: create_spiral_plot(data, is_light),
        'mayer_multiple': lambda: create_mayer_multiple_plot(data, is_light),
        'daily_returns_avg': lambda: create_daily_returns_avg_plot(data, is_light),
        'required_returns': lambda: create_required_return_plot(data, is_light),
        'seasonality': lambda: create_seasonality_plot(data, is_light),
        'regression_fit_quality': lambda: create_regression_fit_quality_plot(data, is_light)
    }
    
    try:
        main_fig = graph_funcs.get(ptype, lambda: go.Figure())()
    except Exception as e:
        logging.error(f"Error creating main graph: {e}")
        main_fig = go.Figure()
    
    # Secondary graph with complementary analysis
    try:
        if ptype == 'log':
            secondary_fig = create_4yr_average_power_law_plot(data, is_light)
        elif ptype == '4yr_average_power_law':
            secondary_fig = create_deviation_analysis_plot(data, is_light)
        else:
            secondary_fig = create_log_log_plot(data, is_light, start_dt, end_dt, axis_scale)
    except Exception as e:
        logging.error(f"Error creating secondary graph: {e}")
        secondary_fig = go.Figure()
    
    return main_fig, secondary_fig

@app.callback(
    Output('prediction-results', 'children'),
    [Input('btc-amt', 'value'),
     Input('years-future', 'value')]
)
def update_prediction_results(num_bitcoins, years_future):
    if num_bitcoins is None or years_future is None:
        return dbc.Alert("Enter values to calculate projection", color="info")

    data = load_data()
    if data.empty:
        return dbc.Alert("Data not available", color="danger")

    # Try to fetch live price; if it fails, fall back to last known price
    _, live_price, _ = fetch_latest_price()
    if live_price is None:
        # fallback
        live_price = data['Price'].iloc[-1]
    live_value_str = f"${live_price * num_bitcoins:,.2f}"

    # Compute future projections (based on historical regression)
    try:
        res = calculate_future_values(data, num_bitcoins, years_future)
    except Exception as e:
        logging.error(f"Error calculating future values: {e}")
        return dbc.Alert("Error in projection calculation", color="danger")

    return dbc.Card([
        dbc.CardHeader("Projection Results"),
        dbc.CardBody([
            dbc.ListGroup([
                # Live current value
                dbc.ListGroupItem([
                    html.Div("Current Value", className="fw-bold"),
                    html.Div(live_value_str)
                ], className="d-flex justify-content-between"),

                # Future projection
                dbc.ListGroupItem([
                    html.Div(f"{years_future}-Year Projection", className="fw-bold"),
                    html.Div(res['future_value_str'], className="text-success fw-bold")
                ], className="d-flex justify-content-between"),

                # Growth stats
                dbc.ListGroupItem([
                    html.Div("Annual Growth", className="fw-bold"),
                    html.Div(res['per_year_increase_str'], className="text-success")
                ], className="d-flex justify-content-between"),
                dbc.ListGroupItem([
                    html.Div("Monthly Growth", className="fw-bold"),
                    html.Div(res['per_month_increase_str'], className="text-success")
                ], className="d-flex justify-content-between"),
                dbc.ListGroupItem([
                    html.Div("Daily Growth", className="fw-bold"),
                    html.Div(res['per_day_increase_str'], className="text-success")
                ], className="d-flex justify-content-between"),

                # Model accuracy
                dbc.ListGroupItem([
                    html.Div("Model Accuracy (R²)", className="fw-bold"),
                    html.Div(f"{res['r_squared']:.4f}", className="text-info")
                ], className="d-flex justify-content-between")
            ], flush=True)
        ])
    ])

@app.callback(
    Output('sim-output', 'children'),
    [Input('run-sim', 'n_clicks')],
    [State('btc-amt', 'value'), 
     State('years-future', 'value'),
     State('theme-selector', 'value')]
)
def run_simulation(n, amt, yrs, theme):
    if not n:
        raise PreventUpdate
        
    data = load_data()
    if data.empty:
        return dbc.Alert("Data not available for simulation", color="danger")
    
    try:
        dates, paths = monte_carlo_paths(data, yrs)
    except Exception as e:
        logging.error(f"Error in Monte Carlo simulation: {e}")
        return dbc.Alert("Error in simulation", color="danger")
    
    fig = go.Figure()
    
    # Add all simulation paths
    for i in range(min(100, paths.shape[1])):
        fig.add_trace(go.Scatter(
            x=dates, 
            y=paths[:, i] * amt, 
            mode='lines',
            line=dict(width=1, color='rgba(13, 110, 253, 0.1)'),
            showlegend=False
        ))
    
    # Calculate statistics
    median = np.median(paths, axis=1) * amt
    upper_90 = np.percentile(paths, 90, axis=1) * amt
    lower_10 = np.percentile(paths, 10, axis=1) * amt
    
    # Add statistical traces
    fig.add_trace(go.Scatter(
        x=dates, y=median, 
        mode='lines', 
        line=dict(color='#FFA500', width=3),
        name='Median'
    ))
    
    fig.add_trace(go.Scatter(
        x=dates, y=upper_90,
        fill=None,
        mode='lines',
        line=dict(color='rgba(40, 167, 69, 0.2)'),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=dates, y=lower_10,
        fill='tonexty',
        mode='lines',
        line=dict(color='rgba(40, 167, 69, 0.2)'),
        name='80% Confidence'
    ))
    
    # Add current value marker
    fig.add_trace(go.Scatter(
        x=[dates[0]], y=[data['Price'].iloc[-1] * amt],
        mode='markers',
        marker=dict(size=12, color='#DC3545'),
        name='Current Value'
    ))
    
    # Update layout
    is_light = (theme == 'light')
    colors = COLOR_SCHEME['light'] if is_light else COLOR_SCHEME['dark']
    
    fig.update_layout(
        title=f'Monte Carlo Simulation ({yrs} Year Projection)',
        template='plotly_white' if is_light else 'plotly_dark',
        xaxis_title='Date',
        yaxis_title='Portfolio Value (USD)',
        hovermode='x unified',
        plot_bgcolor=colors['card'],
        paper_bgcolor=colors['bg'],
        font=dict(color=colors['text'])
    )
    
    return dcc.Graph(figure=fig, className="shadow")

@app.callback(
    [Output('theme-store', 'data'),
     Output('theme-toggle', 'children')],
    [Input('theme-toggle', 'n_clicks')],
    [State('theme-store', 'data')]
)
def toggle_theme(n, current_theme):
    if not n:
        return current_theme, html.I(className="fas fa-moon")
    
    new_theme = 'light' if current_theme == 'dark' else 'dark'
    icon = html.I(className="fas fa-sun") if new_theme == 'light' else html.I(className="fas fa-moon")
    return new_theme, icon

@app.callback(
    [Output('current-price-card', 'children'),
     Output('market-cap-card', 'children'),
     Output('volume-card', 'children'),
     Output('fear-greed-card', 'children')],
    [Input('price-updater', 'n_intervals')]
)
def update_metrics_cards(n):
    # Placeholder for real metrics - in a real app you'd fetch these from APIs
    _, price, change = fetch_latest_price()
    
    return (
        create_metric_card("Current Price", f"${price:,.2f}" if price else "$--", 
                           change, "fa-coins", "warning").children,
        create_metric_card("Market Cap", "$1.2T", 2.5, "fa-globe", "info").children,
        create_metric_card("24h Volume", "$42.5B", -1.2, "fa-chart-bar", "success").children,
        create_metric_card("Fear & Greed", "75 (Greed)", 5, "fa-gauge-high", "danger").children
    )

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
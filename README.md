# Cryptofied_Unsupervised_Dynamic_Portfolio_Allocation

Main Code is available in this file : 25MarFusion02PipelineRQAclusters.ipynb

## Overview
This project analyzes historical cryptocurrency data to identify trends and patterns using various statistical and machine learning techniques. Key features include:

- Downloading and processing cryptocurrency data.
- Generating technical indicators (RSI, ATR, etc.).
- Applying clustering algorithms for momentum segmentation.
- Visualizing results and trends.

## Installation
To set up the environment, ensure you have Python 3.8+ installed. Then, install the required packages using:

```bash
pip install pandas numpy matplotlib statsmodels pandas_datareader yfinance scikit-learn PyPortfolioOpt pandas-ta
```

## Data
### Description
The data consists of historical cryptocurrency information, including:
- Adjusted Close Prices
- Volume
- High/Low Prices
- Open Prices

Additional computed metrics include:
- **Relative Strength Index (RSI)**
- **Average True Range (ATR)**
- **Returns (1m, 2m, ... 12m)**

Data is aggregated and normalized for machine learning applications, including clustering and visualization.

### Data Download
Data is downloaded using Open Source Yahoo Finance for a predefined list of cryptocurrency tickers (top 200). Example tickers include `BTC-USD`, `ETH-USD`, `ADA-USD`, and more.

### Code for Download and Preprocessing

```python
import yfinance as yf
import pandas as pd
import pandas_ta
import numpy as np

# Define the list of cryptocurrency symbols
crypto_symbols = ["BTC-USD", "ETH-USD", "USDT-USD", "BNB-USD"]  # Add more symbols as needed

# Define start and end dates
end_date = '2024-01-01'
start_date = pd.to_datetime(end_date) - pd.DateOffset(days=365*10)

# Download historical data for the specified symbols and date range
crypto_data = yf.download(tickers=crypto_symbols, start=start_date, end=end_date).stack(level=1)
crypto_data.index.names = ['Date', 'Ticker']  # Set the column headings for the index levels
crypto_data.columns = crypto_data.columns.str.lower()  # Lowercase column names

# Display the first few rows
print(crypto_data.head())

# Add indicators like RSI and ATR
def compute_atr(stock_data):
    atr = pandas_ta.atr(high=stock_data['high'], low=stock_data['low'], close=stock_data['close'], length=14)
    return atr.sub(atr.mean()).div(atr.std())

crypto_data['rsi'] = crypto_data.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.rsi(close=x, length=20))
crypto_data['atr'] = crypto_data.groupby(level=1, group_keys=False).apply(compute_atr)

# Save processed data
crypto_data.to_csv('processed_crypto_data.csv')
```

## Features and Indicators
- **Relative Strength Index (RSI)**: Measures the speed and change of price movements.
- **Average True Range (ATR)**: Indicates market volatility.
- **Dollar Volume**: Adjusted close price multiplied by volume.
- **Returns**: Computed for different time horizons (1 month, 2 months, etc.).

## Clustering Analysis
### Techniques Used
- **K-Means**: Segments cryptocurrencies based on indicators like RSI and ATR.
- **Gaussian Mixture Models (GMM)**: Identifies probabilistic clusters.
- **DBSCAN**: Detects clusters and noise.

### Example Code for Clustering
```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Normalize the data
scaler = StandardScaler()
data[['atr', 'rsi']] = scaler.fit_transform(data[['atr', 'rsi']])

# Apply KMeans clustering
kmeans = KMeans(n_clusters=5, random_state=0)
data['cluster'] = kmeans.fit_predict(data[['atr', 'rsi']])

# Save clustered data
data.to_csv('clustered_crypto_data.csv')
```

## Visualization
Clusters and trends are visualized using Matplotlib and Seaborn. Example:

```python
import matplotlib.pyplot as plt

# Plot clusters
def plot_clusters(data):
    for cluster_id in data['cluster'].unique():
        cluster_data = data[data['cluster'] == cluster_id]
        plt.scatter(cluster_data['atr'], cluster_data['rsi'], label=f'Cluster {cluster_id}')

    plt.xlabel('ATR')
    plt.ylabel('RSI')
    plt.legend()
    plt.show()

plot_clusters(data)
```

## Files
- `processed_crypto_data.csv`: Preprocessed data with indicators.
- `clustered_crypto_data.csv`: Data with assigned cluster labels.

## Results
Key observations:
- Cryptocurrencies with high ATR and RSI tend to exhibit volatile behavior.
- Clustering helps identify distinct market segments.

## License
This project is licensed under the MIT License.


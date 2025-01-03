import re
import threading
import time
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import yfinance as yf
from flask import Flask, request, jsonify
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)


def get_data(assets):
    df = pd.DataFrame()
    yf.pdr_override()
    end_date = datetime.now()
    for stock in assets:
        df[stock] = yf.download(stock, start='2014-01-01', end=end_date.strftime('%Y-%m-%d'), interval='1d')['Adj Close']
    return df


def monteCarlo(df, assets, num_of_portfolios=100):
    log_returns = np.log(1 + df.pct_change(fill_method=None))

    all_weights = np.zeros((num_of_portfolios, len(assets)))

    ret_arr = np.zeros(num_of_portfolios)
    vol_arr = np.zeros(num_of_portfolios)
    sharpe_arr = np.zeros(num_of_portfolios)

    for i in range(num_of_portfolios):
        monte_weights = np.random.random(len(assets))
        monte_weights = monte_weights / np.sum(monte_weights)

        all_weights[i, :] = monte_weights

        portfolio_return = np.sum((log_returns.mean() * monte_weights) * 252)
        portfolio_std_dev = np.sqrt(np.dot(monte_weights.T, np.dot(log_returns.cov() * 252, monte_weights)))

        ret_arr[i] = portfolio_return * 100
        vol_arr[i] = portfolio_std_dev
        sharpe_arr[i] = portfolio_return / portfolio_std_dev

    simulations_data = [ret_arr, vol_arr, sharpe_arr, all_weights]
    simulations_df = pd.DataFrame(data=simulations_data).T
    simulations_df.columns = ["Returns", "Volatility", "Sharpe Ratio", "Weights"]

    simulations_df = simulations_df.infer_objects()

    return simulations_df


@app.route('/portfolio', methods=['POST'])
def optimise_port():
    data = request.json
    ticker = str(data['Symbols'])
    ticker = remove_spaces(ticker)
    assets = ticker.split(',')
    df = get_data(assets)
    sim_no = data.get('sim_no', 1000)

    # Monte Carlo Simulation
    num_of_portfolios = sim_no
    sim_df = monteCarlo(df, assets, num_of_portfolios)
    sim_df['Volatility'] = sim_df['Volatility'].round(2)
    idx = sim_df.groupby('Volatility')['Returns'].idxmax()
    max_df = sim_df.loc[idx].reset_index(drop=True)
    max_df = max_df.sort_values(by='Volatility').reset_index(drop=True)
    max_df['Weights'] = max_df['Weights'].apply(lambda x: {col: weight for col, weight in zip(df.columns, x)})
    max_df = max_df.to_dict(orient='records')

    # Selecting the portfolio with the highest Sharpe Ratio
    max_returns = sim_df.loc[sim_df['Sharpe Ratio'].idxmax()]
    optimal_weights = max_returns['Weights']

    # Creating DataFrame for weights
    weights_df = pd.DataFrame(optimal_weights, index=df.columns, columns=['Weights'])
    weights_dict = weights_df.to_dict()

    # Expected annual return, volatility, and Sharpe ratio
    # mean_returns = df.pct_change(fill_method=None).mean()
    # print(mean_returns)
    # cov_matrix = df.pct_change(fill_method=None).cov() * 252
    # annual_return = np.sum(mean_returns * optimal_weights) * 252
    # print(annual_return)
    # port_variance = np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights))
    # port_volatility = np.sqrt(port_variance)
    # sharpe_ratio = (annual_return - 0.02) / port_volatility
    # "annual_return": annual_return * 100,
    # "port_volatility": port_volatility,
    # "sharpe_ratio": sharpe_ratio,
    return jsonify({
        "weights_df": weights_dict,
        "annual_return": max_returns['Returns'],
        "port_volatility": max_returns['Volatility']*100,
        "sharpe_ratio": max_returns['Sharpe Ratio'],
        "array_of_allocation": max_df
    })


def optimize_portfolio(df, assets):
    num_of_portfolios = 6000
    sim_df = monteCarlo(df, assets, num_of_portfolios)
    sim_df['Volatility'] = sim_df['Volatility'].round(2)
    idx = sim_df.groupby('Volatility')['Returns'].idxmax()
    max_df = sim_df.loc[idx].reset_index(drop=True)
    max_df = max_df.sort_values(by='Volatility').reset_index(drop=True)
    max_df['Weights'] = max_df['Weights'].apply(lambda x: {col: weight for col, weight in zip(df.columns, x)})
    max_df = max_df.to_dict(orient='records')

    # Selecting the portfolio with the highest Sharpe Ratio
    max_returns = sim_df.loc[sim_df['Sharpe Ratio'].idxmax()]
    optimal_weights = max_returns['Weights']

    # Expected annual return, volatility, and Sharpe ratio
    weights_df = pd.DataFrame(optimal_weights, columns=['Weights'])
    weights_df.index = assets  # Assign tickers as index

    # Expected annual return, volatility, and Sharpe ratio
    # mean_returns = df.pct_change(fill_method=None).mean()
    # cov_matrix = df.pct_change(fill_method=None).cov() * 252
    # annual_return = np.sum(mean_returns * optimal_weights) * 252
    # port_variance = np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights))
    # port_volatility = np.sqrt(port_variance)
    # sharpe_ratio = (annual_return - 0.02) / port_volatility
    # print("done")
    return weights_df, max_returns['Returns'], max_returns['Volatility']*100, max_returns['Sharpe Ratio'], max_df


def plot_chart(df, title):
    fig = px.line(df, title=title)
    return fig


def gaussian_pdf(x, mean, std):
    return (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)


def euclidean_distance(point, centroid):
    return np.sqrt(np.sum((point - centroid) ** 2))


def calculate_probabilities(user_point, centroids):
    distances = np.array([euclidean_distance(user_point, centroid) for centroid in centroids])
    std = np.std(distances)
    probabilities = np.array([gaussian_pdf(dist, 0, std) for dist in distances])
    normalized_probabilities = probabilities / np.sum(probabilities)
    return normalized_probabilities


def remove_spaces(text):
    return re.sub(r'\s+', '', text)


@app.route('/optimize', methods=['POST'])
def optimize():
    data = request.json

    lifestyle_risk = data['lifestyle_risk']
    expected_annual_roi = data['expected_annual_roi']
    principal_amount = data['principal_amount']

    info_data = [
        {"Symbol": "AAPL", "Annualized ROI": 17.624598, "Volatility": 0.027930},
        {"Symbol": "AMZN", "Annualized ROI": 30.874074, "Volatility": 0.035528},
        {"Symbol": "ARKK", "Annualized ROI": 7.464424, "Volatility": 0.023803},
        {"Symbol": "BABA", "Annualized ROI": -1.149219, "Volatility": 0.026288},
        {"Symbol": "BTC-USD", "Annualized ROI": 57.557349, "Volatility": 0.036763},
        {"Symbol": "ELF", "Annualized ROI": 21.738374, "Volatility": 0.031783},
        {"Symbol": "ETH-USD", "Annualized ROI": 35.917395, "Volatility": 0.046763},
        {"Symbol": "GC=F", "Annualized ROI": 8.950140, "Volatility": 0.010886},
        {"Symbol": "GOOGL", "Annualized ROI": 22.442833, "Volatility": 0.019344},
        {"Symbol": "GSK", "Annualized ROI": 10.011961, "Volatility": 0.017046},
        {"Symbol": "ITC.NS", "Annualized ROI": 16.263788, "Volatility": 0.020341},
        {"Symbol": "JNJ", "Annualized ROI": 10.925850, "Volatility": 0.014432},
        {"Symbol": "MSFT", "Annualized ROI": 24.020376, "Volatility": 0.021158},
        {"Symbol": "NFLX", "Annualized ROI": 31.412082, "Volatility": 0.035304},
        {"Symbol": "NVDA", "Annualized ROI": 34.711791, "Volatility": 0.037871},
        {"Symbol": "PLD", "Annualized ROI": 5.721513, "Volatility": 0.023028},
        {"Symbol": "QCOM", "Annualized ROI": 18.908380, "Volatility": 0.030630},
        {"Symbol": "SQ", "Annualized ROI": 17.814773, "Volatility": 0.036852},
        {"Symbol": "TCEHY", "Annualized ROI": 17.299087, "Volatility": 0.023695},
        {"Symbol": "TSLA", "Annualized ROI": 37.054781, "Volatility": 0.035839},
        {"Symbol": "XOM", "Annualized ROI": 7.051585, "Volatility": 0.014561}
    ]

    info = pd.DataFrame(info_data)
    del info_data
    model_data = info[['Annualized ROI', 'Volatility']]
    kmeans = KMeans(n_clusters=3, random_state=4)
    kmeans.fit(model_data)
    clusters = kmeans.predict(model_data)
    info['Cluster'] = clusters
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(model_data, clusters)
    centroids = kmeans.cluster_centers_

    if lifestyle_risk == 0:
        expected_volatility = centroids[0][1]
        x = "low risk"
    elif lifestyle_risk == 1:
        expected_volatility = centroids[2][1]
        x = "high risk"
    elif lifestyle_risk == 2:
        expected_volatility = centroids[1][1]
        x = "mid risk"
    else:
        return jsonify({"error": "Invalid lifestyle risk value"})

    model_input = np.array([[expected_annual_roi, expected_volatility]])
    predicted_cluster = knn.predict(model_input)

    probabilities = calculate_probabilities(model_input, centroids)
    nearest_centroid_index = np.argmax(probabilities)
    weighted_amounts = principal_amount * probabilities
    weights_df = pd.DataFrame({'Weight': weighted_amounts})

    clusters_data = {
        "Symbols": [
            "ARKK, BABA, GC=F, GSK, JNJ, PLD, XOM",
            "AMZN, BTC-USD, ETH-USD, NFLX, NVDA, TSLA",
            "AAPL, ELF, GOOGL, ITC.NS, MSFT, QCOM, SQ, TCEHY"
        ]
    }

    clusters_df = pd.DataFrame(clusters_data)
    clusters_df['Weights'] = weights_df['Weight']

    del info
    del knn
    del kmeans
    results = []
    for index, row in clusters_df.iterrows():
        ticker = str(row['Symbols'])
        ticker = remove_spaces(ticker)
        assets = ticker.split(',')
        df = get_data(assets)
        # print(df)
        # print(ticker)
        starting_amount = row['Weights']
        weights_allocated, annual_return, port_volatility, sharpe_ratio, max_df = optimize_portfolio(df, assets)
        # print("1")
        # print(weights_allocated)
        del df
        # print("hi this is max df",max_df)
        # print(weights_df)
        # print(annual_return)
        # print(port_volatility)
        # print("2")
        results.append({
            "Symbols": row['Symbols'],
            "Weights": weights_allocated.to_dict(),
            "Annual Return": annual_return,
            "Volatility": port_volatility,
            "Sharpe Ratio": sharpe_ratio,
            "Array_of_allocations": max_df
        })
        # print("3")
    # print(type(results))
    return jsonify({"type of person": x, "results": results, "clusters": clusters_df.to_dict(orient='records')})


@app.route('/weights', methods=['POST'])
def weights():
    data = request.json

    lifestyle_risk = data['lifestyle_risk']
    expected_annual_roi = data['expected_annual_roi']
    principal_amount = data['principal_amount']

    info_data = [
        {"Symbol": "AAPL", "Annualized ROI": 17.624598, "Volatility": 0.027930},
        {"Symbol": "AMZN", "Annualized ROI": 30.874074, "Volatility": 0.035528},
        {"Symbol": "ARKK", "Annualized ROI": 7.464424, "Volatility": 0.023803},
        {"Symbol": "BABA", "Annualized ROI": -1.149219, "Volatility": 0.026288},
        {"Symbol": "BTC-USD", "Annualized ROI": 57.557349, "Volatility": 0.036763},
        {"Symbol": "ELF", "Annualized ROI": 21.738374, "Volatility": 0.031783},
        {"Symbol": "ETH-USD", "Annualized ROI": 35.917395, "Volatility": 0.046763},
        {"Symbol": "GC=F", "Annualized ROI": 8.950140, "Volatility": 0.010886},
        {"Symbol": "GOOGL", "Annualized ROI": 22.442833, "Volatility": 0.019344},
        {"Symbol": "GSK", "Annualized ROI": 10.011961, "Volatility": 0.017046},
        {"Symbol": "ITC.NS", "Annualized ROI": 16.263788, "Volatility": 0.020341},
        {"Symbol": "JNJ", "Annualized ROI": 10.925850, "Volatility": 0.014432},
        {"Symbol": "MSFT", "Annualized ROI": 24.020376, "Volatility": 0.021158},
        {"Symbol": "NFLX", "Annualized ROI": 31.412082, "Volatility": 0.035304},
        {"Symbol": "NVDA", "Annualized ROI": 34.711791, "Volatility": 0.037871},
        {"Symbol": "PLD", "Annualized ROI": 5.721513, "Volatility": 0.023028},
        {"Symbol": "QCOM", "Annualized ROI": 18.908380, "Volatility": 0.030630},
        {"Symbol": "SQ", "Annualized ROI": 17.814773, "Volatility": 0.036852},
        {"Symbol": "TCEHY", "Annualized ROI": 17.299087, "Volatility": 0.023695},
        {"Symbol": "TSLA", "Annualized ROI": 37.054781, "Volatility": 0.035839},
        {"Symbol": "XOM", "Annualized ROI": 7.051585, "Volatility": 0.014561}
    ]

    info = pd.DataFrame(info_data)

    model_data = info[['Annualized ROI', 'Volatility']]
    kmeans = KMeans(n_clusters=3, random_state=4)
    kmeans.fit(model_data)
    clusters = kmeans.predict(model_data)
    info['Cluster'] = clusters
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(model_data, clusters)
    centroids = kmeans.cluster_centers_
    if lifestyle_risk == 0:
        expected_volatility = centroids[0][1]
        x = "low risk"
    elif lifestyle_risk == 1:
        expected_volatility = centroids[2][1]
        x = "high risk"
    elif lifestyle_risk == 2:
        expected_volatility = centroids[1][1]
        x = "mid risk"
    else:
        return jsonify({"error": "Invalid lifestyle risk value"})

    model_input = np.array([[expected_annual_roi, expected_volatility]])

    probabilities = calculate_probabilities(model_input, centroids)
    weighted_amounts = principal_amount * probabilities
    weights_df = pd.DataFrame({'Weight': weighted_amounts})

    clusters_data = {
        "Symbols": [
            "ARKK, BABA, GC=F, GSK, JNJ, PLD, XOM",
            "AMZN, BTC-USD, ETH-USD, NFLX, NVDA, TSLA",
            "AAPL, ELF, GOOGL, ITC.NS, MSFT, QCOM, SQ, TCEHY"
        ]
    }

    clusters_df = pd.DataFrame(clusters_data)
    clusters_df['Weights'] = weights_df['Weight']
    return jsonify({"type of person": x,"clusters": clusters_df.to_dict(orient='records')})


@app.route('/heartbeat', methods=['GET'])
def heartbeat():
    return jsonify({"status": "alive"}), 200


def send_heartbeat():
    while True:
        try:
            requests.get("https://montecarloapi.onrender.com/heartbeat")
        except requests.exceptions.RequestException as e:
            print(f"Heartbeat failed: {e}")
        time.sleep(300)  # Send a heartbeat request every 5 minutes


if __name__ == '__main__':
    # Start the heartbeat thread
    heartbeat_thread = threading.Thread(target=send_heartbeat)
    heartbeat_thread.daemon = True
    heartbeat_thread.start()

    # Run the Flask application
    app.run(debug=True)

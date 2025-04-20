
import streamlit as st
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy.optimize import brentq
from io import BytesIO

def black_scholes_call_put(S, K, T, r, sigma):
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    call = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    put = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return call, put, d1, d2

def option_greeks(S, K, T, r, sigma, d1, d2):
    delta_call = norm.cdf(d1)
    delta_put = norm.cdf(d1) - 1
    gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
    vega = S * norm.pdf(d1) * math.sqrt(T)
    theta_call = (-S * norm.pdf(d1) * sigma / (2 * math.sqrt(T))) - r * K * math.exp(-r * T) * norm.cdf(d2)
    theta_put = (-S * norm.pdf(d1) * sigma / (2 * math.sqrt(T))) + r * K * math.exp(-r * T) * norm.cdf(-d2)
    rho_call = K * T * math.exp(-r * T) * norm.cdf(d2)
    rho_put = -K * T * math.exp(-r * T) * norm.cdf(-d2)
    return delta_call, delta_put, gamma, vega, theta_call, theta_put, rho_call, rho_put

def implied_volatility(option_price, S, K, T, r, option_type='call'):
    def obj_func(sigma):
        call, put, _, _ = black_scholes_call_put(S, K, T, r, sigma)
        return (call - option_price) if option_type == 'call' else (put - option_price)
    try:
        return brentq(obj_func, 1e-6, 5.0)
    except ValueError:
        return None

def monte_carlo_pnl(S, K, T, r, sigma, option_type='call', sims=10000):
    Z = np.random.normal(0, 1, sims)
    ST = S * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)
    if option_type == 'call':
        payoff = np.maximum(ST - K, 0)
    else:
        payoff = np.maximum(K - ST, 0)
    pnl = payoff - np.mean(payoff)
    return pnl

def generate_excel(data):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        data.to_excel(writer, index=False, sheet_name='Option Data')
    output.seek(0)
    return output

# === Streamlit UI ===
st.title("📊 Black-Scholes Quant Dashboard")

S = st.number_input("Stock Price (S)", value=100.0)
K = st.number_input("Strike Price (K)", value=100.0)
T = st.number_input("Time to Maturity (T in years)", value=1.0)
r = st.number_input("Risk-Free Rate (r)", value=0.05)
sigma = st.number_input("Volatility (σ)", value=0.2)

if st.button("Run Analysis"):
    call, put, d1, d2 = black_scholes_call_put(S, K, T, r, sigma)
    delta_call, delta_put, gamma, vega, theta_call, theta_put, rho_call, rho_put = option_greeks(S, K, T, r, sigma, d1, d2)

    st.success(f"Call Price: R {call:.2f} | Put Price: R {put:.2f}")

    st.subheader("🧠 Greeks")
    st.write(pd.DataFrame({
        "Greek": ["Delta (Call)", "Delta (Put)", "Gamma", "Vega", "Theta (Call)", "Theta (Put)", "Rho (Call)", "Rho (Put)"],
        "Value": [delta_call, delta_put, gamma, vega, theta_call, theta_put, rho_call, rho_put]
    }))

    st.subheader("📈 P&L Profile with Breakeven")
    prices = np.linspace(0.5 * S, 1.5 * S, 100)
    call_pnl = np.maximum(prices - K, 0) - call
    put_pnl = np.maximum(K - prices, 0) - put

    fig, ax = plt.subplots()
    ax.plot(prices, call_pnl, label='Call P&L', color='green')
    ax.plot(prices, put_pnl, label='Put P&L', color='red')
    ax.axhline(0, linestyle='--', color='gray')
    ax.legend()
    st.pyplot(fig)

    st.subheader("🧠 Implied Volatility Solver")
    mkt_price = st.number_input("Market Option Price", value=call)
    opt_type = st.selectbox("Option Type", ["call", "put"])
    iv = implied_volatility(mkt_price, S, K, T, r, opt_type)
    st.info(f"Implied Volatility: {iv:.4f}" if iv else "No solution found")

    st.subheader("🛠️ Strategy Builder: Straddle & Bull Call Spread")
    straddle = np.maximum(prices - K, 0) + np.maximum(K - prices, 0) - (call + put)
    spread = np.maximum(prices - K, 0) - np.maximum(prices - (K + 10), 0) - (call - black_scholes_call_put(S, K+10, T, r, sigma)[0])

    fig2, ax2 = plt.subplots()
    ax2.plot(prices, straddle, label='Straddle', color='purple')
    ax2.plot(prices, spread, label='Bull Call Spread', color='blue')
    ax2.axhline(0, linestyle='--', color='gray')
    ax2.legend()
    st.pyplot(fig2)

    st.subheader("🎲 Monte Carlo P&L Simulation")
    mc = monte_carlo_pnl(S, K, T, r, sigma, opt_type)
    fig3, ax3 = plt.subplots()
    ax3.hist(mc, bins=50, color='skyblue', edgecolor='black')
    ax3.set_title("Monte Carlo P&L")
    st.pyplot(fig3)

    st.subheader("🔳 Heatmaps for Call/Put vs Volatility & Strike")
    vol_range = np.linspace(0.05, 1.0, 20)
    strike_range = np.linspace(0.5 * S, 1.5 * S, 20)
    call_matrix = np.zeros((len(vol_range), len(strike_range)))
    put_matrix = np.zeros((len(vol_range), len(strike_range)))

    for i, vol in enumerate(vol_range):
        for j, strike in enumerate(strike_range):
            c, p, _, _ = black_scholes_call_put(S, strike, T, r, vol)
            call_matrix[i, j] = c
            put_matrix[i, j] = p

    fig4, ax4 = plt.subplots(figsize=(10, 6))
    sns.heatmap(call_matrix, xticklabels=np.round(strike_range, 1), yticklabels=np.round(vol_range, 2), ax=ax4, cmap='YlGnBu')
    ax4.set_title("Call Price Heatmap")
    st.pyplot(fig4)

    fig5, ax5 = plt.subplots(figsize=(10, 6))
    sns.heatmap(put_matrix, xticklabels=np.round(strike_range, 1), yticklabels=np.round(vol_range, 2), ax=ax5, cmap='YlOrRd')
    ax5.set_title("Put Price Heatmap")
    st.pyplot(fig5)

    st.subheader("📤 Export to Excel")
    df_export = pd.DataFrame({
        "Call Price": [call], "Put Price": [put],
        "Implied Volatility": [iv if iv else "N/A"],
        "Delta": [delta_call], "Gamma": [gamma], "Vega": [vega],
        "Theta": [theta_call], "Rho": [rho_call]
    })
    st.download_button("Download Results", generate_excel(df_export), file_name="option_results.xlsx")

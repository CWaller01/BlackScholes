import streamlit as st
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from io import BytesIO

def black_scholes_call_put(S, K, T, r, sigma):
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
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

def generate_excel(data):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        data.to_excel(writer, index=False, sheet_name='Option Data')
    output.seek(0)
    return output

st.title("🧮 Black-Scholes Dashboard")

S = st.number_input("Stock Price (S)", value=100.0)
K = st.number_input("Strike Price (K)", value=100.0)
T = st.number_input("Time to Maturity (T in years)", value=1.0, step=0.01)
r = st.number_input("Risk-Free Rate (r)", value=0.05, step=0.01)
sigma = st.number_input("Volatility (σ)", value=0.2, step=0.01)

if st.button("Calculate and Show Results"):
    call, put, d1, d2 = black_scholes_call_put(S, K, T, r, sigma)
    delta_call, delta_put, gamma, vega, theta_call, theta_put, rho_call, rho_put = option_greeks(S, K, T, r, sigma, d1, d2)

    st.success(f"Call Price: R {call:.4f}  |  Put Price: R {put:.4f}")

    st.subheader("📈 Option Greeks")
    st.write(pd.DataFrame({
        'Greek': ['Delta (Call)', 'Delta (Put)', 'Gamma', 'Vega', 'Theta (Call)', 'Theta (Put)', 'Rho (Call)', 'Rho (Put)'],
        'Value': [delta_call, delta_put, gamma, vega, theta_call, theta_put, rho_call, rho_put]
    }))

    st.subheader("📊 Profit/Loss Profile")
    prices = np.linspace(0.5 * S, 1.5 * S, 100)
    call_pnl = np.maximum(prices - K, 0) - call
    put_pnl = np.maximum(K - prices, 0) - put

    fig, ax = plt.subplots()
    ax.plot(prices, call_pnl, label='Call P&L', color='green')
    ax.plot(prices, put_pnl, label='Put P&L', color='red')
    ax.axhline(0, linestyle='--', color='gray')
    ax.set_xlabel("Underlying Price at Expiry")
    ax.set_ylabel("Profit / Loss")
    ax.set_title("Option Profit/Loss at Expiry")
    ax.legend()
    st.pyplot(fig)

    st.subheader("📤 Download Results")
    result_df = pd.DataFrame({
        'Stock Price': [S], 'Strike Price': [K], 'Time to Maturity': [T], 'Risk-Free Rate': [r], 'Volatility': [sigma],
        'Call Price': [call], 'Put Price': [put], 'Delta (Call)': [delta_call], 'Delta (Put)': [delta_put],
        'Gamma': [gamma], 'Vega': [vega], 'Theta (Call)': [theta_call], 'Theta (Put)': [theta_put],
        'Rho (Call)': [rho_call], 'Rho (Put)': [rho_put]
    })
    excel_data = generate_excel(result_df)
    st.download_button(label="Download Excel", data=excel_data, file_name="option_pricing_results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


import streamlit as st
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy.optimize import brentq
from io import BytesIO

# Set Streamlit dark theme and branding
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto+Slab:wght@400;700&display=swap');

        html, body, [class*="css"]  {
            font-family: 'Roboto Slab', serif;
            background-color: #1e1e1e;
            color: white;
        }
        .stButton>button {
            color: white;
            background-color: #4b4b4b;
        }
        .stTextInput>div>input, .stNumberInput>div>input {
            background-color: #333;
            color: white;
        }
        .stDataFrame thead tr th {
            background-color: #333 !important;
            color: white !important;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #f0f0f0 !important;
        }
    </style>
""", unsafe_allow_html=True)

# ... existing logic remains unchanged ...

    # Call and Put price display in separate blocks
    col1, col2 = st.columns(2)
    with col1:
        st.metric("📈 Call Price", f"R {call:.2f}")
    with col2:
        st.metric("📉 Put Price", f"R {put:.2f}")

    # P&L Plot
    st.subheader("📈 P&L Profile with Breakeven")
    fig, ax = plt.subplots()
    ax.plot(prices, call_pnl, label='Call P&L', color='green')
    ax.plot(prices, put_pnl, label='Put P&L', color='red')
    ax.axhline(0, linestyle='--', color='gray')
    ax.legend()
    st.pyplot(fig)
    st.caption("The graph shows profit/loss for call and put options at expiry across price scenarios.")

    # Strategy builder
    st.subheader("🛠️ Strategy Builder: Straddle & Bull Call Spread")
    fig2, ax2 = plt.subplots()
    ax2.plot(prices, straddle, label='Straddle', color='purple')
    ax2.plot(prices, spread, label='Bull Call Spread', color='blue')
    ax2.axhline(0, linestyle='--', color='gray')
    ax2.legend()
    st.pyplot(fig2)
    st.caption("This plot compares the P&L for a straddle and bull call spread strategy.")

    # Monte Carlo
    st.subheader("🎲 Monte Carlo P&L Simulation")
    mc = monte_carlo_pnl(S, K, T, r, sigma, opt_type)
    fig3, ax3 = plt.subplots()
    ax3.hist(mc, bins=50, color='skyblue', edgecolor='black')
    ax3.set_title("Monte Carlo P&L")
    st.pyplot(fig3)
    st.caption("Distribution of simulated P&L outcomes using Monte Carlo simulation.")

    # Heatmaps
    st.subheader("🔳 Heatmaps for Call/Put vs Volatility & Strike")
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    sns.heatmap(call_matrix, xticklabels=np.round(strike_range, 1), yticklabels=np.round(vol_range, 2), ax=ax4, cmap='YlGnBu')
    ax4.set_title("Call Price Heatmap")
    st.pyplot(fig4)
    st.caption("Call option values across different strike prices and volatilities.")

    fig5, ax5 = plt.subplots(figsize=(10, 6))
    sns.heatmap(put_matrix, xticklabels=np.round(strike_range, 1), yticklabels=np.round(vol_range, 2), ax=ax5, cmap='YlOrRd')
    ax5.set_title("Put Price Heatmap")
    st.pyplot(fig5)
    st.caption("Put option values across different strike prices and volatilities.")

    # Modern styled download panel
    with st.expander("📤 Download Results"):
        df_export = pd.DataFrame({
            "Call Price": [call], "Put Price": [put],
            "Implied Volatility": [iv if iv else "N/A"],
            "Delta": [delta_call], "Gamma": [gamma], "Vega": [vega],
            "Theta": [theta_call], "Rho": [rho_call]
        })
        st.download_button(
            label="💾 Save to Excel",
            data=generate_excel(df_export),
            file_name="option_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

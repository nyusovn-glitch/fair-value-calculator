import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta


st.set_page_config(page_title="Fair Value Calculator", layout="wide")


st.markdown("""
<style>
    .metric-box {
        border: 1px solid #333;
        border-radius: 5px;
        padding: 10px;
        text-align: center;
        background-color: #1e1e1e;
        color: white;
    }
    .big-font { font-size: 24px !important; font-weight: bold; }
    .green-text { color: #4caf50; }
    .red-text { color: #f44336; }
    .historic-growth {
        font-size: 11px;
        color: #888;
        margin-top: -8px;
        margin-bottom: 12px;
    }
    .stButton>button {
        width: 100%;
        background-color: #5cb85c;
        color: white;
        font-weight: bold;
    }
    .breakdown-step {
        background-color: #1a1a1a;
        border-left: 3px solid #5cb85c;
        padding: 10px 12px;
        margin-bottom: 12px;
        border-radius: 4px;
        font-size: 14px;
    }
    .breakdown-step h4 {
        color: #5cb85c;
        margin: 0 0 8px 0;
        font-size: 16px;
    }
    .breakdown-step p {
        margin: 4px 0;
        line-height: 1.4;
    }
</style>
""", unsafe_allow_html=True)


def calculate_cagr(end_value, start_value, periods):
    if start_value == 0 or periods <= 0:
        return 0
    if start_value < 0 or end_value < 0:
        return 0
    return (end_value / start_value) ** (1 / periods) - 1


# Compact dialog with 2-column layout
@st.dialog("Fair Value Calculation Breakdown", width="large")
def show_breakdown(metric_label, current_val, growth_rate, years, final_metric,
                   current_shares, shares_growth, final_shares,
                   metric_per_share_final, terminal_multiple, future_price,
                   discount_rate, fair_value, multiple_name):

    # Create 2 columns for compact layout
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div class='breakdown-step'>
            <h4>Step 1: Project {metric_label}</h4>
            <p>Most recent {metric_label}: <strong>${current_val:,.0f}M</strong></p>
            <p>Growth rate: <strong>{growth_rate*100:.1f}%</strong> over <strong>{years} years</strong></p>
            <p>Future {metric_label}: <strong>${final_metric:,.0f}M</strong></p>
        </div>
        
        <div class='breakdown-step'>
            <h4>Step 2: Project Shares</h4>
            <p>Current shares: <strong>{current_shares:,.0f}M</strong></p>
            <p>Growth rate: <strong>{shares_growth*100:.1f}%</strong> over <strong>{years} years</strong></p>
            <p>Future shares: <strong>{final_shares:,.0f}M</strong></p>
        </div>
        
        <div class='breakdown-step'>
            <h4>Step 3: {metric_label} Per Share</h4>
            <p><strong>{metric_label} Per Share = {metric_label} / Shares</strong></p>
            <p>${final_metric:,.0f}M / {final_shares:,.0f}M = <strong>${metric_per_share_final:,.2f}</strong></p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class='breakdown-step'>
            <h4>Step 4: Future Stock Price</h4>
            <p>{metric_label} per share: <strong>${metric_per_share_final:,.2f}</strong></p>
            <p>{multiple_name}: <strong>{terminal_multiple:.1f}x</strong></p>
            <p><strong>Future Price = ${metric_per_share_final:,.2f} × {terminal_multiple:.1f} = ${future_price:,.2f}</strong></p>
        </div>
        
        <div class='breakdown-step'>
            <h4>Step 5: Discount to Today</h4>
            <p>Future price: <strong>${future_price:,.2f}</strong> in {years} years</p>
            <p>Discount rate: <strong>{discount_rate*100:.1f}%</strong></p>
            <p><strong>Fair Value = ${future_price:,.2f} / (1.{int(discount_rate*100):02d})^{years}</strong></p>
            <p><strong>= ${fair_value:,.2f}</strong></p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Close", use_container_width=True, type="primary"):
        st.rerun()


main_col, input_col = st.columns([3, 1])

with input_col:
    st.header("DCF Inputs")
    ticker_input = st.text_input("Ticker Symbol", value="GOOG").upper()

    if st.button("Fetch Data"):
        try:
            stock = yf.Ticker(ticker_input)
            info = stock.info

            st.session_state['ticker'] = ticker_input
            st.session_state['price'] = info.get('currentPrice', 150.0)
            st.session_state['shares'] = info.get('sharesOutstanding', 5000000000) / 1000000

            financials = stock.financials.T.sort_index()
            cashflow = stock.cashflow.T.sort_index()
            combined = financials.join(cashflow, lsuffix='_inc', rsuffix='_cf', how='inner')

            st.session_state['financials'] = combined
            st.session_state['data_fetched'] = True

        except Exception as e:
            st.error(f"Error fetching data: {e}")

    metric_type = st.radio("Valuation Metric", ["Free Cash Flow (FCF)", "Earnings (Net Income)"])
    metric_label = "FCF" if "Cash" in metric_type else "Earnings"

    default_val = 60000.0
    default_shares = 5000.0
    default_price = 150.0
    growth_rates = {"1y": 0, "3y": 0}

    if st.session_state.get('data_fetched', False):
        try:
            default_price = st.session_state.get('price', 150.0)
            default_shares = st.session_state.get('shares', 5000.0)
            combined = st.session_state.get('financials')

            target_col = None

            if metric_label == "Earnings":
                for col_name in ["Net Income", "Net Income Applicable to Common Shares", "Net Income Common Stockholders"]:
                    if col_name in combined.columns:
                        target_col = col_name
                        break
            else:
                if "Free Cash Flow" in combined.columns:
                    target_col = "Free Cash Flow"
                elif "Operating Cash Flow" in combined.columns and "Capital Expenditure" in combined.columns:
                    combined["Free Cash Flow"] = combined["Operating Cash Flow"] + combined["Capital Expenditure"]
                    target_col = "Free Cash Flow"

            if target_col and target_col in combined.columns:
                data_series = combined[target_col].dropna()
                if len(data_series) > 0:
                    current_val_calc = data_series.iloc[-1] / 1000000
                    default_val = current_val_calc

                    years_avail = len(data_series)
                    if years_avail >= 2:
                        growth_rates["1y"] = calculate_cagr(data_series.iloc[-1], data_series.iloc[-2], 1)
                    if years_avail >= 4:
                        growth_rates["3y"] = calculate_cagr(data_series.iloc[-1], data_series.iloc[-4], 3)

        except Exception as e:
            st.error(f"Error processing data: {e}")

    st.subheader("Assumptions")

    years_to_project = st.selectbox("Number of Years To Project", [5, 10], index=0)

    def format_growth_inline(val):
        color = "#4caf50" if val > 0 else "#f44336"
        return f"<span style='color:{color}'>{val:.1%}</span>"

    growth_rate = st.number_input(f"{metric_label} Growth Rate (%)", value=12.0, step=0.5) / 100
    historic_text = f"Historic: 1Y: {format_growth_inline(growth_rates['1y'])} | 3Y: {format_growth_inline(growth_rates['3y'])}"
    st.markdown(f"<div class='historic-growth'>{historic_text}</div>", unsafe_allow_html=True)

    shares_growth = st.number_input("Shares Change Rate (%)", value=-3.0, step=0.1,
                                    help="Negative = Buybacks") / 100

    multiple_name = "Price to FCF" if metric_label == "FCF" else "P/E Ratio"
    terminal_multiple = st.number_input(f"{multiple_name} (Terminal)", value=20.0, step=1.0)

    discount_rate = st.number_input("Discount Rate (WACC) %", value=10.0, step=0.5) / 100

    st.markdown("---")
    st.caption("Advanced Overrides")
    current_val_override = st.number_input(f"Starting {metric_label} ($M)", value=float(default_val))
    current_shares_override = st.number_input("Shares Outstanding (M)", value=float(default_shares))


def calculate_dcf_per_share():
    future_years = list(range(1, years_to_project + 1))
    val_projections = []
    curr_val = current_val_override
    for _ in future_years:
        curr_val = curr_val * (1 + growth_rate)
        val_projections.append(curr_val)

    share_counts = []
    curr_shares = current_shares_override
    for _ in future_years:
        curr_shares = curr_shares * (1 + shares_growth)
        share_counts.append(curr_shares)

    fcf_per_share = [val / shares for val, shares in zip(val_projections, share_counts)]
    terminal_val_per_share = fcf_per_share[-1] * terminal_multiple
    discount_factors = [(1 + discount_rate) ** y for y in future_years]
    pv_fcf_per_share = [f / d for f, d in zip(fcf_per_share, discount_factors)]
    pv_terminal_per_share = terminal_val_per_share / ((1 + discount_rate) ** years_to_project)
    fair_value_ps = sum(pv_fcf_per_share) + pv_terminal_per_share

    return fair_value_ps, val_projections, share_counts


fair_value, proj_vals, proj_shares = calculate_dcf_per_share()

final_metric_value = proj_vals[-1]
final_shares = proj_shares[-1]
metric_per_share_final = final_metric_value / final_shares
future_price = metric_per_share_final * terminal_multiple

with main_col:
    st.title(f"Fair Value Calculator: {ticker_input}")
    st.caption(f"Valuation based on: **{metric_type}**")

    m1, m2, m3, m4 = st.columns(4)
    upside = (fair_value - default_price) / default_price
    with m1:
        st.markdown(
            f"<div class='metric-box'>Stock Price<br><span class='big-font'>${default_price:,.2f}</span></div>",
            unsafe_allow_html=True)
    with m2:
        st.markdown(
            f"<div class='metric-box'>Fair Value<br><span class='big-font'>${fair_value:,.2f}</span></div>",
            unsafe_allow_html=True)
    with m3:
        color = "green-text" if fair_value > default_price else "red-text"
        st.markdown(
            f"<div class='metric-box'>Upside / Downside<br><span class='big-font {color}'>{upside:.1%}</span></div>",
            unsafe_allow_html=True)
    with m4:
        st.markdown(
            f"<div class='metric-box'>Terminal {multiple_name}<br><span class='big-font'>{terminal_multiple}x</span></div>",
            unsafe_allow_html=True)

    st.write("")

    if st.button("Show Calculation Breakdown", use_container_width=True):
        show_breakdown(
            metric_label, current_val_override, growth_rate, years_to_project,
            final_metric_value, current_shares_override, shares_growth, final_shares,
            metric_per_share_final, terminal_multiple, future_price,
            discount_rate, fair_value, multiple_name
        )

    # SAFE HISTORY FETCH
    try:
        hist_data = yf.Ticker(ticker_input).history(period="2y")
        if hist_data is not None and not hist_data.empty:
            hist_dates = hist_data.index
            hist_prices = hist_data["Close"]
        else:
            raise ValueError("No history returned")
    except Exception:
        hist_dates = pd.date_range(end=datetime.today(), periods=100)
        hist_prices = [100] * 100

    last_date = hist_dates[-1]
    future_dates = [last_date + timedelta(days=365 * i / 12) for i in range(years_to_project * 12)]
    fair_value_curve = [fair_value * ((1 + discount_rate) ** (i / 12)) for i in range(len(future_dates))]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_dates, y=hist_prices, mode='lines', name='Stock Price',
                             line=dict(color='white', width=2),
                             fill='tozeroy', fillcolor='rgba(255,255,255,0.1)'))
    fig.add_trace(go.Scatter(x=future_dates, y=fair_value_curve, mode='lines',
                             name='Fair Value Projection',
                             line=dict(color='#ff4b4b', width=2, dash='dash')))
    fig.add_trace(go.Scatter(x=[last_date], y=[fair_value], mode='markers+text',
                             marker=dict(color='#ff4b4b', size=12),
                             text=[f"${fair_value:.0f}"],
                             textposition="top center",
                             name="Fair Value Today"))

    fig.update_layout(template="plotly_dark", title=f"Price vs. Fair Value",
                      hovermode="x unified", yaxis_title="Price ($)",
                      height=500, legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Show Calculation Data"):
        df_proj = pd.DataFrame({
            "Year": range(1, years_to_project + 1),
            f"Total {metric_label} ($M)": proj_vals,
            "Share Count (M)": proj_shares,
            f"{metric_label} Per Share": [v / s for v, s in zip(proj_vals, proj_shares)]
        })
        st.dataframe(df_proj.style.format("{:,.2f}"))

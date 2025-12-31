# =========================================================
# MacroCycle Lab — Streamlit App (Single-file, paste-ready)
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# =========================================================
# 1) App Config & Constants
# =========================================================
st.set_page_config(page_title="MacroCycle Lab", layout="wide")

INITIAL_CAPITAL = 100_000_000
TOTAL_MONTHS = 120

TARGET_INFLATION = 2.0
POTENTIAL_GROWTH = 2.5
NEUTRAL_RATE = 2.5
BASE_PE = 20.0

# Natural unemployment (game baseline)
U_STAR = 4.0

# Surprise standardization sigmas (tuning params)
SIG_GDP = 0.25
SIG_CPI = 0.25
SIG_PMI = 1.20

# Stock index soft cap (runaway prevention)
STOCK_SOFT_CAP = 700.0


# =========================================================
# 2) Header (Always Render)
# =========================================================
st.markdown("# MacroCycle Lab")
st.caption("An interactive global macro learning simulator — regimes, rates, risk, and surprises.")
st.markdown("---")


# =========================================================
# 3) Session State Init (Reset Counter & History)
# =========================================================
if "reset_count" not in st.session_state:
    st.session_state.reset_count = 0

if "month" not in st.session_state:
    st.session_state.month = 0
    st.session_state.my_balance = INITIAL_CAPITAL
    st.session_state.bm_balance = INITIAL_CAPITAL

    st.session_state.history = pd.DataFrame(
        columns=[
            "Month", "My Wealth", "BM Wealth", "Regime",
            "GDP", "CPI", "PMI", "BaseRate",
            "StockIdx", "PE", "EPS", "Yield10Y", "Yield2Y",
            "Unemployment", "BEI", "Spread",
            # NEW
            "RiskScore", "RiskLabel", "zGDP", "zCPI", "zPMI",
        ]
    )

    st.session_state.last_returns = {k: 0.0 for k in ["주식", "국채2년", "국채10년", "원유", "금", "현금"]}
    st.session_state.last_news = "🧪 MacroCycle Lab 시작! PMI·CPI·금리·스프레드를 보며 국면과 리스크를 읽어보세요."


# =========================================================
# 4) UI Styling (CSS)
# =========================================================
st.markdown(
    """
    <style>
        .block-container { padding: 1rem 2rem; }
        section[data-testid="stSidebar"] > div { padding-top: 1rem; }
        section[data-testid="stSidebar"] h3 { margin-top: 0rem !important; margin-bottom: 1rem !important; }

        .metric-card {
            background-color: white;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 15px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        .metric-title { font-size: 0.85rem; color: #666; font-weight: 600; margin-bottom: 5px; }
        .metric-value { font-size: 1.4rem; font-weight: 800; color: #333; }
        .metric-delta { font-size: 0.85rem; font-weight: 500; }
        .metric-sub { font-size: 0.75rem; color: #999; margin-top: 5px; }

        .fin-card {
            background-color: #f8faff;
            border: 1px solid #d0d7de;
            border-radius: 8px;
            padding: 10px;
            text-align: center;
            height: 100%;
        }
        .fin-title { font-size: 0.75rem; color: #555; font-weight: 700; margin-bottom: 3px; }
        .fin-value { font-size: 1.2rem; font-weight: 800; color: #2962FF; }

        .desc-box {
            background-color: #f8f9fa;
            border-left: 4px solid #2962FF;
            padding: 10px 15px;
            margin-bottom: 10px;
            font-size: 0.9rem;
            color: #444;
            height: 100%;
        }

        section[data-testid="stSidebar"] div.stButton > button {
            min-height: 50px !important;
            height: 50px !important;
            line-height: 1.0 !important;
            border-radius: 6px;
            font-weight: bold;
            width: 100%;
            margin-top: 0px !important;
            padding: 0px !important;
        }

        div[data-testid="column"]:nth-of-type(1) div.stButton > button {
            background-color: #2962FF; color: white; border: none;
        }
        div[data-testid="column"]:nth-of-type(1) div.stButton > button:hover {
            background-color: #0039CB;
        }

        div[data-testid="column"]:nth-of-type(2) div.stButton > button {
            background-color: #ffffff; color: #d32f2f; border: 1px solid #d32f2f;
        }
        div[data-testid="column"]:nth-of-type(2) div.stButton > button:hover {
            background-color: #ffebee;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 5) Economic Engine
# =========================================================
class EconomicEngineV21:
    """
    기존 EconomicEngineV21(사용자 제공) 기반.
    [NEW 블록]만 추가/최소 교체로 다음을 반영:
    1) Risk Score 공개(내부→UI)
    2) Surprise 표준화(z-score)로 거시 반응 스케일 안정화
    3) 정책 관성(Policy Inertia) 추가
    4) 10Y: Flight-to-Quality(침체 시 더 내려가게)
    5) 주식 변동성: risk_score에 연동(평시 안정/위기 폭발)
    6) 원유/금: 장기 드리프트 소량 추가
    7) 주가지수: 소프트 캡(폭주 방지)
    8) PMI 현실화(상단 억제): 평균회귀 + 고레벨 변동성 축소 + 클립(이미 반영)
    """

    def __init__(self):
        self.regime = "회복 (Recovery)"
        self.gdp_growth = 2.0
        self.pmi = 50.0
        self.unemployment = 4.0
        self.cpi = 2.0
        self.base_rate = 1.5
        self.cpi_history = [2.0, 2.0, 2.0]

        self.eps = 100.0
        self.pe_ratio = 20.0
        self.stock_index = 100.0
        self.oil_index = 100.0
        self.gold_index = 100.0

        self.yield_2y = 1.6
        self.yield_10y = 2.5
        self.spread = 0.9

        self.bei = 2.0

        self.surprises = {k: 0.0 for k in ["GDP", "CPI", "PMI"]}
        self.delta_rate = 0.0
        self.delta_cpi = 0.0
        self.delta_pmi = 0.0

        # Index normalization base
        self.base_eps0 = self.eps
        self.base_pe0 = self.pe_ratio
        self.base_index_level = 100.0

        # Policy inertia
        self.last_policy_dir = 0  # -1 cut, 0 neutral, +1 hike

        # Risk/Surprise diagnostics (UI)
        self.risk_score = 0.0
        self.risk_label = "LOW"
        self.z_gdp = 0.0
        self.z_cpi = 0.0
        self.z_pmi = 0.0

        # Risk parts (UI explain)
        self.risk_parts = {"PMI": 0.0, "Spread": 0.0, "UnempJump": 0.0, "CPI": 0.0}

    def next_turn(self):
        prev_stock = self.stock_index
        prev_oil = self.oil_index
        prev_gold = self.gold_index

        prev_yield_2y = self.yield_2y
        prev_yield_10y = self.yield_10y
        prev_rate = self.base_rate

        prev_cpi = self.cpi
        prev_pmi = self.pmi
        prev_unemp = self.unemployment
        prev_bei = self.bei
        prev_real_yield = prev_yield_10y - prev_bei

        # -------------------------------------------------
        # 1) Macro Economy
        # -------------------------------------------------
        consensus_gdp = self.gdp_growth + np.random.normal(0, 0.1)
        consensus_cpi = self.cpi + np.random.normal(0, 0.1)
        consensus_pmi = self.pmi + np.random.normal(0, 0.6)

        # Regime Logic
        prob = np.random.rand()
        if self.regime == "회복 (Recovery)":
            self.pmi += 0.8
            self.gdp_growth += 0.1
            self.unemployment -= 0.15
            self.cpi += 0.05
            if self.pmi > 55 and self.cpi > 2.5:
                self.regime = "확장/과열 (Expansion)"

        elif self.regime == "확장/과열 (Expansion)":
            self.pmi += 0.3
            self.cpi += 0.15
            self.unemployment -= 0.05
            if self.cpi > 3.5 or self.base_rate > 3.5:
                self.regime = "스태그/둔화 (Slowdown)"

        elif self.regime == "스태그/둔화 (Slowdown)":
            self.pmi -= 1.2
            self.gdp_growth -= 0.2
            self.cpi += 0.05
            if self.pmi < 49:
                self.regime = "침체 (Recession)"

        elif self.regime == "침체 (Recession)":
            self.pmi -= 0.8
            self.unemployment += 0.4
            self.cpi -= 0.15
            self.gdp_growth -= 0.4
            if (self.base_rate < 0.5 and prob < 0.4) or self.pmi < 35:
                self.regime = "회복 (Recovery)"

        # Random Shock (+ PMI high-level vol damp)
        self.gdp_growth += np.random.normal(0, 0.15)
        self.cpi += np.random.normal(0, 0.10)

        pmi_vol = 0.9 - 0.01 * max(0.0, self.pmi - 55)
        pmi_vol = max(0.45, pmi_vol)
        self.pmi += np.random.normal(0, pmi_vol)

        # Rate -> Real economy feedback (lag-lite)
        rate_gap = (self.base_rate - NEUTRAL_RATE)
        self.gdp_growth += -0.15 * rate_gap + np.random.normal(0, 0.05)
        self.pmi += -0.60 * rate_gap + np.random.normal(0, 0.10)
        self.cpi += -0.08 * rate_gap + np.random.normal(0, 0.03)

        # Okun
        okun_k = 0.35
        self.unemployment += -okun_k * (self.gdp_growth - POTENTIAL_GROWTH) + np.random.normal(0, 0.04)

        # Phillips + Mean Reversion
        output_gap = (self.gdp_growth - POTENTIAL_GROWTH)
        u_gap = (self.unemployment - U_STAR)
        self.cpi += (
            0.10 * output_gap
            - 0.08 * u_gap
            + 0.08 * (TARGET_INFLATION - self.cpi)
            + np.random.normal(0, 0.05)
        )
        if self.cpi < 1.0:
            self.cpi += 0.08 * (TARGET_INFLATION - self.cpi)

        # PMI Mean-Reversion
        pmi_anchor = 52.5
        k_pmi = 0.24
        self.pmi += -k_pmi * (self.pmi - pmi_anchor)

        # Constraints
        self.pmi = max(35.0, min(66.0, self.pmi))
        self.cpi = max(0.0, min(10.0, self.cpi))
        self.unemployment = max(3.0, min(15.0, self.unemployment))
        self.gdp_growth = max(-5.0, min(6.0, self.gdp_growth))

        # Surprise
        self.surprises["GDP"] = self.gdp_growth - consensus_gdp
        self.surprises["CPI"] = self.cpi - consensus_cpi
        self.surprises["PMI"] = self.pmi - consensus_pmi

        # z-scores
        self.z_gdp = self.surprises["GDP"] / max(1e-6, SIG_GDP)
        self.z_cpi = self.surprises["CPI"] / max(1e-6, SIG_CPI)
        self.z_pmi = self.surprises["PMI"] / max(1e-6, SIG_PMI)

        # CPI history
        self.cpi_history.append(self.cpi)
        if len(self.cpi_history) > 3:
            self.cpi_history.pop(0)

        # BEI update
        avg_cpi = sum(self.cpi_history) / len(self.cpi_history)
        self.bei = (
            0.85 * self.bei
            + 0.15 * (0.6 * avg_cpi + 0.4 * TARGET_INFLATION)
            + np.random.normal(0, 0.05)
        )
        self.bei = max(0.0, min(6.0, self.bei))

        # -------------------------------------------------
        # 2) Central Bank Policy
        # -------------------------------------------------
        inflation_gap = avg_cpi - TARGET_INFLATION
        target_rate = NEUTRAL_RATE + 1.5 * inflation_gap + 0.6 * (self.gdp_growth - POTENTIAL_GROWTH)

        desired_dir = 0
        if target_rate > self.base_rate + 0.25:
            desired_dir = +1
        elif target_rate < self.base_rate - 0.25:
            desired_dir = -1

        step = 0.25
        if desired_dir != 0:
            if desired_dir == self.last_policy_dir:
                step = 0.25
            else:
                step = 0.25
            self.base_rate += desired_dir * step
            self.last_policy_dir = desired_dir
        else:
            self.last_policy_dir = 0

        self.base_rate = max(0.0, min(10.0, self.base_rate))

        # -------------------------------------------------
        # 3) Asset Pricing
        # -------------------------------------------------
        # (A) EPS
        pmi_effect = (self.pmi - 50) * 0.003
        eps_growth = (self.gdp_growth / 100) * 1.2 + pmi_effect

        margin_drag = (
            0.002 * max(0.0, (self.unemployment - U_STAR))
            + 0.003 * max(0.0, (self.base_rate - NEUTRAL_RATE))
            + 0.002 * max(0.0, (self.cpi - TARGET_INFLATION))
        )
        eps_growth -= margin_drag

        if eps_growth < -0.05:
            eps_growth = -0.05

        self.eps *= (1 + eps_growth)

        # (B) Valuation (P/E)
        target_pe = BASE_PE

        if self.base_rate < 2.0:
            target_pe += (2.0 - self.base_rate) * 4.0
        elif self.base_rate > 4.0:
            target_pe -= (self.base_rate - 4.0) * 2.0

        level_score = max(0, (self.pmi - 50) * 0.4)
        pmi_delta = self.pmi - prev_pmi

        turnaround_premium = 0
        if self.pmi < 50 and pmi_delta > 0:
            turnaround_premium = (pmi_delta * 1.5) + (50 - self.pmi) * 0.1
        elif self.pmi >= 50 and pmi_delta > 0:
            turnaround_premium = pmi_delta * 0.5

        target_pe += level_score + turnaround_premium

        target_pe -= (self.z_cpi * 0.8)   # CPI surprise
        target_pe += (self.z_pmi * 0.6)   # PMI surprise

        unemp_jump = max(0.0, self.unemployment - prev_unemp)

        risk_off = 0.0
        risk_off += 0.8 * max(0.0, (self.unemployment - U_STAR))
        risk_off += 3.0 if self.spread < 0 else 0.0
        risk_off += 0.25 * max(0.0, (50 - self.pmi))
        risk_off += 1.2 * (unemp_jump / 0.3)
        target_pe -= risk_off

        real_yield_proxy = (prev_yield_10y - prev_bei)
        erp = 3.5 + 0.6 * max(0, 50 - self.pmi) / 10 + (0.8 if self.spread < 0 else 0.0)
        disc = max(1.0, real_yield_proxy + erp)
        g = 1.5 + 0.05 * (self.pmi - 50)
        struct_pe = 100 / max(1.0, (disc - g))
        target_pe = 0.8 * target_pe + 0.2 * struct_pe

        if self.pmi < 50 and (self.pmi - prev_pmi) > 0:
            target_pe += 0.6 * (self.pmi - prev_pmi)

        self.pe_ratio = self.pe_ratio * 0.35 + target_pe * 0.65 + np.random.normal(0, 0.35)
        self.pe_ratio = max(8.0, min(40.0, self.pe_ratio))

        raw = (self.eps / self.base_eps0) * (self.pe_ratio / self.base_pe0)
        self.stock_index = self.base_index_level * raw

        # -------------------------------------------------
        # Risk-off equity shock (vol linked to risk_score)
        # -------------------------------------------------
        sigma_eq = 0.004 + 0.0035 * min(5.0, self.risk_score)
        equity_shock = -0.018 * self.risk_score + np.random.normal(0, sigma_eq)

        equity_shock = max(-0.20, min(0.04, equity_shock))
        if self.regime == "침체 (Recession)":
            equity_shock = max(-0.32, min(0.04, equity_shock))

        self.stock_index *= (1 + equity_shock)
        self.stock_index = STOCK_SOFT_CAP * np.tanh(self.stock_index / STOCK_SOFT_CAP)

        # -------------------------------------------------
        # (C) Bonds & Commodities
        # -------------------------------------------------
        # Bonds
        self.yield_2y = self.base_rate * 0.9 + 0.3 + np.random.normal(0, 0.05)
        self.yield_2y = max(0.0, self.yield_2y)

        term_premium = 0.8 + 0.15 * np.tanh((self.pmi - 50) / 8)

        ftq = 0.14 * min(6.0, self.risk_score)
        target_10y = 0.6 * self.base_rate + 0.4 * (self.bei + 1.0) + term_premium - ftq

        self.yield_10y = 0.82 * self.yield_10y + 0.18 * target_10y + np.random.normal(0, 0.06)
        self.yield_10y = max(self.base_rate - 0.5, self.yield_10y)

        self.spread = self.yield_10y - self.yield_2y

        # Recompute risk AFTER spread update
        unemp_jump = max(0.0, self.unemployment - prev_unemp)

        risk_score = 0.0
        risk_score += 0.6 * max(0.0, 50 - self.pmi)
        risk_score += 2.5 if self.spread < 0 else 0.0
        risk_score += 1.2 * (unemp_jump / 0.3)
        risk_score += 0.8 * max(0.0, self.z_cpi)

        self.risk_score = float(risk_score)

        if self.risk_score < 1.5:
            self.risk_label = "LOW"
        elif self.risk_score < 3.5:
            self.risk_label = "MED"
        else:
            self.risk_label = "HIGH"

        self.risk_parts = {
            "PMI": 0.6 * max(0.0, 50 - self.pmi),
            "Spread": 2.5 if self.spread < 0 else 0.0,
            "UnempJump": 1.2 * (unemp_jump / 0.3),
            "CPI": 0.8 * max(0.0, self.z_cpi),
        }

        # Commodities
        oil_drift = 0.002 * max(0.0, self.cpi - TARGET_INFLATION) + 0.001 * max(0.0, self.gdp_growth - POTENTIAL_GROWTH)
        oil_chg = 0.002 * (self.pmi - 50) + 0.01 * (self.gdp_growth / POTENTIAL_GROWTH) + oil_drift + np.random.normal(0, 0.03)
        self.oil_index *= (1 + oil_chg)

        real_yield = self.yield_10y - self.bei
        d_real = real_yield - prev_real_yield

        gold_risk_prem = 0.002 * min(6.0, self.risk_score)
        risk_on_off = (-0.015 if self.spread < 0 else 0.0)

        gold_chg = (-0.03 * d_real) + risk_on_off + gold_risk_prem + np.random.normal(0, 0.02)
        if self.regime == "침체 (Recession)":
            gold_chg += 0.005
        self.gold_index *= (1 + gold_chg)

        # -------------------------------------------------
        # 4) Returns
        # -------------------------------------------------
        ret_stock = (self.stock_index / prev_stock - 1) * 100
        ret_oil = (self.oil_index / prev_oil - 1) * 100
        ret_gold = (self.gold_index / prev_gold - 1) * 100

        ret_bond2 = (prev_yield_2y / 12) + (-1.8 * (self.yield_2y - prev_yield_2y))
        ret_bond10 = (prev_yield_10y / 12) + (-8.0 * (self.yield_10y - prev_yield_10y))
        ret_cash = prev_rate / 12

        returns = {
            "주식": ret_stock,
            "국채2년": ret_bond2,
            "국채10년": ret_bond10,
            "원유": ret_oil,
            "금": ret_gold,
            "현금": ret_cash,
        }

        self.delta_rate = self.base_rate - prev_rate
        self.delta_cpi = self.cpi - prev_cpi
        self.delta_pmi = self.pmi - prev_pmi

        # News
        if turnaround_premium > 3.0 and self.pmi < 50:
            news = f"📈 바닥 확인? PMI 반등({self.pmi:.1f})에 저가 매수세 유입! (Risk {self.risk_label})"
        elif abs(ret_stock) > 5.0:
            news = f"💥 주식시장 {'급등' if ret_stock>0 else '급락'}! P/E {self.pe_ratio:.1f}배. (Risk {self.risk_label})"
        elif self.base_rate < 1.0 and self.regime == "회복 (Recovery)":
            news = f"💸 저금리 유동성 장세 (Goldilocks). 주가 상승 압력. (Risk {self.risk_label})"
        elif self.spread < 0:
            news = f"⚠️ 장단기 금리 역전 ({self.spread:.2f}%p). 경기 침체 경고. (Risk {self.risk_label})"
        elif self.cpi > 4.0:
            news = f"🔥 인플레이션 경고 (CPI {self.cpi:.1f}%). 긴축 우려. (Risk {self.risk_label})"
        else:
            news = f"시장 상황: {self.regime}. 매크로 지표 안정적. (Risk {self.risk_label})"

        return returns, news


# =========================================================
# 6) Engine Init + Initial History Row
# =========================================================
if "engine" not in st.session_state:
    st.session_state.engine = EconomicEngineV21()

    eng = st.session_state.engine
    st.session_state.history.loc[0] = [
        0, INITIAL_CAPITAL, INITIAL_CAPITAL, "Start",
        eng.gdp_growth, eng.cpi, eng.pmi, eng.base_rate,
        eng.stock_index, eng.pe_ratio, eng.eps, eng.yield_10y, eng.yield_2y,
        eng.unemployment, eng.bei, eng.spread,
        eng.risk_score, eng.risk_label, eng.z_gdp, eng.z_cpi, eng.z_pmi,
    ]


# =========================================================
# 7) Sidebar (Weights + Buttons)
# =========================================================
with st.sidebar:
    st.markdown(f"### 📅 Month: {st.session_state.month} / {TOTAL_MONTHS}")

    rid = st.session_state.reset_count

    def get_weight(name, default_val):
        return st.session_state.get(f"{name}_{rid}", default_val)

    current_stock = get_weight("w_stock", 20)
    current_bond2 = get_weight("w_bond2", 10)
    current_bond10 = get_weight("w_bond10", 30)
    current_oil = get_weight("w_oil", 10)
    current_gold = get_weight("w_gold", 10)
    current_cash = get_weight("w_cash", 20)

    total_weight = current_stock + current_bond2 + current_bond10 + current_oil + current_gold + current_cash
    btn_disabled = total_weight != 100

    col_btn1, col_btn2 = st.columns([3, 1])

    # ---- Execute one month
    with col_btn1:
        if st.button("⚡ 투자 실행", disabled=btn_disabled, key="exec_btn", use_container_width=True):
            if st.session_state.month < TOTAL_MONTHS:
                eng = st.session_state.engine
                returns, news = eng.next_turn()

                w = {
                    "주식": current_stock,
                    "국채2년": current_bond2,
                    "국채10년": current_bond10,
                    "원유": current_oil,
                    "금": current_gold,
                    "현금": current_cash,
                }

                my_ret = sum([w[k] * returns[k] for k in w]) / 100
                bm_ret = 0.6 * returns["주식"] + 0.4 * returns["국채10년"]

                st.session_state.my_balance *= (1 + my_ret / 100)
                st.session_state.bm_balance *= (1 + bm_ret / 100)
                st.session_state.month += 1

                st.session_state.last_returns = returns
                st.session_state.last_news = news

                st.session_state.history.loc[len(st.session_state.history)] = [
                    st.session_state.month, st.session_state.my_balance, st.session_state.bm_balance, eng.regime,
                    eng.gdp_growth, eng.cpi, eng.pmi, eng.base_rate,
                    eng.stock_index, eng.pe_ratio, eng.eps, eng.yield_10y, eng.yield_2y,
                    eng.unemployment, eng.bei, eng.spread,
                    eng.risk_score, eng.risk_label, eng.z_gdp, eng.z_cpi, eng.z_pmi,
                ]

                st.rerun()

    # ---- Reset
    with col_btn2:
        if st.button("🔄", help="초기화", key="reset_btn", use_container_width=True):
            st.session_state.clear()
            st.session_state.reset_count = rid + 1
            st.rerun()

    # ---- Auto-run 6 months
    if st.button("⏩ 6개월 자동 진행", use_container_width=True, disabled=btn_disabled, key="auto_run_6m"):
        with st.spinner("6개월 자동 진행 중..."):
            for _ in range(6):
                if st.session_state.month >= TOTAL_MONTHS:
                    break

                eng = st.session_state.engine
                returns, news = eng.next_turn()

                w = {
                    "주식": current_stock,
                    "국채2년": current_bond2,
                    "국채10년": current_bond10,
                    "원유": current_oil,
                    "금": current_gold,
                    "현금": current_cash,
                }

                my_ret = sum([w[k] * returns[k] for k in w]) / 100
                bm_ret = 0.6 * returns["주식"] + 0.4 * returns["국채10년"]

                st.session_state.my_balance *= (1 + my_ret / 100)
                st.session_state.bm_balance *= (1 + bm_ret / 100)
                st.session_state.month += 1

                st.session_state.last_returns = returns
                st.session_state.last_news = news

                st.session_state.history.loc[len(st.session_state.history)] = [
                    st.session_state.month, st.session_state.my_balance, st.session_state.bm_balance, eng.regime,
                    eng.gdp_growth, eng.cpi, eng.pmi, eng.base_rate,
                    eng.stock_index, eng.pe_ratio, eng.eps, eng.yield_10y, eng.yield_2y,
                    eng.unemployment, eng.bei, eng.spread,
                    eng.risk_score, eng.risk_label, eng.z_gdp, eng.z_cpi, eng.z_pmi,
                ]

        st.rerun()

    if btn_disabled:
        st.error(f"합계 {total_weight}% (100% 맞춰주세요)")

    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
    st.markdown("**포트폴리오 비중 설정**")

    st.slider("주식 (willouu)", 0, 100, value=20, key=f"w_stock_{rid}", step=10)
    st.slider("국채 2년 (단기)", 0, 100, value=10, key=f"w_bond2_{rid}", step=10)
    st.slider("국채 10년 (장기)", 0, 100, value=30, key=f"w_bond10_{rid}", step=10)
    st.slider("원유 (WTI)", 0, 100, value=10, key=f"w_oil_{rid}", step=10)
    st.slider("금 (Gold)", 0, 100, value=10, key=f"w_gold_{rid}", step=10)
    st.slider("현금 (Cash)", 0, 100, value=20, key=f"w_cash_{rid}", step=10)


# =========================================================
# 8) Main Dashboard
# =========================================================
eng = st.session_state.engine
df = st.session_state.history

# ---- Macro Top Row
c1, c2, c3, c4 = st.columns(4)


def metric_html(label, value, delta, sub, color="#333"):
    d_color = "red" if delta > 0 else "blue"
    if label == "PMI" and delta < 0:
        d_color = "blue"
    if delta == 0:
        d_color = "#999"
    return f"""
    <div class="metric-card">
        <div class="metric-title">{label}</div>
        <div class="metric-value" style="color:{color}">{value}</div>
        <div class="metric-delta" style="color:{d_color}">{delta:+.2f} <span style="font-size:0.8em;color:#999;">(MoM)</span></div>
        <div class="metric-sub">{sub}</div>
    </div>
    """


with c1:
    st.markdown(metric_html("ISM 제조업 PMI", f"{eng.pmi:.1f}", eng.delta_pmi, "경기 선행 (Ref 50)"), unsafe_allow_html=True)
with c2:
    st.markdown(metric_html("소비자물가 (CPI)", f"{eng.cpi:.1f}%", eng.delta_cpi, f"목표 {TARGET_INFLATION}%"), unsafe_allow_html=True)
with c3:
    st.markdown(metric_html("기준금리 (Base Rate)", f"{eng.base_rate:.2f}%", eng.delta_rate, "Policy Rate"), unsafe_allow_html=True)
with c4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">현재 국면 (Regime)</div>
            <div class="metric-value" style="color:#6200EA; font-size:1.2rem;">{eng.regime}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---- Risk/Surprise Row
r1, r2, r3, r4 = st.columns(4)

risk_color = "#2e7d32" if eng.risk_label == "LOW" else ("#ef6c00" if eng.risk_label == "MED" else "#c62828")

with r1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Risk Level</div>
            <div class="metric-value" style="color:{risk_color}">{eng.risk_label}</div>
            <div class="metric-sub">Score: {eng.risk_score:.2f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with r2:
    rp_pmi = eng.risk_parts.get("PMI", 0.0)
    rp_sp = eng.risk_parts.get("Spread", 0.0)
    rp_u = eng.risk_parts.get("UnempJump", 0.0)
    rp_cpi = eng.risk_parts.get("CPI", 0.0)

    st.markdown(
        f"""
        <div class="metric-card" title="PMI/Spread/Unemp/CPI risk contributions">
            <div class="metric-title">Risk Parts</div>
            <div class="metric-value" style="font-size:1.0rem;color:#333;">
                PMI {rp_pmi:.3f} · Spread {rp_sp:.3f}
            </div>
            <div class="metric-sub">
                Unemp {rp_u:.3f} · CPI {rp_cpi:.3f}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with r3:
    st.markdown(
        f"""
        <div class="metric-card" title="Standardized surprises (z-scores)">
            <div class="metric-title">Surprise (z)</div>
            <div class="metric-value" style="font-size:1.0rem;color:#333;">
                zPMI {eng.z_pmi:+.3f} · zCPI {eng.z_cpi:+.3f}
            </div>
            <div class="metric-sub">zGDP {eng.z_gdp:+.3f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with r4:
    real_y = eng.yield_10y - eng.bei
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Real Yield (10Y-BEI)</div>
            <div class="metric-value" style="font-size:1.1rem;color:#333;">{real_y:.2f}%</div>
            <div class="metric-sub">Gold/Valuation Driver</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ---- News & Returns
c_news, c_ret = st.columns([1, 1])
with c_news:
    st.info(f"📰 {st.session_state.last_news}")

with c_ret:
    df_ret = pd.DataFrame([st.session_state.last_returns]).rename(index={0: "수익률"})
    st.dataframe(
        df_ret.style.format("{:.1f}%").background_gradient(cmap="RdYlGn", vmin=-3, vmax=3, axis=1),
        use_container_width=True,
    )

# ---- Wealth chart
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=df["Month"],
        y=df["My Wealth"] / 10000,
        name="내 자산",
        line=dict(color="blue", width=3),
    )
)
fig.add_trace(
    go.Scatter(
        x=df["Month"],
        y=df["BM Wealth"] / 10000,
        name="벤치마크 (60/40)",
        line=dict(color="gray", dash="dot"),
    )
)
fig.update_layout(
    height=300,
    margin=dict(t=10, b=10, l=10, r=10),
    yaxis_title="자산 (만원)",
    xaxis_title="Months",
    yaxis=dict(tickformat=","),
)
st.plotly_chart(fig, use_container_width=True)

# =========================================================
# 9) Market Snapshot
# =========================================================
st.markdown("### 🏦 금융시장 스냅샷 (Market Snapshot)")
f1, f2, f3, f4, f5 = st.columns(5)


def fin_card(title, value, suffix=""):
    return f"""
    <div class="fin-card">
        <div class="fin-title">{title}</div>
        <div class="fin-value">{value}{suffix}</div>
    </div>
    """


with f1:
    st.markdown(fin_card("willouu 지수", f"{eng.stock_index:.1f}"), unsafe_allow_html=True)
with f2:
    pe_color = "red" if eng.pe_ratio > 25 else ("blue" if eng.pe_ratio < 15 else "black")
    st.markdown(
        f"""
        <div class="fin-card">
            <div class="fin-title">주가수익비율 (P/E)</div>
            <div class="fin-value" style="color:{pe_color}">{eng.pe_ratio:.1f}x</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with f3:
    st.markdown(fin_card("국채 10년 금리", f"{eng.yield_10y:.2f}", "%"), unsafe_allow_html=True)
with f4:
    st.markdown(fin_card("국채 2년 금리", f"{eng.yield_2y:.2f}", "%"), unsafe_allow_html=True)
with f5:
    st.markdown(fin_card("원유(WTI) 지수", f"{eng.oil_index:.1f}"), unsafe_allow_html=True)

st.markdown("---")


# =========================================================
# 10) Tabs — Financial Indicators
# =========================================================
st.markdown("### 📉 금융시장 지표 (Financial Market Indicators)")
tab_f1, tab_f2, tab_f3 = st.tabs(["🏢 펀더멘털 & 밸류에이션", "💸 금리 & 채권", "📋 기록 데이터"])

with tab_f1:
    col_f1, col_f2 = st.columns([3, 1])
    with col_f1:
        f_chart = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("주가지수(Left) vs EPS(Right)", "P/E Ratio 추이"),
            specs=[[{"secondary_y": True}, {"secondary_y": False}]],
        )

        f_chart.add_trace(
            go.Scatter(x=df["Month"], y=df["StockIdx"], name="주가지수", line=dict(color="blue")),
            row=1,
            col=1,
            secondary_y=False,
        )
        f_chart.add_trace(
            go.Scatter(x=df["Month"], y=df["EPS"], name="EPS", line=dict(color="orange", dash="dot")),
            row=1,
            col=1,
            secondary_y=True,
        )

        f_chart.add_trace(
            go.Scatter(x=df["Month"], y=df["PE"], name="P/E", line=dict(color="purple"), showlegend=False),
            row=1,
            col=2,
        )
        f_chart.add_hline(y=BASE_PE, line_dash="dash", line_color="gray", row=1, col=2)

        f_chart.update_layout(height=300, showlegend=True, legend=dict(x=0, y=1.1, orientation="h"))
        st.plotly_chart(f_chart, use_container_width=True)

    with col_f2:
        st.markdown(
            """
            <div class="desc-box">
            <b>1. 주가 = EPS × P/E</b><br>
            - 📉 <b>EPS (오렌지 점선)</b>: 기업의 이익 체력. 우측 축(Y-Axis) 확인.<br>
            - 📊 <b>P/E (보라색 실선)</b>: 투자자의 심리와 유동성. 금리가 오르면 P/E는 낮아집니다.
            </div>
            """,
            unsafe_allow_html=True,
        )

with tab_f2:
    col_i1, col_i2 = st.columns([3, 1])
    with col_i1:
        i1 = make_subplots(rows=1, cols=2, subplot_titles=("국채 금리 (Yields)", "장단기 스프레드 (10Y-2Y)"))
        i1.add_trace(go.Scatter(x=df["Month"], y=df["Yield10Y"], name="10Y"), row=1, col=1)
        i1.add_trace(go.Scatter(x=df["Month"], y=df["Yield2Y"], name="2Y"), row=1, col=1)
        i1.add_trace(
            go.Scatter(
                x=df["Month"],
                y=df["Yield10Y"] - df["Yield2Y"],
                name="Spread",
                fill="tozeroy",
            ),
            row=1,
            col=2,
        )
        i1.add_hline(y=0, line_color="red", row=1, col=2)
        i1.update_layout(height=300, showlegend=True)
        st.plotly_chart(i1, use_container_width=True)

    with col_i2:
        st.markdown(
            """
            <div class="desc-box">
            <b>1. 수익률 곡선 (Yield Curve)</b><br>
            - <b>정상 (우상향)</b>: 장기금리 > 단기금리. 경기 확장기.<br>
            - <b>역전 (Inverted)</b>: 단기금리 > 장기금리. 경기 침체의 강력한 전조입니다.
            </div>
            """,
            unsafe_allow_html=True,
        )

with tab_f3:
    st.dataframe(df.sort_values(by="Month", ascending=False), use_container_width=True)


st.markdown("<br>", unsafe_allow_html=True)


# =========================================================
# 11) Tabs — Economic Indicators
# =========================================================
st.markdown("### 📊 경제지표 상세 분석 (Economic Indicators)")
tab_e1, tab_e2, tab_e3 = st.tabs(["📈 성장 (Growth)", "💸 물가 (Inflation)", "🏛 정책 (Policy)"])

with tab_e1:
    col_g1, col_g2 = st.columns([2, 1])
    with col_g1:
        g1 = make_subplots(rows=1, cols=2, subplot_titles=("ISM 제조업 PMI", "실업률 (Unemployment)"))
        g1.add_trace(go.Scatter(x=df["Month"], y=df["PMI"], name="PMI", line=dict(color="green")), row=1, col=1)
        g1.add_trace(go.Scatter(x=df["Month"], y=df["Unemployment"], name="실업률", line=dict(color="orange")), row=1, col=2)
        g1.add_hline(y=50, line_dash="dash", line_color="gray", row=1, col=1)
        g1.update_layout(height=300, showlegend=False, margin=dict(t=30))
        st.plotly_chart(g1, use_container_width=True)

    with col_g2:
        st.markdown(
            """
            <div class="desc-box">
            <b>1. ISM 제조업 PMI (선행)</b><br>
            - 50 이상: 경기 확장 / 50 미만: 경기 수축<br>
            - 주식시장에 가장 큰 영향을 미치는 선행지표. GDP보다 먼저 움직입니다.
            </div>
            <div class="desc-box">
            <b>2. 실업률 (후행)</b><br>
            - 경기가 완전히 식어야 오르기 시작합니다.<br>
            - 실업률이 급등하면 연준이 금리를 내리기 시작합니다.
            </div>
            """,
            unsafe_allow_html=True,
        )

with tab_e2:
    col_k1, col_k2 = st.columns([2, 1])
    with col_k1:
        k1 = make_subplots(rows=1, cols=2, subplot_titles=("소비자물가 (CPI)", "기대인플레이션 (BEI)"))
        k1.add_trace(go.Scatter(x=df["Month"], y=df["CPI"], name="CPI", line=dict(color="red")), row=1, col=1)
        k1.add_trace(go.Scatter(x=df["Month"], y=df["BEI"], name="BEI", line=dict(color="purple")), row=1, col=2)
        k1.add_hline(y=2.0, line_dash="dash", line_color="gray", row=1, col=1)
        k1.update_layout(height=300, showlegend=False, margin=dict(t=30))
        st.plotly_chart(k1, use_container_width=True)

    with col_k2:
        st.markdown(
            """
            <div class="desc-box">
            <b>1. 소비자물가 (CPI)</b><br>
            - 연준 금리 결정의 핵심.<br>
            - 예상보다 높게 나오면(Surprise) 금리 인상 공포로 주식/채권 모두 하락합니다.
            </div>
            <div class="desc-box">
            <b>2. 기대인플레이션 (BEI)</b><br>
            - 시장이 예상하는 미래 물가.<br>
            - 이것이 오르면 명목 금리가 상승하여 채권 가격이 떨어집니다.
            </div>
            """,
            unsafe_allow_html=True,
        )

with tab_e3:
    col_pol1, col_pol2 = st.columns([2, 1])
    with col_pol1:
        pol1 = make_subplots(rows=1, cols=2, subplot_titles=("기준금리 (Fed Rate)", "장단기 금리차 (10Y-2Y)"))
        pol1.add_trace(go.Scatter(x=df["Month"], y=df["BaseRate"], name="금리", line=dict(color="black")), row=1, col=1)
        pol1.add_trace(go.Scatter(x=df["Month"], y=df["Spread"], name="Spread", line=dict(color="blue")), row=1, col=2)
        pol1.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=2)
        pol1.update_layout(height=300, showlegend=False, margin=dict(t=30))
        st.plotly_chart(pol1, use_container_width=True)

    with col_pol2:
        st.markdown(
            """
            <div class="desc-box">
            <b>1. 기준금리 (Fed Funds Rate)</b><br>
            - 모든 자산 가격의 중력.<br>
            - 금리가 오르면 현금 매력이 증가하고 자산 가격은 조정받습니다.
            </div>
            <div class="desc-box">
            <b>2. 장단기 금리차 (Spread)</b><br>
            - 0 아래로 내려가면(역전) 1년 내 경기 침체 확률 90% 이상.<br>
            - 침체 시그널이 오면 주식을 줄이고 장기채를 사야 합니다.
            </div>
            """,
            unsafe_allow_html=True,
        )


# =========================================================
# 12) End Condition
# =========================================================
if st.session_state.month >= TOTAL_MONTHS:
    st.balloons()
    final_alpha = (st.session_state.my_balance - st.session_state.bm_balance) / INITIAL_CAPITAL * 100
    st.success(f"시뮬레이션 종료! Final Alpha: {final_alpha:.2f}%p")

    if st.button("🔄 리셋"):
        st.session_state.clear()
        st.rerun()

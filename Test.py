import requests

# ==== Hàm gọi mô hình AI local qua Ollama ====
def ask_ai(prompt):
    try:
        url = "http://localhost:11434/api/generate"
        res = requests.post(url, json={
            "model": "mistral",
            "prompt": prompt,
            "stream": False
        })
        return res.json()["response"] if res.ok else "⚠️ Lỗi gọi mô hình AI."
    except Exception as e:
        return f"❌ Lỗi: {e}"


# ===== 📦 LIBRARIES =====  
import os
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import cdist
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from tslearn.clustering import TimeSeriesKMeans
from tslearn.metrics import cdist_dtw
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# ===== 📌 CONFIG FILE PATHS =====
RATIO_CSV = r"D:\VSCODE\Data\ratios_df-_1_.csv"
FINANCIAL_XLSX = r"D:\VSCODE\Data\df_all (3).xlsx"
FORECAST_XLSX = r"D:\VSCODE\Data\forecast_2025_Q1_Q4.xlsx"
RELIABLE_FORECAST_XLSX = r"D:\VSCODE\Data\reliable_forecasts_q1_q4_2025.xlsx"
EVAL_XLSX = r"D:\VSCODE\Data\model_evaluation_results.xlsx"
COMPANY_XLS = r"D:\VSCODE\Data\12company.xlsx"
inv = pd.read_csv(r'D:\VSCODE\Data\investor_preferences_100000.csv')
# ===== Streamlit Config =====
st.set_page_config(page_title="Dashboard", layout="wide")
st.title("Hệ thống hỗ trợ phân tích doanh nghiệp và đề xuất đầu tư")
st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 16px;
        padding: 8px 20px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df_ratio = pd.read_csv(RATIO_CSV)
    df_fin   = pd.read_excel(FINANCIAL_XLSX)
    fc       = pd.read_excel(FORECAST_XLSX)
    fc_rel   = pd.read_excel(RELIABLE_FORECAST_XLSX) 
    ev       = pd.read_excel(EVAL_XLSX)
    comp     = pd.read_excel(COMPANY_XLS)
    return df_ratio, df_fin, fc, fc_rel, ev, comp

df_ratio, df_fin, fc_df, fc_rel_df, ev_df, comp_df = load_data()
def_cols = ['DebtEquity','CurrentRatio','QuickRatio','GrossMargin','NetMargin','CashFlow_to_Profit','CashFlow_to_Debt','SellingExpense_Ratio','debt_to_asset','EBIT_to_Interest','AssetTurnover','ROA','ROE']
avail = [c for c in def_cols if c in df_ratio.columns]
DF = df_ratio.copy()
DF[avail] = DF.groupby(['Năm','Kỳ'])[avail].transform(lambda x: x.fillna(x.mean()))

# ===== Tabs =====
tabs = st.tabs([
    "📊 Đánh giá doanh nghiệp",
    "📌 Phân cụm & Trực quan",
    "🧭 Gợi ý đầu tư"
])

# ===== 📊 TAB 0: DASHBOARD ĐÁNH GIÁ DOANH NGHIỆP =====
with tabs[0]:
    st.header("📊 Đánh giá tài chính doanh nghiệp")
    cp = st.selectbox("Chọn mã cổ phiếu", sorted(DF['CP'].unique()), key='eval_cp')
    years = sorted(DF['Năm'].unique())
    from_year, to_year = st.select_slider("Chọn khoảng năm", options=years, value=(2013, years[-1]), key='eval_year_range')

    df_cp = DF[(DF['CP'] == cp) & (DF['Năm'] >= from_year) & (DF['Năm'] <= to_year)].sort_values(['Năm', 'Kỳ'])
    latest = df_cp.dropna(subset=avail).iloc[-1]

    col1, col2, col3 = st.columns(3)
    col1.metric("ROE", f"{latest['ROE']:.2%}")
    col2.metric("Biên EBIT", f"{(latest['EBIT_to_Interest']/100 if latest['EBIT_to_Interest'] else 0):.2%}")
    col3.metric("Current Ratio", f"{latest['CurrentRatio']:.2f}")

    c1, c2 = st.columns([1, 1.2])
    with c1:
        st.subheader("Hiệu suất sinh lời")
        radar_vars = ['ROE', 'ROA', 'GrossMargin', 'NetMargin', 'AssetTurnover']
        radar_all = DF.groupby('CP')[radar_vars].mean().dropna()
        scaler = MinMaxScaler()
        radar_scaled = pd.DataFrame(scaler.fit_transform(radar_all), columns=radar_vars, index=radar_all.index)
        if cp in radar_scaled.index:
            values = radar_scaled.loc[cp].values
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(r=list(values) + [values[0]], theta=radar_vars + [radar_vars[0]], fill='toself', name=cp))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1], tickformat=".2f")), showlegend=False, margin=dict(l=30, r=30, t=30, b=30), font=dict(size=14))
            st.plotly_chart(fig_radar, use_container_width=True)

            with st.expander("🤖 Phân tích AI: Hiệu suất sinh lời"):
                vals = radar_all.loc[cp].round(3)
                radar_desc = ", ".join([f"{k}={v:.3f}" for k, v in vals.items()])
                radar_prompt = f"Các chỉ số sinh lời của công ty {cp}: {radar_desc}. Hãy phân tích ý nghĩa và nhận định tổng quan."
                if st.button("💬 Phân tích AI Radar", key="radar_ai"):
                    st.success(ask_ai(radar_prompt))
        else:
            st.warning("Không đủ dữ liệu để hiển thị radar chart.")

    with c2:
        st.subheader("Rủi ro và thanh khoản")
        bar_vars = ['DebtEquity', 'debt_to_asset', 'CurrentRatio', 'QuickRatio']
        bar_df = pd.DataFrame({"Chỉ số": bar_vars, "Giá trị": latest[bar_vars].values})
        fig_bar = px.bar(bar_df, x="Chỉ số", y="Giá trị", text_auto=True)
        fig_bar.update_layout(font=dict(size=14))
        st.plotly_chart(fig_bar, use_container_width=True)

        with st.expander("🤖 Phân tích AI: Rủi ro & thanh khoản"):
            bar_values = ", ".join([f"{i['Chỉ số']}={i['Giá trị']:.2f}" for i in bar_df.to_dict("records")])
            bar_prompt = f"Các chỉ số thanh khoản & rủi ro của công ty {cp}: {bar_values}. Đánh giá tổng thể về mức độ an toàn tài chính."
            if st.button("💬 Phân tích AI Bar", key="bar_ai"):
                st.success(ask_ai(bar_prompt))

    st.subheader("Xu hướng ROE & NetMargin theo thời gian")
    df_line = df_cp.copy()
    df_line['Thời gian'] = df_line['Năm'].astype(str) + 'Q' + df_line['Kỳ'].astype(str)
    fig_line = px.line(df_line, x='Thời gian', y=['ROE', 'NetMargin'], markers=True)
    fig_line.update_layout(yaxis=dict(title='Tỷ suất (%)', tickformat='.2%'), legend=dict(orientation="h", y=1.02, x=1), font=dict(size=14))
    st.plotly_chart(fig_line, use_container_width=True)

    with st.expander("🤖 Phân tích AI: Xu hướng ROE & NetMargin"):
        df_line['ROE%'] = df_line['ROE'].map(lambda x: f"{x:.2%}")
        df_line['NetMargin%'] = df_line['NetMargin'].map(lambda x: f"{x:.2%}")
        trend = "\n".join([f"{row['Thời gian']}: ROE={row['ROE%']}, NetMargin={row['NetMargin%']}" for _, row in df_line.iterrows()])
        line_prompt = f"Xu hướng ROE và NetMargin của công ty {cp} theo thời gian:\n{trend}\n→ Đưa ra nhận xét về hiệu quả tài chính và xu hướng lợi nhuận."
        if st.button("💬 Phân tích AI Line", key="line_ai"):
            st.success(ask_ai(line_prompt))

    st.subheader("Tăng trưởng trung bình năm")
    trend_df = df_cp.groupby('Năm')[['ROE', 'NetMargin']].mean().reset_index()
    fig_growth = px.bar(trend_df, x='Năm', y=['ROE', 'NetMargin'], barmode='group')
    fig_growth.update_layout(yaxis_tickformat=".2%", font=dict(size=14))
    st.plotly_chart(fig_growth, use_container_width=True)

    with st.expander("🤖 Phân tích AI: Tăng trưởng trung bình năm"):
        growth_desc = ", ".join([f"{r['Năm']}: ROE={r['ROE']:.2%}, NetMargin={r['NetMargin']:.2%}" for _, r in trend_df.iterrows()])
        growth_prompt = f"Tăng trưởng ROE và NetMargin trung bình năm của công ty {cp}: {growth_desc}. Đưa ra đánh giá tổng quan."
        if st.button("💬 Phân tích AI Growth", key="growth_ai"):
            st.success(ask_ai(growth_prompt))

    d1, d2 = st.columns([1, 1.4])
    with d1:
        st.subheader("📋 Toàn bộ chỉ số mới nhất")
        st.dataframe(latest[avail].round(2).to_frame("Giá trị"), height=300, use_container_width=True)

    with d2:
        st.subheader("Dòng tiền từ hoạt động kinh doanh")
        cash_col = 'Lưu chuyển tiền tệ ròng từ các hoạt động SXKD'
        df_cash = df_fin[(df_fin['CP'] == cp) & (df_fin['Năm'] >= from_year) & (df_fin['Năm'] <= to_year)].copy()
        df_cash['Thời gian'] = df_cash['Năm'].astype(str) + 'Q' + df_cash['Kỳ'].astype(str)
        df_cash = df_cash.sort_values(['Năm', 'Kỳ'])
        if cash_col in df_cash.columns and not df_cash[cash_col].isnull().all():
            fig_cash = px.line(df_cash, x='Thời gian', y=cash_col, markers=True)
            fig_cash.update_traces(line_color='green').update_layout(font=dict(size=14))
            st.plotly_chart(fig_cash, use_container_width=True)

            with st.expander("🤖 Phân tích AI: Dòng tiền kinh doanh"):
                cash_vals = "\n".join([f"{r['Thời gian']}: {r[cash_col]:,.0f}" for _, r in df_cash.iterrows() if pd.notna(r[cash_col])])
                cash_prompt = f"Dòng tiền kinh doanh của công ty {cp} theo thời gian:\n{cash_vals}\n→ Nhận xét về tính ổn định và hiệu quả dòng tiền."
                if st.button("💬 Phân tích AI Cashflow", key="cash_ai"):
                    st.success(ask_ai(cash_prompt))
        else:
            st.info("⛔ Không có dữ liệu dòng tiền để hiển thị.")
    # ==== 🔮 Dự báo tài chính (Standard & Reliable) ====
    st.markdown("---")
    st.subheader("🔮 Dự báo tài chính")

    # 1️⃣ Chọn nguồn dữ liệu
    source = st.radio("Chọn bộ dữ liệu:", ["Standard", "Reliable"], horizontal=True)

    # 2️⃣ Thiết lập DataFrame và label chart
    if source == "Standard":
        fc_src = fc_df
        label  = ""
    else:
        fc_src = fc_rel_df
        label  = " (Reliable)"

    # 3️⃣ Selector CP & Chỉ số trên bộ đã chọn
    col1, col2 = st.columns(2)
    with col1:
        cp_fc  = st.selectbox(f"Mã CP{label}",     fc_src["CP"].unique(),       key=f"{source}_cp")
    with col2:
        met_fc = st.selectbox(f"Chỉ số{label}",   fc_src["Chỉ số"].unique(),   key=f"{source}_met")

    # 4️⃣ Chuẩn bị Actual
    act = (
        DF
        .melt(["CP","Năm","Kỳ"], avail, "Chỉ số","Giá trị")
        .query("CP == @cp_fc and `Chỉ số` == @met_fc")
        .assign(
            ds=lambda df_: pd.PeriodIndex(
                df_["Năm"].astype(str) + "Q" + df_["Kỳ"].astype(str),
                freq="Q"
            ).to_timestamp(how="end")
        )
    )

    # 5️⃣ Chuẩn bị Forecast
    fc_sel = (
        fc_src
        .query("CP == @cp_fc and `Chỉ số` == @met_fc")
        .rename(columns={"Dự báo":"Giá trị"})
        .assign(Type="Forecast")
    )

    # 6️⃣ Nếu không có Actual → cảnh báo
    if act.empty:
        st.warning("⛔ Không có dữ liệu Actual để hiển thị.")
    else:
        # 6.1️⃣ Up-to-4 actual gần nhất
        n_act = min(4, len(act))
        recent = (
            act[["ds","Giá trị"]]
            .assign(Type="Actual")
            .sort_values("ds")
            .tail(n_act)
        )

        # 6.2️⃣ Bridge point để nối Forecast liền mạch
        last_ds  = recent["ds"].max()
        last_val = recent.loc[recent["ds"] == last_ds, "Giá trị"].iat[0]
        bridge   = pd.DataFrame({
            "ds": [last_ds],
            "Giá trị": [last_val],
            "Type": ["Forecast"]
        })

        # 6.3️⃣ Gộp actual + bridge + forecast
        dfp = pd.concat([recent, bridge, fc_sel]) \
                .drop_duplicates(["ds","Type"]) \
                .sort_values("ds")

        # 7️⃣ Vẽ biểu đồ nối gap + marker
        fig = px.line(
            dfp, x="ds", y="Giá trị", color="Type",
            title=f"{cp_fc} — {met_fc}{label}",
            labels={"ds":"Thời điểm","Giá trị":"Giá trị"}
        )
        fig.update_traces(connectgaps=True, mode="lines+markers")
        fig.update_layout(margin=dict(t=40,b=20))
        st.plotly_chart(fig, use_container_width=True)

        # 8️⃣ Hiển thị đánh giá mô hình (MAE/RMSE/R²)
        ev = ev_df.query("CP == @cp_fc and `Chỉ số` == @met_fc")
        if not ev.empty:
            st.markdown("**Đánh giá mô hình (Test set):**")
            for _, r in ev.iterrows():
                st.write(f"- **{r['Mô hình']}**: MAE={r['MAE']:.2f}, RMSE={r['RMSE']:.2f}, R²={r['R2']:.2f}")
        else:
            st.info("⛔ Chưa có kết quả đánh giá cho cặp CP & Chỉ số này.")


# ======================= TAB 1: PHÂN CỤM & TRỰC QUAN =======================
with tabs[1]:
    st.header("\U0001F4CC Phân cụm & Trực quan doanh nghiệp")

    # ===== CHỌN K & PHƯƠNG PHÁP =====
    col1, col2, col3, col4, col5 = st.columns([1.2, 1.2, 1.5, 1.1, 1.1])
    with col1:
        k = st.slider("\U0001F522 Số cụm (k)", 2, 10, 4)
    with col2:
        method = st.selectbox("\U0001F9E0 Phương pháp", ['Feat-KMeans', 'GMM', 'Spectral', 'DTW-KMeans'])
    with col3:
        selected_metrics = st.multiselect("\U0001F4CF Chỉ số đánh giá", ['Silhouette','Dunn Index'], default=['Silhouette','Dunn Index'])

    # ===== TÍNH TOÁN ĐẶC TRƯNG =====
    feat_rows, cps = [], []
    for cp, g in DF.groupby('CP'):
        f = []; ok = True
        for c in avail:
            s = pd.Series(g[c]).replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
            if len(s) < 5: ok = False; break
            x = np.arange(len(s)).reshape(-1, 1)
            y = s.values.reshape(-1, 1)
            slope = LinearRegression().fit(x, y).coef_[0][0]
            d = np.diff(s)
            f += [slope, d.max() if d.size else 0, d.mean() if d.size else 0, skew(s), kurtosis(s), s.rolling(4).std().mean(), s.autocorr(1), s.autocorr(4)]
        if ok:
            feat_rows.append(f)
            cps.append(cp)
    feat_df = pd.DataFrame(feat_rows, index=cps)
    X_feat = StandardScaler().fit_transform(feat_df)

    # ===== XÂY DỰNG DỮ LIỆU TIME SERIES CHUẨN =====
    X_ts, cps2 = [], []
    keys = sorted(DF[['Năm', 'Kỳ']].drop_duplicates().values.tolist())
    for cp in DF['CP'].unique():
        company_df = DF[DF['CP'] == cp].set_index(['Năm', 'Kỳ']).sort_index()
        ts_matrix = []
        for col in avail:
            values = [company_df[col].get((y, k), np.nan) for (y, k) in keys]
            ts_matrix.append(values)
        ts_matrix = np.array(ts_matrix).T
        ts_matrix = pd.DataFrame(ts_matrix).replace([np.inf, -np.inf], np.nan).ffill().bfill().values
        if ts_matrix.shape == (len(keys), len(avail)) and not np.isnan(ts_matrix).any():
            X_ts.append(ts_matrix)
            cps2.append(cp)
    X_ts = np.array(X_ts)
    X_flat = np.nan_to_num(X_ts.reshape(X_ts.shape[0], -1), nan=0.0, posinf=0.0, neginf=0.0)
    X_scaled = StandardScaler().fit_transform(X_flat)

    # ===== CHẠY PHÂN CỤM =====
    labels_dict, index_dict = {}, {}
    labels_dict['Feat-KMeans'] = KMeans(n_clusters=k, random_state=42).fit_predict(X_feat)
    labels_dict['GMM'] = GaussianMixture(n_components=k, random_state=42).fit_predict(X_scaled)
    labels_dict['Spectral'] = SpectralClustering(n_clusters=k, affinity='precomputed', random_state=42).fit_predict(1 - np.corrcoef(X_scaled))
    labels_dtw = TimeSeriesKMeans(n_clusters=k, metric="dtw", random_state=42).fit_predict(X_ts)
    labels_dict["DTW-KMeans"] = labels_dtw

    index_dict = {
        'Feat-KMeans': cps,
        'GMM': cps2,
        'Spectral': cps2,
        'DTW-KMeans': cps2
    }
    df_dtw_cluster = pd.DataFrame({'Company': cps2, 'Cluster': labels_dtw})

    # ===== ĐÁNH GIÁ =====
    labels = labels_dict[method]
    if len(np.unique(labels)) < 2:
        sil_score, dunn_score = np.nan, np.nan
    else:
        try:
            dist_matrix = cdist_dtw(X_ts) if method == 'DTW-KMeans' else cdist(
                X_feat if method == 'Feat-KMeans' else X_scaled,
                X_feat if method == 'Feat-KMeans' else X_scaled
            )
            np.fill_diagonal(dist_matrix, 0.0)
            sil_score = silhouette_score(dist_matrix, labels, metric='precomputed')
            dunn_score = np.min([
                np.min(dist_matrix[np.ix_(labels == i, labels == j)])
                for i in np.unique(labels) for j in np.unique(labels) if i < j
            ]) / max([
                np.max(dist_matrix[np.ix_(labels == i, labels == i)])
                for i in np.unique(labels)
            ])
        except:
            sil_score, dunn_score = np.nan, np.nan

    with col4:
        st.metric("Silhouette", "N/A" if np.isnan(sil_score) else f"{sil_score:.4f}")
    with col5:
        st.metric("Dunn", "N/A" if np.isnan(dunn_score) else f"{dunn_score:.4f}")

    # ===== BẢNG PHÂN CỤM =====
    st.subheader("\U0001F4CA Phân bố doanh nghiệp")
    c1, c2 = st.columns([2, 1])
    with c1:
        ss = pd.Series(labels).value_counts().sort_index()
        fig = px.bar(x=ss.index.map(str), y=ss.values, labels={'x': 'Cụm', 'y': 'Số lượng'})
        fig.update_layout(font=dict(size=20), xaxis=dict(tickfont=dict(size=18)), yaxis=dict(tickfont=dict(size=18)))
        st.plotly_chart(fig, use_container_width=False, width=650, height=350)
    with c2:
        df_as = pd.DataFrame({
            'Company': index_dict[method],
            'Cluster': labels_dict[method]
        }).sort_values(by='Cluster').reset_index(drop=True)
        styled = (
            df_as.style
                .set_properties(**{'font-size': '18px', 'text-align': 'center'})
                .set_table_styles([
                    {'selector': 'th', 'props': [('font-size', '20px')]},
                    {'selector': 'td', 'props': [('font-size', '18px')]}
                ])
        )
        st.dataframe(styled, use_container_width=True, height=350)
        st.download_button("Download Assignments", df_as.to_csv(index=False).encode(), f"assign_{method}.csv")

    # ===== TRỰC QUAN THEO CỤM =====
    st.subheader("\U0001F4C8 Trực quan theo cụm")
    if method == 'DTW-KMeans':
        df_t = DF.copy()
        df_t['Cluster'] = df_t['CP'].map(dict(zip(cps2, labels)))
        df_t['Period'] = df_t['Năm'].astype(str) + '-Q' + df_t['Kỳ'].astype(str)
        sel = st.selectbox('Chọn chỉ số', avail)
        piv = df_t.pivot_table(index='Period', columns='Cluster', values=sel)
        st.line_chart(piv)
    else:
        Xvis = X_feat if method == 'Feat-KMeans' else X_scaled
        p = PCA(2).fit_transform(Xvis)
        dfv = pd.DataFrame(p, columns=['PC1','PC2'])
        dfv['Cluster'] = labels.astype(str)
        fig = px.scatter(dfv, x='PC1', y='PC2', color='Cluster')
        st.plotly_chart(fig, use_container_width=True)



import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import skew, kurtosis
from sklearn.metrics.pairwise import cosine_similarity

with tabs[2]:
    st.header("🔍 Đề xuất & Phân loại rủi ro doanh nghiệp")

    # --- 0. Load metadata và chuẩn hóa ---
    meta = pd.read_excel(r"D:\VSCODE\Data\Raw_data\Company_Name.xlsx")
    # Drop các cột Unnamed nếu có
    meta = meta.loc[:, ~meta.columns.str.contains("^Unnamed")]
    # Đổi tên cột Mã thành CP để merge cho đồng bộ
    meta = meta.rename(columns={"Mã": "CP"})
    meta["CP"] = meta["CP"].astype(str)

    # --- 1. Load lịch sử và forecast ---
    hist = comp_df.copy()
    hist["CP"] = hist["CP"].astype(str)
    hist["ds"] = pd.to_datetime(
        hist["Năm"].astype(str) + "-" +
        hist["Kỳ"].map({1:"03-31",2:"06-30",3:"09-30",4:"12-31"})
    )
    fc = pd.read_excel(RELIABLE_FORECAST_XLSX).rename(
        columns={"Dự báo":"Forecast","Chỉ số":"Metric"}
    )
    fc["CP"] = fc["CP"].astype(str)
    # --- Chuẩn hóa CP: uppercase và strip khoảng trắng ---
    meta["CP"] = meta["CP"].astype(str).str.upper().str.strip()
    hist["CP"] = hist["CP"].astype(str).str.upper().str.strip()
    fc["CP"]   = fc["CP"].astype(str).str.upper().str.strip()
    # --- Cập nhật lại cp_list ---
    cp_list = sorted(hist["CP"].unique())
    fc["ds"] = pd.to_datetime(fc["ds"])

    # --- 2. Chọn metrics coverage ≥50% ---
    cp_list   = sorted(hist["CP"].unique())
    num_cp    = len(cp_list)
    fc_counts = fc.dropna(subset=["Forecast"]).groupby("Metric")["CP"].nunique()
    sel_metrics = [m for m,ct in fc_counts.items() if ct >= 0.5 * num_cp]

    # --- 3. Median imputation cho forecast ---
    pivot_all = (
        fc.pivot_table(index="ds", columns="Metric", values="Forecast", aggfunc="mean")
          .reindex(columns=sel_metrics)
    )
    medians = pivot_all.median()

    # --- 4. Lấy label cluster từ lịch sử ---
    cluster_lbl = hist.groupby("CP")["Cluster"] \
                     .agg(lambda x: x.mode().iloc[0])

    # --- 5. Build features cho similarity ---
    def build_features(metrics):
        rows = []
        for cp in cp_list:
            rec = {"CP": cp}
            sub_h = hist[hist["CP"]==cp].set_index("ds").sort_index()
            sub_f = (fc[fc["CP"]==cp]
                     .pivot_table(index="ds", columns="Metric", values="Forecast", aggfunc="mean")
                     .reindex(columns=metrics)
                     .fillna(medians))
            for m in metrics:
                arr_h = sub_h[m].dropna().values
                dh    = np.diff(arr_h)
                rec[f"{m}_hist_slope"]        = np.polyfit(np.arange(len(arr_h)), arr_h,1)[0] if len(arr_h)>1 else np.nan
                rec[f"{m}_hist_delta_max"]    = np.max(np.abs(dh)) if dh.size else np.nan
                rec[f"{m}_hist_delta_mean"]   = np.mean(np.abs(dh)) if dh.size else np.nan
                rec[f"{m}_hist_autocorr_lag1"]= pd.Series(arr_h).autocorr(lag=1) if len(arr_h)>1 else np.nan
                arr_f = sub_f[m].values
                df_f  = np.diff(arr_f)
                rec[f"{m}_fc_slope"]          = np.polyfit(np.arange(len(arr_f)), arr_f,1)[0] if len(arr_f)>1 else np.nan
                rec[f"{m}_fc_delta_max"]      = np.max(np.abs(df_f)) if df_f.size else np.nan
                rec[f"{m}_fc_delta_mean"]     = np.mean(np.abs(df_f)) if df_f.size else np.nan
                rec[f"{m}_fc_autocorr_lag1"]  = pd.Series(arr_f).autocorr(lag=1) if len(arr_f)>1 else np.nan
            rec["cluster_score"] = int(cluster_lbl.loc[cp])
            rows.append(rec)
        return pd.DataFrame(rows)

    features = build_features(sel_metrics)
    feat_cols = [c for c in features.columns if c not in ["CP","cluster_score"]]

    # --- 6. Tính cosine similarity ---
    def masked_cos(u, v):
        mask = (~np.isnan(u)) & (~np.isnan(v))
        return (cosine_similarity(u[mask].reshape(1,-1), v[mask].reshape(1,-1))[0,0]
                if mask.sum()>0 else np.nan)

    data = features[feat_cols].values
    sims = np.zeros((len(data), len(data)))
    for i in range(len(data)):
        for j in range(len(data)):
            sims[i,j] = masked_cos(data[i], data[j])
    sims_df = pd.DataFrame(sims, index=cp_list, columns=cp_list)

    # --- 7. UI chọn CP gốc & số đề xuất ---
    ticker = st.selectbox("Chọn CP làm gốc đề xuất", cp_list, index=cp_list.index("FPT"))
    top_n  = st.slider("Số lượng đề xuất", 1, 10, 5)
    recs   = sims_df[ticker].drop(ticker).nlargest(top_n).index.astype(str).tolist()

    # --- 8. Tính risk_label ---
    df_sel = fc[fc["Metric"].isin(sel_metrics)].copy()
    df_sel["time_idx"] = pd.to_datetime(df_sel["ds"]).dt.quarter
    score_records = []
    for (cp,m), grp in df_sel.groupby(["CP","Metric"]):
        vals  = grp.sort_values("time_idx")["Forecast"].values
        t     = grp["time_idx"].values.reshape(-1,1)
        slope = LinearRegression().fit(t, vals).coef_[0]
        vol   = np.std(vals, ddof=1)
        cv    = vol / vals.mean() if vals.mean()!=0 else 0
        diffs = np.diff(vals)
        score_records.append({
            "CP":cp, "Metric":m,
            "slope":slope, "volatility":vol, "coef_var":cv,
            "delta_max":diffs.max() if diffs.size else 0,
            "delta_mean":np.mean(np.abs(diffs)) if diffs.size else 0,
            "skew":skew(vals), "kurtosis":kurtosis(vals),
            "autocorr_lag1":(np.corrcoef(vals[:-1],vals[1:])[0,1] if vals.size>2 else 0)
        })
    df_score = pd.DataFrame(score_records)
    df_score[["slope","volatility","coef_var","delta_max","delta_mean","skew","kurtosis","autocorr_lag1"]] = \
        MinMaxScaler().fit_transform(df_score[["slope","volatility","coef_var","delta_max","delta_mean","skew","kurtosis","autocorr_lag1"]])

    # Định nghĩa hướng score và gán điểm
    directions = {
        "AssetTurnover":     {"slope":"low","autocorr_lag1":"low", **{f:"high" for f in ["volatility","coef_var","delta_max","delta_mean","skew","kurtosis"]}},
        "CashFlow_to_Profit":{"slope":"low","autocorr_lag1":"low", **{f:"high" for f in ["volatility","coef_var","delta_max","delta_mean","skew","kurtosis"]}},
        "CashFlow_to_Debt":  {"slope":"high","autocorr_lag1":"low",**{f:"high" for f in ["volatility","coef_var","delta_max","delta_mean","skew","kurtosis"]}},
        "DebtEquity":        {"slope":"high","autocorr_lag1":"low",**{f:"high" for f in ["volatility","coef_var","delta_max","delta_mean","skew","kurtosis"]}},
        "CurrentRatio":      {"slope":"low","autocorr_lag1":"low", **{f:"high" for f in ["volatility","coef_var","delta_max","delta_mean","skew","kurtosis"]}}
    }
    for m, feats in directions.items():
        mask = df_score["Metric"]==m
        sub  = df_score[mask]
        for f,d in feats.items():
            thr = sub[f].quantile(0.75 if d=="high" else 0.25)
            df_score.loc[mask, f+"_score"] = ((sub[f]>thr) if d=="high" else (sub[f]<thr)).astype(int).values

    score_cols = [c for c in df_score.columns if c.endswith("_score")]
    cp_scores  = df_score.groupby("CP")[score_cols].sum().sum(axis=1).to_frame("score_total")
    cp_scores  = cp_scores.join(cluster_lbl.rename("cluster_score"))
    cp_scores["score_total"] += cp_scores["cluster_score"]
    l_th, h_th = np.quantile(cp_scores["score_total"], [0.33, 0.66])
    cp_scores["risk_label"] = cp_scores["score_total"].apply(
        lambda x: "Thấp" if x<=l_th else ("Trung bình" if x<=h_th else "Cao")
    )
    cp_scores = cp_scores.reset_index()
    cp_scores["CP"] = cp_scores["CP"].astype(str)

        # --- 9. Chuẩn bị merged và ép kiểu ---
    merged = pd.DataFrame({
        "CP": recs,
        "Similarity": sims_df.loc[ticker, recs].round(4)
    })

    # Ép CP trong merged và meta về uppercase, strip()
    merged["CP"] = merged["CP"].astype(str).str.upper().str.strip()
    meta["CP"]    = meta["CP"].astype(str).str.upper().str.strip()

    # Check xem có mã nào thiếu meta không
    miss = set(merged["CP"]) - set(meta["CP"])
    if miss:
        st.warning(f"Không tìm thấy metadata cho: {', '.join(sorted(miss))}")

    # Merge risk_label
    merged = merged.merge(
        cp_scores[["CP","risk_label"]].assign(CP=lambda df: df["CP"].str.upper().str.strip()),
        on="CP", how="left"
    )

    # Đánh dấu UNI/SRA
    merged.loc[merged["CP"].isin(["UNI","SRA"]), "risk_label"] = "Rủi ro không thể lường trước"

    # Merge metadata
    merged = merged.merge(
        meta[["CP","Công ty","Ngành","Sàn","Khối lượng NY/ĐKGD"]],
        on="CP", how="left"
    )
    st.markdown("### 📋 Thông tin chi tiết cho CP “%s”" % ticker)

    info_meta = meta[meta["CP"] == ticker]
    if not info_meta.empty:
        st.subheader("Thông tin công ty")
        # chuyển sang DataFrame 2 cột
        info_dict = info_meta.iloc[0].to_dict()
        df_info = (
            pd.DataFrame.from_dict(info_dict, orient="index", columns=["Giá trị"])
              .reset_index()
              .rename(columns={"index": "Trường"})
        )
        st.table(df_info)
    else:
        st.warning("Không tìm thấy metadata cho CP này.")

    # --- 10. Hiển thị ---
    final = merged[[
        "CP","Công ty","Ngành","Sàn","Khối lượng NY/ĐKGD",
        "Similarity","risk_label"
    ]]
    st.dataframe(final, use_container_width=True)

    # --- 2. Hiển thị Dự báo 2025 ---
    st.subheader("Dự báo 2025")

    fc_sel = (
        fc[fc["CP"] == ticker]
        .copy()
        .sort_values("ds")
        .loc[:, ["Metric", "ds", "Forecast"]]
    )

    if not fc_sel.empty:
        # tạo cột Quý (ví dụ “2025Q1”)
        fc_sel["Quarter"] = fc_sel["ds"].dt.to_period("Q").astype(str)
        # pivot
        fc_pivot = (
            fc_sel.pivot_table(
                index="Metric",
                columns="Quarter",
                values="Forecast",
                aggfunc="mean"
            )
            .round(2)
            .fillna("-")
        )
        st.table(fc_pivot)
    else:
        st.info("Không có dữ liệu forecast cho CP này.")



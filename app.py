import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import io
from pathlib import Path
import os
import altair as alt

# ==========================================
# 1. 介面與環境設定
# ==========================================
st.set_page_config(page_title="馬達預警系統", layout="wide")

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container {padding-top: 2rem;}
    html, body, [class*="st-"] { font-size: 18px !important; }
    h1 { font-size: 42px !important; font-weight: bold !important; color: #111111 !important; }
    h2 { font-size: 36px !important; font-weight: bold !important; color: #222222 !important; }
    h3 { font-size: 28px !important; font-weight: bold !important; color: #333333 !important; }
    [data-testid="stMetricValue"] { font-size: 42px !important; font-weight: bold !important; }
    [data-testid="stMetricLabel"] { font-size: 22px !important; font-weight: bold !important; color: #333333 !important; }
    [data-testid="stDataFrame"] div[role="gridcell"], 
    [data-testid="stDataFrame"] div[role="columnheader"] { text-align: left !important; justify-content: flex-start !important; }
    </style>
""", unsafe_allow_html=True)

# 雲端自動抓取當前路徑
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "lstm_attention_best_search.pth"
FEATURE_SCALER_PATH = BASE_DIR / "feature_scaler.pkl"
TARGET_SCALER_PATH = BASE_DIR / "target_scaler.pkl"

SEQUENCE_LENGTH = 120
SMOOTHING_WINDOW = 30
NOMINAL_LIFESPAN = 7200.0  

# 模型嚴格需要的 9 個特徵
MODEL_FEATURES = ["RPM", "Vm", "CH1", "CH2", "dB", "Im", "Time_Step", "Vm_std", "dB_std"]

COLUMN_RENAME_MAP = {
    "Time": "時間序列", 
    "Time_Step": "已運行時間(s)",
    "RPM": "回轉速(RPM)", 
    "Vm": "電壓(Vm)",
    "CH1": "驅動器溫度(℃)", 
    "CH2": "馬達溫度(℃)",
    "dB": "音量(dB)", 
    "Im": "電流(Im)"
}

# ==========================================
# 2. 定義模型架構 
# ==========================================
class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_outputs):
        attn_weights = self.attention(lstm_outputs)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context_vector = torch.sum(attn_weights * lstm_outputs, dim=1)
        return context_vector

class RUL_LSTM_Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim=1):
        super(RUL_LSTM_Attention, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.attention = AttentionLayer(hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.attention(out)
        out = self.fc(out)
        return out

# ==========================================
# 3. 載入模型與縮放器
# ==========================================
@st.cache_resource
def load_artifacts():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        f_scaler = joblib.load(FEATURE_SCALER_PATH)
        t_scaler = joblib.load(TARGET_SCALER_PATH)
        
        scaler_dim = getattr(f_scaler, 'n_features_in_', 9)
        
        model = RUL_LSTM_Attention(input_dim=9, hidden_dim=128, num_layers=2).to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        
        return model, f_scaler, t_scaler, device, scaler_dim
    except Exception as e:
        st.error(f"模型或縮放器載入失敗: {e}")
        st.stop()

model, feature_scaler, target_scaler, device, SCALER_DIM = load_artifacts()

# ==========================================
# 4. 後台邏輯處理區
# ==========================================
def process_single_input(rpm, vm, ch1, ch2, db, im, current_time_step):
    data = {
        "RPM": [rpm] * SEQUENCE_LENGTH,
        "Vm": [vm] * SEQUENCE_LENGTH,
        "CH1": [ch1] * SEQUENCE_LENGTH,
        "CH2": [ch2] * SEQUENCE_LENGTH,
        "dB": [db] * SEQUENCE_LENGTH,
        "Im": [im] * SEQUENCE_LENGTH,
    }
    df = pd.DataFrame(data)
    
    start_step = max(1, current_time_step - SEQUENCE_LENGTH + 1)
    df["Time_Step"] = np.arange(start_step, start_step + SEQUENCE_LENGTH)
    
    for col in ["RPM", "Vm", "CH1", "CH2", "dB", "Im"]:
        df[col] = df[col].ewm(span=SMOOTHING_WINDOW, adjust=False).mean()
            
    df["Vm_std"] = 0.0
    df["dB_std"] = 0.0
            
    df_final = df[MODEL_FEATURES]
    
    try:
        scaled_features = feature_scaler.transform(df_final.values)
    except ValueError:
        pad_df = np.zeros((len(df_final), SCALER_DIM))
        pad_df[:, :9] = df_final.values
        scaled_features = feature_scaler.transform(pad_df)[:, :9]
        
    tensor_x = torch.FloatTensor(scaled_features).unsqueeze(0)
    return tensor_x

def process_batch_input(df):
    req_cols = ["RPM", "Vm", "CH1", "CH2", "dB", "Im"]
    missing = [c for c in req_cols if c not in df.columns]
    if missing:
        return None, f"上傳的檔案缺少必要欄位: {', '.join(missing)}"
    
    df_processed = df.copy()
    
    for col in ["Vm", "dB"]:
        if f"{col}_std" not in df_processed.columns:
            df_processed[f"{col}_std"] = 0.0

    for col in req_cols:
        df_processed[col] = df_processed[col].rolling(window=SMOOTHING_WINDOW, min_periods=1, center=False).mean()
        
    df_final = df_processed[MODEL_FEATURES]
    
    try:
        scaled_features = feature_scaler.transform(df_final.values)
    except ValueError:
        pad_df = np.zeros((len(df_final), SCALER_DIM))
        pad_df[:, :9] = df_final.values
        scaled_features = feature_scaler.transform(pad_df)[:, :9]
    
    if len(scaled_features) < SEQUENCE_LENGTH:
        return None, f"資料筆數太少，模型需要至少 {SEQUENCE_LENGTH} 筆連續資料才能進行預測。"

    windows = []
    for i in range(len(scaled_features) - SEQUENCE_LENGTH + 1):
        windows.append(scaled_features[i : i + SEQUENCE_LENGTH])
        
    tensor_x = torch.FloatTensor(np.array(windows)).to(device)
    return tensor_x, None

def highlight_health_status(val):
    if val == "危險": return 'background-color: #dc3545; color: #ffffff; font-weight: bold;'
    elif val == "注意": return 'background-color: #ffc107; color: #000000; font-weight: bold;'
    elif val == "良好": return 'background-color: #28a745; color: #ffffff;'
    return ''

# ==========================================
# 5. UI 介面設計
# ==========================================

st.title("馬達狀態預測系統")

if SCALER_DIM != 9:
    st.error(f"🚨 **嚴重警告：縮放器檔案錯誤！**\n\n系統偵測到您上傳至 GitHub 的 `feature_scaler.pkl` 包含了 **{SCALER_DIM} 個特徵**，但您的模型是用 **9 個特徵** 訓練的。請上傳正確的檔案覆蓋。")

st.markdown("請選擇「實時單筆預測」或是「批次檔案上傳」來評估馬達的剩餘使用壽命 (RUL) 與健康指數。")
st.markdown("---")

tab1, tab2 = st.tabs(["實時單筆參數預測", "批次檔案上傳分析"])

# ================== 第一頁：單筆實時預測 ==================
with tab1:
    st.subheader("實時運行參數輸入 (Real-Time Operating Parameters)")
    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            in_rpm = st.number_input("轉速 (RPM)", value=3000.0)
            in_ch1 = st.number_input("驅動器溫度 (℃)", value=42.5)
        with col2:
            in_vm = st.number_input("馬達電壓 (Vm)", value=24.1)
            in_ch2 = st.number_input("馬達溫度 (℃)", value=58.2)
        with col3:
            in_load_pct = st.number_input("馬達負載電流率 (%)", value=30.0, min_value=0.0, max_value=200.0, step=1.0)
            in_db = st.number_input("音量 (dB)", value=68.5)

    st.markdown("<br>", unsafe_allow_html=True) 
    in_time = st.number_input("機台已運行時間 (s)", value=3450, step=100)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("啟動 單筆預測核心 (Run Prediction)", use_container_width=True):
        with st.spinner("推論中..."):
            actual_im = 4.2 * (in_load_pct / 100.0)
            
            try:
                # ★ 新增規則：如果轉速為 0，剩餘壽命直接判定為 0
                if in_rpm <= 0.0:
                    final_rul = 0.0
                else:
                    input_tensor = process_single_input(in_rpm, in_vm, in_ch1, in_ch2, in_db, actual_im, in_time).to(device)
                    
                    with torch.no_grad():
                        pred_scaled = model(input_tensor).cpu().numpy()
                    
                    pred_raw = target_scaler.inverse_transform(pred_scaled).item()
                    final_rul = max(0.0, float(pred_raw))
                
                health_pct = min(100.0, max(0.0, (final_rul / NOMINAL_LIFESPAN) * 100))
                final_rul_mins = final_rul / 60.0
                
                st.success("運算完成！Connection Status: Connected")
                
                st.markdown("---")
                st.subheader("預測結果 (Prediction Core)")
                res_left, res_right = st.columns([1, 1])
                
                with res_left:
                    st.markdown("##### 壽命預測數值")
                    final_value_str = f"{int(final_rul)} 秒 (約 {int(final_rul_mins)} 分鐘)"
                    
                    if health_pct > 50:
                        st.metric(label="預估剩餘壽命 (RUL)", value=final_value_str, delta="馬達健康正常", delta_color="normal")
                    elif health_pct > 20:
                        st.metric(label="預估剩餘壽命 (RUL)", value=final_value_str, delta="請準備排修", delta_color="off")
                    else:
                        st.metric(label="預估剩餘壽命 (RUL)", value=final_value_str, delta="即將停機", delta_color="inverse")

                with res_right:
                    st.markdown("##### 馬達健康指數 (Health Index)")
                    if health_pct > 50:
                        bar_color = "#28a745" ; status_text = "良好 (Good)"
                    elif health_pct > 20:
                        bar_color = "#ffc107" ; status_text = "注意 (Warning)"
                    else:
                        bar_color = "#dc3545" ; status_text = "危險 (Critical)"
                        
                    health_html = f"""
                    <div style="background-color: #f8f9fa; border-radius: 8px; width: 100%; height: 40px; margin-top: 15px; border: 1px solid #ced4da;">
                        <div style="background-color: {bar_color}; width: {health_pct}%; height: 100%; border-radius: 8px; display: flex; align-items: center; justify-content: center; color: {'black' if health_pct > 20 and health_pct <=50 else 'white'}; font-weight: bold; font-size: 18px; transition: width 0.5s ease-in-out;">
                            {int(health_pct)}%
                        </div>
                    </div>
                    <div style="margin-top: 10px; font-size: 18px; font-weight: 500;">
                        狀態：<span style="color: {bar_color};">{status_text}</span>
                    </div>
                    """
                    st.markdown(health_html, unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    if health_pct > 50:
                        st.info("系統建議：參數均在正常範圍內，請維持日常保養週期。")
                    elif health_pct > 20:
                        st.warning("系統建議：馬達已進入退化期，請通知維修部門備料，並規劃近期保養排程。")
                    else:
                        st.error("系統建議：剩餘壽命已低於安全閾值！強烈建議立即安排停機檢查，避免無預警當機影響產線。")
            
            except Exception as e:
                st.error(f"運算發生錯誤: {e}")

# ================== 第二頁：批次檔案上傳 ==================
with tab2:
    st.subheader("上傳歷史數據進行趨勢分析")
    st.markdown("支援 CSV 或 Excel (xlsx) 格式。檔案必須包含 `RPM`, `Vm`, `CH1`, `CH2`, `dB`, `Im` 等核心特徵欄位。")
    
    uploaded_file = st.file_uploader("選擇數據檔案", type=['csv', 'xlsx'])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df_upload = pd.read_csv(uploaded_file)
            else:
                df_upload = pd.read_excel(uploaded_file)
                
            if "Time_Step" not in df_upload.columns:
                if "Time" in df_upload.columns:
                    df_upload["Time_Step"] = df_upload["Time"]
                else:
                    df_upload["Time_Step"] = np.arange(1, len(df_upload) + 1)
                
            df_preview = df_upload.head().rename(columns=COLUMN_RENAME_MAP)
            df_preview = df_preview.loc[:, ~df_preview.columns.duplicated()]
            preview_cols = [c for c in df_preview.columns if not str(c).endswith("_std") and c != "時間序列"]
            
            st.write("預覽上傳數據的前 5 筆：")
            st.dataframe(df_preview[preview_cols], hide_index=True, use_container_width=True)
            
            if st.button("開始批次趨勢預測", use_container_width=True):
                with st.spinner("模型分析中，請稍候..."):
                    result = process_batch_input(df_upload)
                    if result[0] is None:
                        st.error(result[1])
                    else:
                        tensor_x = result[0]
                        with torch.no_grad():
                            preds_scaled = model(tensor_x).cpu().numpy()
                            
                        preds_raw = target_scaler.inverse_transform(preds_scaled).flatten()
                        preds_final = np.clip(preds_raw, 0.0, None)
                        
                        pad_length = SEQUENCE_LENGTH - 1
                        if len(preds_final) > 0:
                            first_valid_pred = preds_final[0]
                            pad_array = np.arange(first_valid_pred + pad_length, first_valid_pred, -1)
                        else:
                            pad_array = np.zeros(pad_length)
                        
                        full_predictions = np.concatenate([pad_array, preds_final])
                        df_upload["預測 RUL (s)"] = np.round(full_predictions).astype(int)
                        
                        # ★ 新增規則：強制將所有原始轉速 <= 0 的資料點，剩餘壽命壓制為 0
                        if "RPM" in df_upload.columns:
                            df_upload.loc[df_upload["RPM"] <= 0, "預測 RUL (s)"] = 0
                        
                        health_pcts = np.clip((df_upload["預測 RUL (s)"].values / NOMINAL_LIFESPAN) * 100, 0, 100)
                        def assign_status(pct):
                            if pct > 50: return "良好"
                            elif pct > 20: return "注意"
                            else: return "危險"
                        df_upload["健康等級"] = [assign_status(p) for p in health_pcts]
                        
                        st.markdown("### 檔案末端最新狀態評估 (目前馬達狀態)")
                        # 取出壓制為 0 之後的最後一筆壽命
                        last_rul = df_upload["預測 RUL (s)"].iloc[-1]
                        last_health_pct = min(100.0, max(0.0, (last_rul / NOMINAL_LIFESPAN) * 100))
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric(label="最新時間點預估 RUL", value=f"{int(last_rul)} 秒")
                        with col_b:
                            if last_health_pct > 50:
                                st.metric(label="健康指數", value=f"{int(last_health_pct)}%", delta="狀態良好", delta_color="normal")
                            elif last_health_pct > 20:
                                st.metric(label="健康指數", value=f"{int(last_health_pct)}%", delta="建議排修", delta_color="off")
                            else:
                                st.metric(label="健康指數", value=f"{int(last_health_pct)}%", delta="危險警報", delta_color="inverse")

                        st.markdown("### 批次預測結果數據")
                        df_display = df_upload.rename(columns=COLUMN_RENAME_MAP)
                        df_display = df_display.loc[:, ~df_display.columns.duplicated()]
                        cols_to_show = [c for c in df_display.columns if not str(c).endswith("_std") and c != "時間序列"]
                        df_display = df_display[cols_to_show]
                        
                        try:
                            styled_df = df_display.style.set_properties(**{'font-size': '16px', 'text-align': 'left'}).map(highlight_health_status, subset=["健康等級"])
                        except AttributeError:
                            styled_df = df_display.style.set_properties(**{'font-size': '16px', 'text-align': 'left'}).applymap(highlight_health_status, subset=["健康等級"])
                            
                        st.dataframe(styled_df, hide_index=True, use_container_width=True)
                        
                        st.markdown("### 下載預測報告")
                        csv_data = df_display.to_csv(index=False).encode('utf-8-sig') 
                        st.download_button(
                            label="下載完整預測結果 (CSV)",
                            data=csv_data,
                            file_name="Motor_RUL_Predictions.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

                        st.markdown("---")
                        st.markdown("### RUL 剩餘壽命趨勢圖")
                        
                        chart_data = pd.DataFrame()
                        chart_data["預測 RUL (s)"] = df_upload["預測 RUL (s)"].values
                        if "RUL" in df_upload.columns:
                            chart_data["真實 RUL"] = df_upload["RUL"].values
                            
                        chart_data.index = df_upload["Time_Step"]
                        chart_data.index.name = "時間(s)"
                        
                        try:
                            st.line_chart(chart_data, x_label="時間(s)", y_label="剩餘使用壽命(s)")
                        except TypeError:
                            df_melt = chart_data.reset_index().melt("時間(s)", var_name="指標", value_name="剩餘使用壽命(s)")
                            chart = alt.Chart(df_melt).mark_line().encode(
                                x=alt.X("時間(s):Q", title="時間(s)"),
                                y=alt.Y("剩餘使用壽命(s):Q", title="剩餘使用壽命(s)"),
                                color=alt.Color("指標:N", title="圖例")
                            ).interactive()
                            st.altair_chart(chart, use_container_width=True)
                        
        except Exception as e:
            st.error(f"檔案讀取或處理時發生錯誤: {e}")

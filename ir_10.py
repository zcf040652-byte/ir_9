import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from sklearn.metrics import r2_score
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tempfile
import re
import io
import zipfile  # 新增：用于打包文件

# ==========================================
# 0. 页面全局配置
# ==========================================
st.set_page_config(page_title="原位红外水峰拟合", layout="wide")

# 初始化 Session State
if 'fit_df' not in st.session_state: st.session_state['fit_df'] = None
if 'raw_spectra' not in st.session_state: st.session_state['raw_spectra'] = None
if 'wavenumbers' not in st.session_state: st.session_state['wavenumbers'] = None
if 'peak_colors' not in st.session_state: st.session_state['peak_colors'] = []
if 'batch_results' not in st.session_state: st.session_state['batch_results'] = None
if 'fit_details' not in st.session_state: st.session_state['fit_details'] = None
if 'last_popt' not in st.session_state: st.session_state['last_popt'] = None
if 'is_sigma_locked' not in st.session_state: st.session_state['is_sigma_locked'] = False
if 'svg_zip_data' not in st.session_state: st.session_state['svg_zip_data'] = None
if 'all_plots_csv' not in st.session_state: st.session_state['all_plots_csv'] = None  # 新增：缓存全谱作图数据


# ==========================================
# 1. 核心算法
# ==========================================

def pseudo_voigt_fn(x, amp, center, sigma, eta):
    """ Pseudo-Voigt 函数 """
    sigma = np.maximum(sigma, 1e-5)  # 保护除零
    L = 1 / (1 + ((x - center) / sigma) ** 2)
    G = np.exp(-np.log(2) * ((x - center) / sigma) ** 2)
    return amp * (eta * L + (1 - eta) * G)


def multi_peak_model(x, *params):
    """ 多峰叠加模型 (标准全参数) """
    y = np.zeros_like(x)
    n_peaks = len(params) // 4
    for i in range(n_peaks):
        a = params[i * 4]
        c = params[i * 4 + 1]
        s = params[i * 4 + 2]
        e = params[i * 4 + 3]
        y += pseudo_voigt_fn(x, a, c, s, e)
    return y


def constrained_multi_peak_model(x, *reduced_params):
    """
    受限多峰模型: P1, P2, P3 共享 Sigma
    """
    full_params = []
    p1 = reduced_params[0:4]
    full_params.extend(p1)
    sigma_shared = p1[2]

    current_idx = 4
    num_params = len(reduced_params)

    # P2
    if current_idx + 3 <= num_params:
        p2_partial = reduced_params[current_idx: current_idx + 3]
        full_params.extend([p2_partial[0], p2_partial[1], sigma_shared, p2_partial[2]])
        current_idx += 3

    # P3
    if current_idx + 3 <= num_params:
        p3_partial = reduced_params[current_idx: current_idx + 3]
        full_params.extend([p3_partial[0], p3_partial[1], sigma_shared, p3_partial[2]])
        current_idx += 3

    # P4+
    while current_idx < num_params:
        full_params.extend(reduced_params[current_idx: current_idx + 4])
        current_idx += 4

    return multi_peak_model(x, *full_params)


def params_full_to_reduced(full_params, n_peaks):
    if n_peaks < 3: return full_params
    reduced = []
    reduced.extend(full_params[0:4])
    reduced.extend([full_params[4], full_params[5], full_params[7]])
    reduced.extend([full_params[8], full_params[9], full_params[11]])
    reduced.extend(full_params[12:])
    return reduced


def params_reduced_to_full(reduced_params, n_peaks):
    if n_peaks < 3: return reduced_params
    full = []
    p1 = reduced_params[0:4]
    full.extend(p1)
    sigma_shared = p1[2]
    idx = 4
    full.extend([reduced_params[idx], reduced_params[idx + 1], sigma_shared, reduced_params[idx + 2]])
    idx += 3
    full.extend([reduced_params[idx], reduced_params[idx + 1], sigma_shared, reduced_params[idx + 2]])
    idx += 3
    full.extend(reduced_params[idx:])
    return full


def auto_guess_parameters(x, y, n_peaks=3):
    if n_peaks < 1: n_peaks = 1
    if np.max(y) == np.min(y):
        guess = []
        for i in range(n_peaks): guess += [0.01, np.mean(x), 10, 0.5]
        return guess

    chunks = np.array_split(np.column_stack((x, y)), n_peaks)
    guess = []
    for chunk in chunks:
        if len(chunk) > 0:
            max_idx = np.argmax(chunk[:, 1])
            peak_x = chunk[max_idx, 0]
            peak_y = chunk[max_idx, 1]
            guess += [peak_y, peak_x, 15, 0.5]
        else:
            guess += [0.01, np.mean(x), 15, 0.5]
    return guess


def calculate_peak_area(x, amp, center, sigma, eta):
    y_vals = pseudo_voigt_fn(x, amp, center, sigma, eta)
    return np.trapz(y_vals, x)


def subtract_linear_baseline(x, y):
    if len(x) < 2:
        return y, np.zeros_like(y)
    x0, y0 = x[0], y[0]
    x1, y1 = x[-1], y[-1]
    slope = (y1 - y0) / (x1 - x0)
    intercept = y0 - slope * x0
    baseline = slope * x + intercept
    y_corrected = y - baseline
    return y_corrected, baseline


# ==========================================
# 2. 侧边栏设置
# ==========================================
st.title("Cu-CO2RR 原位光谱拟合工具 V26 (Sigma Lock + SVG/Data Export)")

with st.sidebar:
    st.header("1. 数据导入")
    uploaded_file = st.file_uploader("上传 Excel/CSV (每两列为一组: Wavenumber, Abs)", type=["xlsx", "xls", "csv"])

    st.markdown("---")
    st.header("2. 拟合参数设置")
    n_peaks = st.slider("拟合峰数量 (Peaks)", 1, 6, 2)

    lock_sigma = False
    if n_peaks >= 3:
        lock_sigma = st.checkbox("🔒 锁定 P1-P3 半峰宽 (Lock Sigma)", value=False,
                                 help="强制 Peak 1, 2, 3 使用相同的半峰宽 (Sigma)")

    st.subheader("分峰颜色 (Peak Colors)")
    default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    current_colors = []
    cols_color = st.columns(2)
    for i in range(n_peaks):
        with cols_color[i % 2]:
            c = st.color_picker(f"Peak {i + 1}", value=default_colors[i % len(default_colors)])
            current_colors.append(c)
    st.session_state['peak_colors'] = current_colors

    st.subheader("截断范围 (Fitting Range)")
    min_w_default = 0.0
    max_w_default = 4000.0
    if st.session_state['wavenumbers'] is not None:
        min_w_default = float(st.session_state['wavenumbers'].min())
        max_w_default = float(st.session_state['wavenumbers'].max())

    fit_min = st.number_input("Min Wavenumber", value=min_w_default)
    fit_max = st.number_input("Max Wavenumber", value=max_w_default)

    st.subheader("基线与校正")
    use_linear_baseline = st.checkbox("✅ 启用线性基线扣除", value=True)
    sg_window = st.slider("平滑窗口 (Savitzky-Golay)", 5, 51, 11, step=2)
    sg_poly = st.slider("多项式阶数", 1, 5, 2)

    st.markdown("---")
    st.header("3. 图表样式设置 (Style)")
    font_family = st.selectbox("字体 (Font Family)", ["Arial", "Times New Roman", "Helvetica", "Calibri"], index=0)
    font_size = st.number_input("字体大小 (Font Size)", value=14, step=1)
    axis_width = st.number_input("坐标轴线宽 (Frame Width px)", value=1.5, step=0.5, min_value=0.5)

    col_style_1, col_style_2 = st.columns(2)
    with col_style_1:
        fit_line_color = st.color_picker("拟合线颜色 (Fit Line)", "#000000")
    with col_style_2:
        data_point_color = st.color_picker("数据点颜色 (Data)", "#808080")
    data_point_size = st.slider("数据点大小 (Data Size)", 1, 15, 4)

    st.markdown("---")
    st.header("4. 坐标轴范围设置 (Axis Limits)")
    col_lim_1, col_lim_2 = st.columns(2)
    with col_lim_1:
        custom_x_min = st.number_input("X Min", value=None, placeholder="Auto", step=10.0)
        custom_y_min = st.number_input("Y Min (Abs)", value=None, placeholder="Auto", step=0.01)
    with col_lim_2:
        custom_x_max = st.number_input("X Max", value=None, placeholder="Auto", step=10.0)
        custom_y_max = st.number_input("Y Max (Abs)", value=None, placeholder="Auto", step=0.01)


def apply_nature_style(fig, font_fam, font_sz, ax_width, legend_inside=True):
    if legend_inside:
        legend_cfg = dict(
            x=0.02, y=0.98, xanchor='left', yanchor='top',
            bgcolor='rgba(255,255,255,0)', borderwidth=0,
            font=dict(size=font_sz - 4, color='black')
        )
    else:
        legend_cfg = dict(font=dict(size=font_sz, color='black'))

    fig.update_layout(
        font=dict(family=font_fam, size=font_sz, color='black'),
        plot_bgcolor='white', paper_bgcolor='white',
        margin=dict(l=60, r=20, t=50, b=60),
        showlegend=True, legend=legend_cfg,
        xaxis=dict(
            showline=True, linewidth=ax_width, linecolor='black', mirror=True,
            ticks='inside', tickwidth=ax_width, tickcolor='black', ticklen=6,
            showgrid=False, zeroline=False,
            title_font=dict(color='black'), tickfont=dict(color='black')
        ),
        yaxis=dict(
            showline=True, linewidth=ax_width, linecolor='black', mirror=True,
            ticks='inside', tickwidth=ax_width, tickcolor='black', ticklen=6,
            showgrid=False, zeroline=False,
            title_font=dict(color='black'), tickfont=dict(color='black')
        )
    )

    new_x_range = [None, None]
    new_y_range = [None, None]
    if custom_x_min is not None: new_x_range[0] = custom_x_min
    if custom_x_max is not None: new_x_range[1] = custom_x_max
    if any(x is not None for x in new_x_range): fig.update_xaxes(range=new_x_range)

    if custom_y_min is not None: new_y_range[0] = custom_y_min
    if custom_y_max is not None: new_y_range[1] = custom_y_max
    if any(y is not None for y in new_y_range): fig.update_yaxes(range=new_y_range)


# ==========================================
# 3. 数据处理
# ==========================================
if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            df_raw = pd.read_csv(uploaded_file, header=None)
        else:
            df_raw = pd.read_excel(uploaded_file, header=None)

        label_row_idx = 2
        data_start_idx = 3

        wavenumbers_base = pd.to_numeric(df_raw.iloc[data_start_idx:, 0], errors='coerce').values
        ocp_abs = pd.to_numeric(df_raw.iloc[data_start_idx:, 1], errors='coerce').values

        valid_mask = ~np.isnan(wavenumbers_base) & ~np.isnan(ocp_abs)
        wavenumbers_base = wavenumbers_base[valid_mask]
        ocp_abs = ocp_abs[valid_mask]

        processed_data = {}
        voltage_map = {}

        num_cols = df_raw.shape[1]

        for i in range(2, num_cols, 2):
            if i + 1 >= num_cols: break

            raw_label = str(df_raw.iloc[label_row_idx, i + 1]).strip()
            final_label = raw_label
            sort_val = -9999

            try:
                clean_str = re.sub(r'[^\d\.\-]', '', raw_label)
                val = float(clean_str)
                final_label = f"{val} V"
                sort_val = val
            except:
                if 'ocp' in raw_label.lower():
                    final_label = "OCP"
                else:
                    final_label = raw_label

            curr_abs = pd.to_numeric(df_raw.iloc[data_start_idx:, i + 1], errors='coerce').values
            curr_abs = curr_abs[valid_mask]

            if len(curr_abs) != len(ocp_abs):
                continue

            corrected_abs = curr_abs - ocp_abs
            processed_data[final_label] = corrected_abs
            voltage_map[final_label] = sort_val

        df_spectra = pd.DataFrame(processed_data, index=wavenumbers_base)
        df_spectra.index.name = "Wavenumber"

        sorted_cols = sorted(df_spectra.columns, key=lambda x: voltage_map.get(x, -9999), reverse=True)
        df_spectra = df_spectra[sorted_cols]
        df_spectra.sort_index(inplace=True)

        st.session_state['raw_spectra'] = df_spectra
        st.session_state['wavenumbers'] = df_spectra.index.values

        if fit_min == 0 and fit_max == 4000:
            st.session_state['fit_min'] = df_spectra.index.min()
            st.session_state['fit_max'] = df_spectra.index.max()

        with st.expander("数据预览 (已扣除 OCP & 排序 0.9V to 0.45V)", expanded=False):
            st.write(f"检测到 {len(df_spectra.columns)} 个有效电位。")
            st.dataframe(df_spectra.head())

    except Exception as e:
        st.error(f"文件处理出错: {e}")
        st.stop()
else:
    st.info("请在左侧上传数据文件。")
    st.stop()

# ==========================================
# 4. 主界面
# ==========================================

if st.session_state['raw_spectra'] is not None:
    df = st.session_state['raw_spectra']

    st.header("3. 拟合控制与分析")

    col1, col2 = st.columns([3, 1])

    # 参数控制区 (放在右侧 col2)
    with col2:
        st.subheader("Step 1: 参数微调")
        target_col = st.selectbox("选择基准电位", df.columns, index=0)

        mask = (df.index >= fit_min) & (df.index <= fit_max)
        x_data = df.index[mask].to_numpy()
        y_raw = df[target_col].values[mask]

        if len(y_raw) > sg_window:
            y_smoothed = savgol_filter(y_raw, sg_window, sg_poly)
        else:
            y_smoothed = y_raw

        if use_linear_baseline:
            y_data, baseline_curve = subtract_linear_baseline(x_data, y_smoothed)
        else:
            y_data = y_smoothed
            baseline_curve = np.zeros_like(y_data)

        guess_params = auto_guess_parameters(x_data, y_data, n_peaks)
        current_params = []
        bounds_lower = []
        bounds_upper = []

        shared_sigma_val = 10.0

        for i in range(n_peaks):
            with st.expander(f"Peak {i + 1}", expanded=True):
                st.markdown(f"Color: **{current_colors[i]}**")

                amp = st.number_input(f"Amp {i + 1}", value=float(guess_params[i * 4]), step=0.0001, format="%.5f")
                cen = st.number_input(f"Center {i + 1}", value=float(guess_params[i * 4 + 1]), step=1.0)

                if lock_sigma and i in [1, 2]:
                    st.info(f"Sigma locked to P1: {shared_sigma_val:.3f}")
                    sig = shared_sigma_val
                else:
                    sig = st.number_input(f"Sigma {i + 1}", value=float(guess_params[i * 4 + 2]), step=0.5,
                                          min_value=0.1)
                    if i == 0: shared_sigma_val = sig

                eta = st.slider(f"Eta {i + 1}", 0.0, 1.0, float(guess_params[i * 4 + 3]), step=0.1)

            current_params.extend([amp, cen, sig, eta])
            bounds_lower.extend([0, fit_min - 50, 0.1, 0])
            bounds_upper.extend([np.inf, fit_max + 50, 200, 1])

        # 执行单帧拟合预览
        try:
            if lock_sigma and n_peaks >= 3:
                p0_reduced = params_full_to_reduced(current_params, n_peaks)
                bounds_l_red = []
                bounds_u_red = []

                bounds_l_red.extend([0, fit_min - 50, 0.1, 0])
                bounds_u_red.extend([np.inf, fit_max + 50, 200, 1])
                bounds_l_red.extend([0, fit_min - 50, 0])
                bounds_u_red.extend([np.inf, fit_max + 50, 1])
                bounds_l_red.extend([0, fit_min - 50, 0])
                bounds_u_red.extend([np.inf, fit_max + 50, 1])
                for k in range(3, n_peaks):
                    bounds_l_red.extend([0, fit_min - 50, 0.1, 0])
                    bounds_u_red.extend([np.inf, fit_max + 50, 200, 1])

                popt_reduced, _ = curve_fit(
                    constrained_multi_peak_model, x_data, y_data,
                    p0=p0_reduced, bounds=(bounds_l_red, bounds_u_red), maxfev=5000
                )
                st.session_state['last_popt'] = popt_reduced
                st.session_state['is_sigma_locked'] = True
                popt = params_reduced_to_full(popt_reduced, n_peaks)
            else:
                popt, pcov = curve_fit(
                    multi_peak_model, x_data, y_data,
                    p0=current_params, bounds=(bounds_lower, bounds_upper), maxfev=5000
                )
                st.session_state['last_popt'] = popt
                st.session_state['is_sigma_locked'] = False
        except Exception as e:
            popt = current_params

        st.divider()
        st.subheader("Step 2: 批量与导出")

        # --- 批量拟合按钮 ---
        if st.button("🚀 批量拟合所有电位", type="primary", use_container_width=True):
            if 'last_popt' not in st.session_state:
                st.error("无法获取初始参数")
            else:
                with st.spinner("正在处理..."):
                    initial_popt = st.session_state['last_popt']
                    is_locked = st.session_state['is_sigma_locked']

                    if is_locked and n_peaks >= 3:
                        bounds_l_batch = []
                        bounds_u_batch = []
                        bounds_l_batch.extend([0, fit_min - 50, 0.1, 0])
                        bounds_u_batch.extend([np.inf, fit_max + 50, 200, 1])
                        bounds_l_batch.extend([0, fit_min - 50, 0])
                        bounds_u_batch.extend([np.inf, fit_max + 50, 1])
                        bounds_l_batch.extend([0, fit_min - 50, 0])
                        bounds_u_batch.extend([np.inf, fit_max + 50, 1])
                        for k in range(3, n_peaks):
                            bounds_l_batch.extend([0, fit_min - 50, 0.1, 0])
                            bounds_u_batch.extend([np.inf, fit_max + 50, 200, 1])
                    else:
                        bounds_l_batch = bounds_lower
                        bounds_u_batch = bounds_upper

                    results_list = []
                    details_list = []

                    for col_name in df.columns:
                        y_curr_raw = df[col_name].values[mask]
                        if len(y_curr_raw) > sg_window:
                            y_curr_smooth = savgol_filter(y_curr_raw, sg_window, sg_poly)
                        else:
                            y_curr_smooth = y_curr_raw

                        if use_linear_baseline:
                            y_curr_fit, baseline_curr = subtract_linear_baseline(x_data, y_curr_smooth)
                        else:
                            y_curr_fit = y_curr_smooth
                            baseline_curr = np.zeros_like(y_curr_fit)

                        try:
                            if is_locked and n_peaks >= 3:
                                p_batch_reduced, _ = curve_fit(
                                    constrained_multi_peak_model, x_data, y_curr_fit,
                                    p0=initial_popt, bounds=(bounds_l_batch, bounds_u_batch), maxfev=5000
                                )
                                p_batch_full = params_reduced_to_full(p_batch_reduced, n_peaks)
                                y_total_fit = multi_peak_model(x_data, *p_batch_full)
                            else:
                                p_batch_full, _ = curve_fit(
                                    multi_peak_model, x_data, y_curr_fit,
                                    p0=initial_popt, bounds=(bounds_l_batch, bounds_u_batch), maxfev=5000
                                )
                                y_total_fit = multi_peak_model(x_data, *p_batch_full)

                            try:
                                v_val = float(re.sub(r'[^\d\.\-]', '', col_name))
                            except:
                                v_val = 0

                            row_res = {"Voltage": v_val, "Label": col_name}
                            r2_val = r2_score(y_curr_fit, y_total_fit)
                            row_res["R2"] = r2_val

                            frame_data = {
                                "x": x_data, "y_raw": y_curr_fit, "y_fit": y_total_fit,
                                "peaks": [], "title": col_name, "r2": r2_val,
                                "params": p_batch_full,
                                "baseline": baseline_curr  # 保存基线数据
                            }

                            for i in range(n_peaks):
                                idx = i * 4
                                amp_b, cen_b, sig_b, eta_b = p_batch_full[idx:idx + 4]
                                area_b = calculate_peak_area(x_data, amp_b, cen_b, sig_b, eta_b)
                                row_res[f"Peak{i + 1}_Center"] = cen_b
                                row_res[f"Peak{i + 1}_Area"] = area_b
                                row_res[f"Peak{i + 1}_Height"] = amp_b
                                y_comp_b = pseudo_voigt_fn(x_data, amp_b, cen_b, sig_b, eta_b)
                                frame_data["peaks"].append(y_comp_b)

                            results_list.append(row_res)
                            details_list.append(frame_data)
                        except:
                            pass

                    res_df = pd.DataFrame(results_list)
                    st.session_state['batch_results'] = res_df
                    st.session_state['fit_details'] = details_list
                    # 清除旧的缓存
                    st.session_state['svg_zip_data'] = None
                    st.session_state['all_plots_csv'] = None
                    st.toast("批量拟合完成！")

        # --- 数据下载区 ---
        if st.session_state['batch_results'] is not None:
            res_df = st.session_state['batch_results']

            # 1. 下载拟合参数汇总 (Excel/CSV)
            try:
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer) as writer:
                    res_df.to_excel(writer, sheet_name='Fitting Results', index=False)
                dl_data = buffer.getvalue()
                dl_name = "fitting_summary.xlsx"
                dl_mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            except Exception:
                buffer = io.BytesIO()
                res_df.to_csv(buffer, index=False)
                dl_data = buffer.getvalue()
                dl_name = "fitting_summary.csv"
                dl_mime = "text/csv"

            st.download_button(
                label="📥 下载拟合参数汇总 (Excel)",
                data=dl_data, file_name=dl_name, mime=dl_mime, use_container_width=True
            )

            # 2. 生成动图 (GIF)
            if st.button("🎞️ 生成动图 (GIF)", use_container_width=True):
                with st.spinner("渲染中..."):
                    frames = st.session_state['fit_details']
                    plt.rcParams['font.family'] = font_family
                    plt.rcParams['font.size'] = font_size
                    plt.rcParams['axes.linewidth'] = axis_width
                    plt.rcParams['text.color'] = 'black'
                    plt.rcParams['axes.labelcolor'] = 'black'
                    plt.rcParams['xtick.color'] = 'black'
                    plt.rcParams['ytick.color'] = 'black'

                    fig_anim, ax_anim = plt.subplots(figsize=(6, 4))
                    ax_anim.tick_params(direction='in', top=True, right=True, width=axis_width, length=4)


                    def update(frame_idx):
                        ax_anim.clear()
                        ax_anim.tick_params(direction='in', top=True, right=True, width=axis_width, length=4,
                                            colors='black')
                        for spine in ax_anim.spines.values():
                            spine.set_linewidth(axis_width)
                            spine.set_edgecolor('black')

                        data = frames[frame_idx]
                        ax_anim.scatter(data['x'], data['y_raw'], color=data_point_color, s=data_point_size, alpha=0.5,
                                        label='Data')
                        ax_anim.plot(data['x'], data['y_fit'], color=fit_line_color, linewidth=1.5, label='Fit')
                        for i, y_p in enumerate(data['peaks']):
                            ax_anim.fill_between(data['x'], y_p, alpha=0.5, color=current_colors[i], label=f'P{i + 1}')

                        ax_anim.set_title(f"Potential: {data['title']}", color='black')
                        ax_anim.set_xlabel("Wavenumber (cm$^{-1}$)", color='black')
                        ax_anim.set_ylabel("Absorbance", color='black')

                        if custom_y_min is not None and custom_y_max is not None:
                            ax_anim.set_ylim(custom_y_min, custom_y_max)
                        elif custom_y_min is not None:
                            ax_anim.set_ylim(bottom=custom_y_min)
                        elif custom_y_max is not None:
                            ax_anim.set_ylim(top=custom_y_max)

                        ax_anim.legend(loc='upper left', frameon=False, fontsize=font_size - 4)


                    ani = animation.FuncAnimation(fig_anim, update, frames=len(frames), interval=500)
                    with tempfile.NamedTemporaryFile(suffix='.gif', delete=False) as tmpfile:
                        ani.save(tmpfile.name, writer='pillow')
                        st.session_state['gif_path'] = tmpfile.name

            if 'gif_path' in st.session_state:
                with open(st.session_state['gif_path'], "rb") as f:
                    st.download_button(
                        label="📥 下载 GIF", data=f, file_name="spectra_evolution.gif", mime="image/gif",
                        use_container_width=True
                    )

            st.divider()

            # --- 新增：批量导出作图数据 (CSV) ---
            with st.expander("📊 批量导出作图数据 (CSV)", expanded=True):
                st.caption("导出包含所有电位作图数据的大表。每组包含：Data, TotalFit, Peaks..., Baseline")
                if st.button("准备全谱作图数据", use_container_width=True):
                    if st.session_state['fit_details'] is None:
                        st.error("请先执行批量拟合！")
                    else:
                        with st.spinner("正在整理数据..."):
                            details = st.session_state['fit_details']
                            # 假设所有 frame 的 x 轴是一样的（基于 batch fit 逻辑确实如此）
                            base_x = details[0]['x']

                            # 构建大字典
                            big_data = {"Wavenumber": base_x}

                            for d_data in details:
                                # 清洗 label 作为列名前缀
                                safe_label = re.sub(r'[^\w\-. ]', '_', str(d_data['title']))

                                big_data[f"{safe_label}_Data"] = d_data['y_raw']
                                big_data[f"{safe_label}_TotalFit"] = d_data['y_fit']

                                for i, yp in enumerate(d_data['peaks']):
                                    big_data[f"{safe_label}_P{i + 1}"] = yp

                                big_data[f"{safe_label}_Baseline"] = d_data['baseline']

                            export_df = pd.DataFrame(big_data)
                            st.session_state['all_plots_csv'] = export_df.to_csv(index=False).encode('utf-8')
                            st.success("数据准备就绪！")

                if st.session_state['all_plots_csv'] is not None:
                    st.download_button(
                        label="📥 下载全谱作图数据 (CSV)",
                        data=st.session_state['all_plots_csv'],
                        file_name="all_potentials_plot_data.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

            # --- 批量导出 SVG 区域 ---
            with st.expander("🖼️ 批量导出 SVG 图片", expanded=True):
                st.caption("将导出符合右侧 Plotly 风格的矢量图。需要安装 kaleido 库。")
                if st.button("开始生成 SVG 图片包 (ZIP)", use_container_width=True):
                    if st.session_state['fit_details'] is None:
                        st.error("请先执行批量拟合！")
                    else:
                        # 检查依赖
                        try:
                            import kaleido
                        except ImportError:
                            st.error("缺少依赖库: kaleido。请在终端运行: pip install -U kaleido")
                            st.stop()

                        with st.spinner("正在逐个生成图片，请稍候..."):
                            zip_buffer = io.BytesIO()
                            details = st.session_state['fit_details']
                            prog_bar = st.progress(0)

                            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                                for idx, d_data in enumerate(details):
                                    # 1. 重建 Plotly Figure 对象 (与详情回溯中的逻辑保持一致)
                                    fig_exp = go.Figure()

                                    # Data Trace
                                    fig_exp.add_trace(go.Scatter(
                                        x=d_data['x'], y=d_data['y_raw'], mode='markers', name='Data',
                                        marker=dict(color=data_point_color, size=data_point_size, opacity=0.6)
                                    ))
                                    # Total Fit Trace
                                    fig_exp.add_trace(go.Scatter(
                                        x=d_data['x'], y=d_data['y_fit'], mode='lines', name='Total Fit',
                                        line=dict(color=fit_line_color, width=2)
                                    ))
                                    # Peak Traces
                                    for i, yp in enumerate(d_data['peaks']):
                                        fig_exp.add_trace(go.Scatter(
                                            x=d_data['x'], y=yp, mode='lines', fill='tozeroy',
                                            name=f'P{i + 1}',
                                            line=dict(color=current_colors[i], width=0),
                                            fillcolor=current_colors[i], opacity=0.5
                                        ))

                                    # Layout & Style
                                    fig_exp.update_layout(
                                        title=f"Potential: {d_data['title']}",
                                        xaxis_title="Wavenumber (cm⁻¹)",
                                        yaxis_title="Absorbance"
                                    )

                                    # 应用当前侧边栏设定的全局样式和坐标轴范围
                                    apply_nature_style(fig_exp, font_family, font_size, axis_width, legend_inside=True)

                                    # 2. 导出为 SVG
                                    try:
                                        # format='svg' 产生矢量图，和相机图标一致
                                        # width/height 设定图片物理尺寸比例
                                        img_bytes = fig_exp.to_image(format="svg", width=600, height=450)

                                        # 文件名处理，去除非法字符
                                        safe_name = re.sub(r'[\\/*?:"<>|]', "_", str(d_data['title']))
                                        zf.writestr(f"{safe_name}.svg", img_bytes)
                                    except Exception as e:
                                        st.error(f"生成图片 {d_data['title']} 失败: {e}")

                                    prog_bar.progress((idx + 1) / len(details))

                            st.session_state['svg_zip_data'] = zip_buffer.getvalue()
                            st.success("打包完成！请点击下方按钮下载。")

                if st.session_state['svg_zip_data'] is not None:
                    st.download_button(
                        label="📥 下载所有 SVG (ZIP)",
                        data=st.session_state['svg_zip_data'],
                        file_name="fitting_plots_svg.zip",
                        mime="application/zip",
                        use_container_width=True
                    )

    with col1:
        # A. 单帧预览图
        y_fit = multi_peak_model(x_data, *popt)
        r2 = r2_score(y_data, y_fit)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_data, y=y_data, mode='markers', name='Data',
            marker=dict(color=data_point_color, size=data_point_size, opacity=0.6)
        ))
        fig.add_trace(go.Scatter(
            x=x_data, y=y_fit, mode='lines', name=f'Total Fit (R²={r2:.3f})',
            line=dict(color=fit_line_color, width=2)
        ))

        for i in range(n_peaks):
            idx = i * 4
            y_comp = pseudo_voigt_fn(x_data, popt[idx], popt[idx + 1], popt[idx + 2], popt[idx + 3])
            fig.add_trace(go.Scatter(
                x=x_data, y=y_comp, mode='lines', fill='tozeroy',
                name=f'P{i + 1}',
                line=dict(color=current_colors[i], width=0),
                fillcolor=current_colors[i], opacity=0.5
            ))

        fig.update_layout(title=f"当前电位拟合预览: {target_col}", xaxis_title="Wavenumber (cm⁻¹)",
                          yaxis_title="Absorbance")
        apply_nature_style(fig, font_family, font_size, axis_width, legend_inside=True)
        st.plotly_chart(fig, use_container_width=True,
                        config={'toImageButtonOptions': {'format': 'svg', 'filename': f'fit_{target_col}'}})

        # B. 趋势分析
        if st.session_state['batch_results'] is not None:
            st.divider()
            st.subheader("趋势分析 & 详情回溯")
            res_df = st.session_state['batch_results']
            details = st.session_state['fit_details']

            t_col1, t_col2 = st.columns(2)
            with t_col1:
                fig_area = go.Figure()
                for i in range(n_peaks):
                    fig_area.add_trace(go.Scatter(
                        x=res_df['Voltage'], y=res_df[f'Peak{i + 1}_Area'],
                        mode='lines+markers', name=f'P{i + 1} Area',
                        line=dict(color=current_colors[i]), marker=dict(size=8)
                    ))
                fig_area.update_layout(title="Peak Area vs. Potential", xaxis_title="V vs RHE")
                apply_nature_style(fig_area, font_family, font_size, axis_width, legend_inside=True)
                st.plotly_chart(fig_area, use_container_width=True,
                                config={'toImageButtonOptions': {'format': 'svg', 'filename': 'area_trend'}})

            with t_col2:
                fig_pos = go.Figure()
                for i in range(n_peaks):
                    fig_pos.add_trace(go.Scatter(
                        x=res_df['Voltage'], y=res_df[f'Peak{i + 1}_Center'],
                        mode='lines+markers', name=f'P{i + 1} Center',
                        line=dict(color=current_colors[i], dash='dash'), marker=dict(symbol='square', size=8)
                    ))
                fig_pos.update_layout(title="Peak Position vs. Potential", xaxis_title="V vs RHE")
                apply_nature_style(fig_pos, font_family, font_size, axis_width, legend_inside=True)
                st.plotly_chart(fig_pos, use_container_width=True,
                                config={'toImageButtonOptions': {'format': 'svg', 'filename': 'pos_trend'}})

            st.divider()

            st.markdown("#### 🔍 拟合详情回溯 (Detail Inspector)")
            label_list = [d['title'] for d in details]
            selected_label = st.select_slider("滑动选择电位", options=label_list, value=label_list[0])
            idx = label_list.index(selected_label)
            d_data = details[idx]
            d_res = res_df.iloc[idx]

            d_col1, d_col2 = st.columns([3, 1])
            with d_col1:
                fig_detail = go.Figure()
                fig_detail.add_trace(go.Scatter(
                    x=d_data['x'], y=d_data['y_raw'], mode='markers', name='Data',
                    marker=dict(color=data_point_color, size=data_point_size, opacity=0.6)
                ))
                fig_detail.add_trace(go.Scatter(
                    x=d_data['x'], y=d_data['y_fit'], mode='lines', name=f'Total Fit',
                    line=dict(color=fit_line_color, width=2)
                ))
                for i, yp in enumerate(d_data['peaks']):
                    fig_detail.add_trace(go.Scatter(
                        x=d_data['x'], y=yp, mode='lines', fill='tozeroy',
                        name=f'P{i + 1}',
                        line=dict(color=current_colors[i], width=0),
                        fillcolor=current_colors[i], opacity=0.5
                    ))
                fig_detail.update_layout(title=f"Fitting Detail @ {selected_label}", xaxis_title="Wavenumber (cm⁻¹)",
                                         yaxis_title="Absorbance")
                apply_nature_style(fig_detail, font_family, font_size, axis_width, legend_inside=True)
                st.plotly_chart(fig_detail, use_container_width=True, config={
                    'toImageButtonOptions': {'format': 'svg', 'filename': f'detail_{selected_label}'}})

            with d_col2:
                st.markdown(f"**Potential:** {d_res['Voltage']} V")
                st.markdown(f"**Fit R²:** `{d_res['R2']:.4f}`")
                st.markdown("---")
                for i in range(n_peaks):
                    st.markdown(f"**Peak {i + 1}**")
                    st.caption(f"Area: {d_res[f'Peak{i + 1}_Area']:.4f}")
                    st.caption(f"Center: {d_res[f'Peak{i + 1}_Center']:.1f}")
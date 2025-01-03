# -*- coding: utf-8 -*-
# @Time    : 2025/1/3 14:39
# @Author  : AI悦创
# @FileName: 24h_improved_multi_days.py
# @Software: PyCharm
# @Blog    ：https://bornforthis.cn/
# code is far away from bugs with the god animal protecting
#    I love animals. They taste delicious.
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, time, timedelta
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os
from matplotlib import rcParams

# 注册中文字体（宋体），以便在PDF和matplotlib图表中正常显示中文
pdfmetrics.registerFont(TTFont('SimSun', 'SimSun.ttf'))
rcParams['font.sans-serif'] = ['SimSun', 'PingFang SC', 'SimHei', 'Songti SC']
rcParams['axes.unicode_minus'] = False  # 解决坐标轴负号显示问题

# ============ 可配置的阈值 ============ #
NORMAL_SBP_RANGE = (90, 140)  # 明细表中用来标示 SBP ↑ 或 ↓
NORMAL_DBP_RANGE = (60, 90)  # 明细表中用来标示 DBP ↑ 或 ↓

# 用于饼图三分法
REF_THRESHOLDS = {
    'day': {'SBP': (90, 135), 'DBP': (60, 85)},  # 白天收缩压/舒张压正常区间
    'night': {'SBP': (80, 120), 'DBP': (50, 70)},  # 夜间收缩压/舒张压正常区间
    'full': {'SBP': (90, 140), 'DBP': (60, 90)},  # 全天收缩压/舒张压正常区间
}


def parse_time_with_day_offset(time_str):
    """
    解析像 '13:30(1)'、'08:00(2)' 等格式的时间：
    返回 (time对象, day_offset)。
      - 若无括号，则 day_offset=0
      - 若有 (n)，则 day_offset = n - 1
        即 (1) 表示当天，(2) 表示加 1 天，以此类推。
    """
    pattern = re.compile(r'(\d{1,2}:\d{2})(?:\((\d+)\))?$')
    match = pattern.match(time_str.strip())
    if not match:
        return None, 0

    base_time_str = match.group(1)  # "13:30"
    offset_str = match.group(2)  # "1", "2"... 可能为 None

    # 转为 time
    try:
        t = datetime.strptime(base_time_str, '%H:%M').time()
    except:
        return None, 0

    day_offset = 0
    if offset_str:
        # 若括号内是 n，则表示比基准日期多 n-1 天
        offset = int(offset_str)
        day_offset = offset - 1  # (1)->0; (2)->1; (3)->2

    return t, day_offset


def mark_value(value, low, high):
    """在明细表中，用于标示高↑ / 低↓。"""
    if value is None:
        return "N/A"
    if value < low:
        return f"{value}↓"
    elif value > high:
        return f"{value}↑"
    else:
        return str(value)


def read_data_from_excel(file_path):
    """
    从Excel中读取数据，并返回DataFrame。
    假设表头： 日期 | 时间 | 高压 | 低压 | 心率 | 情况
    时间列里可能是类似 "13:30(1)" / "13:30(2)" 等格式。
    """
    # 读取Excel
    df = pd.read_excel(file_path)

    # 重命名列以便后续编程处理
    df.rename(columns={
        '日期': 'Date',
        '时间': 'Time',
        '高压': 'SBP',
        '低压': 'DBP',
        '心率': 'HR',
        '情况': 'Note'
    }, inplace=True)

    # 自定义解析函数
    def parse_datetime(row):
        # 1) 先解析日期 (YYYY-mm-dd)
        base_date_str = str(row['Date']).strip()
        try:
            base_date = datetime.strptime(base_date_str, '%Y-%m-%d').date()
        except:
            # 若无法解析日期，则返回 NaT
            return pd.NaT

        # 2) 解析时间(含日偏移)
        time_str = str(row['Time']).strip()
        time_part, day_offset = parse_time_with_day_offset(time_str)
        if not time_part:
            return pd.NaT

        # 3) 计算最终日期时间
        final_date = base_date + timedelta(days=day_offset)
        return datetime.combine(final_date, time_part)

    # 生成新列 DateTime
    df['DateTime'] = df.apply(parse_datetime, axis=1)

    # 去除无效行
    df.dropna(subset=['DateTime', 'SBP', 'DBP', 'HR'], inplace=True)
    df.sort_values(by='DateTime', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 计算脉压差
    df['PP'] = df['SBP'] - df['DBP']
    return df


def split_day_night_multi_day(df, day_start='08:00', night_start='23:00'):
    """
    多日拆分:
    对于每个自然日分别将 day_start ~ night_start 视为白天, 其余为夜间。
    返回 day_df, night_df, full_df(原数据)。
    """
    if df.empty:
        return df.copy(), df.copy(), df.copy()

    # 先提取日期
    df['DateOnly'] = df['DateTime'].dt.date
    day_parts = []
    night_parts = []

    # 转换成 time 对象
    fmt = '%H:%M'
    day_start_t = datetime.strptime(day_start, fmt).time()  # 08:00
    night_start_t = datetime.strptime(night_start, fmt).time()  # 23:00

    # 遍历每个日期分组
    for date_val, g in df.groupby('DateOnly'):
        sub_df = g.sort_values(by='DateTime')
        sub_df_day = []
        sub_df_night = []

        # day: day_start -> night_start
        def in_daytime(dt):
            t = dt.time()
            return (t >= day_start_t) and (t < night_start_t)

        # night: 其它都算夜间
        def in_nighttime(dt):
            return not in_daytime(dt)

        # 分割
        day_df_ = sub_df[sub_df['DateTime'].apply(in_daytime)]
        night_df_ = sub_df[sub_df['DateTime'].apply(in_nighttime)]

        day_parts.append(day_df_)
        night_parts.append(night_df_)

    # 合并
    day_df = pd.concat(day_parts, ignore_index=True)
    night_df = pd.concat(night_parts, ignore_index=True)

    # 清理中间列
    df.drop(columns=['DateOnly'], inplace=True, errors='ignore')
    day_df.drop(columns=['DateOnly'], inplace=True, errors='ignore')
    night_df.drop(columns=['DateOnly'], inplace=True, errors='ignore')

    return day_df, night_df, df


def calc_statistics(df, label='全天'):
    """
    计算一些基础统计指标。
    """
    if df.empty:
        return {
            'label': label,
            'mean_sbp': None,
            'max_sbp': None,
            'max_sbp_time': None,
            'min_sbp': None,
            'min_sbp_time': None,
            'mean_dbp': None,
            'max_dbp': None,
            'max_dbp_time': None,
            'min_dbp': None,
            'min_dbp_time': None,
            'mean_map': None,
            'mean_hr': None,
            'sbp_std': None,
            'dbp_std': None,
            'sbp_load': None,
            'dbp_load': None,
            'sbp_arv': None,
            'dbp_arv': None,
            'count': 0
        }

    # 平均动脉压
    df['MAP'] = df['DBP'] + (df['SBP'] - df['DBP']) / 3.0

    max_sbp_idx = df['SBP'].idxmax()
    min_sbp_idx = df['SBP'].idxmin()
    max_dbp_idx = df['DBP'].idxmax()
    min_dbp_idx = df['DBP'].idxmin()

    def average_real_variation(series):
        arr = series.dropna().values
        if len(arr) < 2:
            return None
        return np.mean(np.abs(np.diff(arr)))

    stats = {
        'label': label,
        'mean_sbp': round(df['SBP'].mean(), 2),
        'max_sbp': int(df.loc[max_sbp_idx, 'SBP']),
        'max_sbp_time': df.loc[max_sbp_idx, 'DateTime'].strftime('%Y-%m-%d %H:%M:%S'),
        'min_sbp': int(df.loc[min_sbp_idx, 'SBP']),
        'min_sbp_time': df.loc[min_sbp_idx, 'DateTime'].strftime('%Y-%m-%d %H:%M:%S'),

        'mean_dbp': round(df['DBP'].mean(), 2),
        'max_dbp': int(df.loc[max_dbp_idx, 'DBP']),
        'max_dbp_time': df.loc[max_dbp_idx, 'DateTime'].strftime('%Y-%m-%d %H:%M:%S'),
        'min_dbp': int(df.loc[min_dbp_idx, 'DBP']),
        'min_dbp_time': df.loc[min_dbp_idx, 'DateTime'].strftime('%Y-%m-%d %H:%M:%S'),

        'mean_map': round(df['MAP'].mean(), 2),
        'mean_hr': round(df['HR'].mean(), 2),
        'sbp_std': round(df['SBP'].std(), 2),
        'dbp_std': round(df['DBP'].std(), 2),

        # 血压负荷 (示例：收缩压>=140, 舒张压>=90)
        'sbp_load': f"{round((df['SBP'] >= 140).sum() / len(df) * 100, 2)}%",
        'dbp_load': f"{round((df['DBP'] >= 90).sum() / len(df) * 100, 2)}%",

        'sbp_arv': round(average_real_variation(df['SBP']), 2) if average_real_variation(df['SBP']) else None,
        'dbp_arv': round(average_real_variation(df['DBP']), 2) if average_real_variation(df['DBP']) else None,

        'count': len(df)
    }
    return stats


def filter_by_time_range(df, start_time_str, end_time_str):
    """
    根据 [start_time, end_time) 筛选。
    若 start_time> end_time，视为跨午夜。
    """
    if df.empty:
        return df
    fmt = '%H:%M'
    start_t = datetime.strptime(start_time_str, fmt).time()
    end_t = datetime.strptime(end_time_str, fmt).time()

    def in_range(t):
        if start_t < end_t:
            return start_t <= t < end_t
        else:
            return (t >= start_t) or (t < end_t)

    return df[df['DateTime'].dt.time.apply(in_range)].copy()


def calc_morning_surge(full_df,
                       morning_start='06:00', morning_end='08:00',
                       pre_start='03:00', pre_end='06:00'):
    """
    计算晨峰血压: 对比 pre_morning (03:00-06:00) 与 morning (06:00-08:00) 平均值。
    """
    if full_df.empty:
        return None

    morning_df = filter_by_time_range(full_df, morning_start, morning_end)
    pre_df = filter_by_time_range(full_df, pre_start, pre_end)
    if morning_df.empty or pre_df.empty:
        return None  # 数据不足

    sbp_m_avg = morning_df['SBP'].mean()
    sbp_p_avg = pre_df['SBP'].mean()
    dbp_m_avg = morning_df['DBP'].mean()
    dbp_p_avg = pre_df['DBP'].mean()

    return {
        'sbp_morning_surge': round(sbp_m_avg - sbp_p_avg, 2),
        'dbp_morning_surge': round(dbp_m_avg - dbp_p_avg, 2),
        'sbp_morning_avg': round(sbp_m_avg, 2),
        'sbp_pre_morning_avg': round(sbp_p_avg, 2),
        'dbp_morning_avg': round(dbp_m_avg, 2),
        'dbp_pre_morning_avg': round(dbp_p_avg, 2),
    }


def calc_extra_indices(day_stats, night_stats, full_df,
                       morning_start='06:00', morning_end='08:00',
                       pre_morning_start='03:00', pre_morning_end='06:00'):
    """
    计算昼夜差、下降率、昼夜比、晨峰血压等。
    """
    if not day_stats or not night_stats:
        return {}

    if day_stats['mean_sbp'] is None or night_stats['mean_sbp'] is None:
        return {}

    sbp_day = day_stats['mean_sbp']
    sbp_night = night_stats['mean_sbp']
    dbp_day = day_stats['mean_dbp']
    dbp_night = night_stats['mean_dbp']

    sbp_diff = sbp_day - sbp_night
    dbp_diff = dbp_day - dbp_night

    sbp_dip = (sbp_diff / sbp_day * 100) if sbp_day != 0 else None
    dbp_dip = (dbp_diff / dbp_day * 100) if dbp_day != 0 else None

    sbp_ratio = (sbp_night / sbp_day * 100) if sbp_day != 0 else None
    dbp_ratio = (dbp_night / dbp_day * 100) if dbp_day != 0 else None

    surge_info = calc_morning_surge(full_df, morning_start, morning_end,
                                    pre_morning_start, pre_morning_end)

    result = {
        'sbp_day': round(sbp_day, 2),
        'sbp_night': round(sbp_night, 2),
        'sbp_diff': round(sbp_diff, 2),
        'sbp_dip': round(sbp_dip, 2) if sbp_dip is not None else None,
        'sbp_ratio': round(sbp_ratio, 2) if sbp_ratio is not None else None,

        'dbp_day': round(dbp_day, 2),
        'dbp_night': round(dbp_night, 2),
        'dbp_diff': round(dbp_diff, 2),
        'dbp_dip': round(dbp_dip, 2) if dbp_dip is not None else None,
        'dbp_ratio': round(dbp_ratio, 2) if dbp_ratio is not None else None,
    }

    if surge_info:
        result.update(surge_info)
    else:
        result.update({
            'sbp_morning_surge': None,
            'dbp_morning_surge': None,
            'sbp_morning_avg': None,
            'sbp_pre_morning_avg': None,
            'dbp_morning_avg': None,
            'dbp_pre_morning_avg': None,
        })

    return result


def plot_bp_six_pies(day_df, night_df, full_df,
                     thresholds=REF_THRESHOLDS,
                     output_file="result/bp_6_pies.png"):
    """
    绘制白天/夜间/全天 的 SBP 与 DBP 饼图（高/正常/低 三分）。
    """
    if not os.path.exists("result"):
        os.makedirs("result")

    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    axes = axes.ravel()

    day_sbp_ref = thresholds['day']['SBP']
    day_dbp_ref = thresholds['day']['DBP']
    night_sbp_ref = thresholds['night']['SBP']
    night_dbp_ref = thresholds['night']['DBP']
    full_sbp_ref = thresholds['full']['SBP']
    full_dbp_ref = thresholds['full']['DBP']

    configs = [
        ("白天 收缩压", day_df['SBP'] if not day_df.empty else pd.Series(dtype=float), day_sbp_ref),
        ("夜间 收缩压", night_df['SBP'] if not night_df.empty else pd.Series(dtype=float), night_sbp_ref),
        ("全天 收缩压", full_df['SBP'], full_sbp_ref),
        ("白天 舒张压", day_df['DBP'] if not day_df.empty else pd.Series(dtype=float), day_dbp_ref),
        ("夜间 舒张压", night_df['DBP'] if not night_df.empty else pd.Series(dtype=float), night_dbp_ref),
        ("全天 舒张压", full_df['DBP'], full_dbp_ref),
    ]

    labels = ["H", "N", "L"]
    colors = ["tomato", "lightgreen", "lightskyblue"]

    for i, (title, series, ref_range) in enumerate(configs):
        ax = axes[i]
        low, high = ref_range

        if series.empty:
            ax.set_title(f"{title}\n无数据")
            ax.axis('off')
            continue

        count_H = (series > high).sum()
        count_L = (series < low).sum()
        count_N = len(series) - count_H - count_L
        total = count_H + count_N + count_L
        if total == 0:
            ax.set_title(f"{title}\n无数据")
            ax.axis('off')
            continue

        data = [count_H, count_N, count_L]
        patches, texts, autotexts = ax.pie(
            data,
            labels=labels,
            colors=colors,
            autopct=lambda pct: f"{pct:.1f}%" if pct > 0 else "",
            startangle=140
        )
        ax.set_title(f"{title}\n(N={total})", fontsize=10)

        for text in texts:
            text.set_fontsize(9)
        for autot in autotexts:
            autot.set_fontsize(9)

    # 在右下角添加图例
    from matplotlib.patches import Patch
    legend_patches = [
        Patch(color="tomato", label="H(高)"),
        Patch(color="lightgreen", label="N(正常)"),
        Patch(color="lightskyblue", label="L(低)")
    ]
    plt.subplots_adjust(bottom=0.15)
    fig.legend(handles=legend_patches, loc='lower right', bbox_to_anchor=(1, 0.02))

    plt.tight_layout()
    plt.savefig(output_file, dpi=100)
    plt.close()
    return output_file


def generate_plot(df, output_png='result/blood_pressure_plot.png'):
    """
    生成收缩压/舒张压随时间折线图。
    """
    if df.empty:
        print("数据为空，无法生成折线图。")
        return None

    x = df['DateTime']
    y_sbp = df['SBP']
    y_dbp = df['DBP']

    plt.figure(figsize=(10, 5))
    plt.plot(x, y_sbp, marker='o', label='SBP(收缩压)')
    plt.plot(x, y_dbp, marker='s', label='DBP(舒张压)')
    plt.title('24h 动态血压趋势图')
    plt.xlabel('时间')
    plt.ylabel('血压 (mmHg)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    if not os.path.exists("result"):
        os.makedirs("result")
    plt.savefig(output_png, dpi=100)
    plt.close()
    return output_png


def generate_pdf_report(day_stats, night_stats, full_stats, extra_indices,
                        df, day_df, night_df,
                        plot_file=None,
                        pie_6_file=None,
                        output_pdf='result/blood_pressure_report.pdf'):
    """
    生成 PDF 报告。
    """
    doc = SimpleDocTemplate(output_pdf, pagesize=A4)
    styles = getSampleStyleSheet()

    # 设置 Normal 和 Title 样式字体为 SimSun
    styles['Normal'].fontName = 'SimSun'
    styles['Title'].fontName = 'SimSun'
    # 适当调大行距
    styles['Normal'].leading = 16
    styles['Title'].leading = 20

    story = []

    # 标题
    story.append(Paragraph("24小时动态血压报告", styles['Title']))
    story.append(Spacer(1, 12))

    # 简要说明
    story.append(Paragraph("开发者：黄家宝", styles['Normal']))
    disclaimer_text = """
    <b>非医疗设备，仅供参考：</b><br/>
    本分析基于个人/家庭测量或记录数据，无法替代专业医疗器械的24h动态血压监测。<br/>
    若有高血压史或异常结果，请结合临床症状及时就医。<br/>开发者微信：Jiabcdefh
    """
    story.append(Paragraph(disclaimer_text, styles['Normal']))
    story.append(Spacer(1, 12))

    # 汇总表
    table_data = [["统计项", "白天", "夜间", "全天"]]
    # 平均SBP
    table_data.append([
        "平均SBP",
        f"{day_stats['mean_sbp']}" if day_stats['mean_sbp'] else "N/A",
        f"{night_stats['mean_sbp']}" if night_stats['mean_sbp'] else "N/A",
        f"{full_stats['mean_sbp']}" if full_stats['mean_sbp'] else "N/A"
    ])
    # 最大SBP(含时刻)
    table_data.append([
        "最大SBP(时刻)",
        f"{day_stats['max_sbp']} ({day_stats['max_sbp_time']})" if day_stats['max_sbp'] else "N/A",
        f"{night_stats['max_sbp']} ({night_stats['max_sbp_time']})" if night_stats['max_sbp'] else "N/A",
        f"{full_stats['max_sbp']} ({full_stats['max_sbp_time']})" if full_stats['max_sbp'] else "N/A"
    ])
    # 最小SBP(含时刻)
    table_data.append([
        "最小SBP(时刻)",
        f"{day_stats['min_sbp']} ({day_stats['min_sbp_time']})" if day_stats['min_sbp'] else "N/A",
        f"{night_stats['min_sbp']} ({night_stats['min_sbp_time']})" if night_stats['min_sbp'] else "N/A",
        f"{full_stats['min_sbp']} ({full_stats['min_sbp_time']})" if full_stats['min_sbp'] else "N/A"
    ])

    # 平均DBP
    table_data.append([
        "平均DBP",
        f"{day_stats['mean_dbp']}" if day_stats['mean_dbp'] else "N/A",
        f"{night_stats['mean_dbp']}" if night_stats['mean_dbp'] else "N/A",
        f"{full_stats['mean_dbp']}" if full_stats['mean_dbp'] else "N/A"
    ])
    # 最大DBP(含时刻)
    table_data.append([
        "最大DBP(时刻)",
        f"{day_stats['max_dbp']} ({day_stats['max_dbp_time']})" if day_stats['max_dbp'] else "N/A",
        f"{night_stats['max_dbp']} ({night_stats['max_dbp_time']})" if night_stats['max_dbp'] else "N/A",
        f"{full_stats['max_dbp']} ({full_stats['max_dbp_time']})" if full_stats['max_dbp'] else "N/A"
    ])
    # 最小DBP(含时刻)
    table_data.append([
        "最小DBP(时刻)",
        f"{day_stats['min_dbp']} ({day_stats['min_dbp_time']})" if day_stats['min_dbp'] else "N/A",
        f"{night_stats['min_dbp']} ({night_stats['min_dbp_time']})" if night_stats['min_dbp'] else "N/A",
        f"{full_stats['min_dbp']} ({full_stats['min_dbp_time']})" if full_stats['min_dbp'] else "N/A"
    ])

    # 标准差和血压负荷
    table_data.append([
        "SBP标准差",
        f"{day_stats['sbp_std']}" if day_stats['sbp_std'] else "N/A",
        f"{night_stats['sbp_std']}" if night_stats['sbp_std'] else "N/A",
        f"{full_stats['sbp_std']}" if full_stats['sbp_std'] else "N/A"
    ])
    table_data.append([
        "DBP标准差",
        f"{day_stats['dbp_std']}" if day_stats['dbp_std'] else "N/A",
        f"{night_stats['dbp_std']}" if night_stats['dbp_std'] else "N/A",
        f"{full_stats['dbp_std']}" if full_stats['dbp_std'] else "N/A"
    ])
    table_data.append([
        "SBP血压负荷",
        f"{day_stats['sbp_load']}" if day_stats['sbp_load'] else "N/A",
        f"{night_stats['sbp_load']}" if night_stats['sbp_load'] else "N/A",
        f"{full_stats['sbp_load']}" if full_stats['sbp_load'] else "N/A"
    ])
    table_data.append([
        "DBP血压负荷",
        f"{day_stats['dbp_load']}" if day_stats['dbp_load'] else "N/A",
        f"{night_stats['dbp_load']}" if night_stats['dbp_load'] else "N/A",
        f"{full_stats['dbp_load']}" if full_stats['dbp_load'] else "N/A"
    ])

    # 新增：平均真实变异(ARV)
    table_data.append([
        "SBP平均真实变异(ARV)",
        f"{day_stats['sbp_arv']}" if day_stats['sbp_arv'] is not None else "N/A",
        f"{night_stats['sbp_arv']}" if night_stats['sbp_arv'] is not None else "N/A",
        f"{full_stats['sbp_arv']}" if full_stats['sbp_arv'] is not None else "N/A",
    ])
    table_data.append([
        "DBP平均真实变异(ARV)",
        f"{day_stats['dbp_arv']}" if day_stats['dbp_arv'] is not None else "N/A",
        f"{night_stats['dbp_arv']}" if night_stats['dbp_arv'] is not None else "N/A",
        f"{full_stats['dbp_arv']}" if full_stats['dbp_arv'] is not None else "N/A",
    ])

    tbl_style = TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('FONTNAME', (0, 0), (-1, -1), 'SimSun'),
    ])
    stat_table = Table(table_data)
    stat_table.setStyle(tbl_style)
    story.append(stat_table)
    story.append(Spacer(1, 20))

    # 显示额外指标：昼夜差、下降率、晨峰等
    story.append(Paragraph("<b>额外指标：</b>", styles['Normal']))
    if extra_indices:
        if extra_indices.get('sbp_morning_surge') is None:
            surge_sbp_text = "N/A(数据不足)"
        else:
            surge_sbp_text = str(extra_indices['sbp_morning_surge'])
        if extra_indices.get('dbp_morning_surge') is None:
            surge_dbp_text = "N/A(数据不足)"
        else:
            surge_dbp_text = str(extra_indices['dbp_morning_surge'])

        text = f"""
        收缩压(白天): {extra_indices.get('sbp_day', 'N/A')} mmHg，
        夜间: {extra_indices.get('sbp_night', 'N/A')} mmHg，<br/>
        差值: {extra_indices.get('sbp_diff', 'N/A')} mmHg，
        下降率: {extra_indices.get('sbp_dip', 'N/A')}%，
        昼夜比(夜/昼): {extra_indices.get('sbp_ratio', 'N/A')}%，
        晨峰血压: {surge_sbp_text} mmHg<br/><br/>

        舒张压(白天): {extra_indices.get('dbp_day', 'N/A')} mmHg，
        夜间: {extra_indices.get('dbp_night', 'N/A')} mmHg，<br/>
        差值: {extra_indices.get('dbp_diff', 'N/A')} mmHg，
        下降率: {extra_indices.get('dbp_dip', 'N/A')}%，
        昼夜比(夜/昼): {extra_indices.get('dbp_ratio', 'N/A')}%，
        晨峰血压: {surge_dbp_text} mmHg
        """
        story.append(Paragraph(text, styles['Normal']))
    else:
        story.append(Paragraph("无可计算结果或数据不足。", styles['Normal']))
    story.append(Spacer(1, 20))

    # 在PDF中插入“六饼图”
    if pie_6_file and os.path.exists(pie_6_file):
        story.append(Paragraph("<b>血压分布（高/正常/低）饼图</b>", styles['Title']))
        story.append(Spacer(1, 10))
        story.append(Image(pie_6_file, width=500, height=350))  # 可自行调大小
        story.append(Spacer(1, 20))
        story.append(PageBreak())

    # 若有折线图，则插入图
    if plot_file and os.path.exists(plot_file):
        story.append(Paragraph("<b>24小时趋势图</b>", styles['Title']))
        story.append(Spacer(1, 10))
        story.append(Image(plot_file, width=400, height=250))
        story.append(Spacer(1, 20))

    # 明细表 - 先增加说明
    story.append(Paragraph("<b>测量明细</b>", styles['Title']))
    note_text = f"""
    <b>说明：</b>
    为方便识别高低血压，系统对收缩压和舒张压进行了标记：<br/>
    · 正常范围：SBP {NORMAL_SBP_RANGE[0]}~{NORMAL_SBP_RANGE[1]} mmHg；DBP {NORMAL_DBP_RANGE[0]}~{NORMAL_DBP_RANGE[1]} mmHg。<br/>
    · 超出上述范围时，数值会带有 ↑ 或 ↓ 标识。<br/>
    """
    story.append(Paragraph(note_text, styles['Normal']))
    story.append(Spacer(1, 8))

    detail_data = [["编号", "日期", "时间", "收缩压", "舒张压", "平均压", "脉率", "脉压差", "说明"]]
    for i, row in df.iterrows():
        dt_obj = row['DateTime']
        date_str = dt_obj.strftime('%Y-%m-%d')
        time_str = dt_obj.strftime('%H:%M:%S')

        map_val = row.get('MAP', None)
        map_val = round(map_val, 2) if map_val is not None else ""

        # 这里用 mark_value() 根据 NORMAL_SBP_RANGE & NORMAL_DBP_RANGE 判断是否加箭头
        sbp_str = mark_value(row['SBP'], NORMAL_SBP_RANGE[0], NORMAL_SBP_RANGE[1])
        dbp_str = mark_value(row['DBP'], NORMAL_DBP_RANGE[0], NORMAL_DBP_RANGE[1])

        detail_data.append([
            i + 1,  # 编号
            date_str,  # 日期
            time_str,  # 时间
            sbp_str,  # 收缩压(附带箭头)
            dbp_str,  # 舒张压(附带箭头)
            map_val,  # 平均压
            row['HR'],  # 脉率
            row['PP'],  # 脉压差
            row['说明']
        ])

    detail_table = Table(detail_data, colWidths=[40, 70, 60, 60, 60, 60, 40, 60])
    detail_style = TableStyle([
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, -1), 'SimSun'),
    ])
    detail_table.setStyle(detail_style)
    story.append(detail_table)
    story.append(Spacer(1, 20))

    # 总结
    summary_text = """
    <b>总结：</b><br/>
    · 本程序开发初衷只为我老婆孕期血压跟踪开发！<br/> 
    · 本报告结果仅供个人参考，不能替代医用设备。<br/> 
    · 建议在医生指导下做进一步检查或专业的动态血压监测。<br/>
    · 若血压持续异常，请尽快就医。
    """
    story.append(Paragraph(summary_text, styles['Normal']))

    # 生成 PDF
    doc.build(story)
    print(f"PDF 报告已生成: {output_pdf}")


def main():
    # 1. 读取 Excel，支持时间格式如 "13:30(1)"、"13:30(2)"
    file_path = "data/blood_pressure-test2.xlsx"
    df = read_data_from_excel(file_path)
    if df.empty:
        print("Excel 数据为空或无法解析，请检查文件。")
        return

    # 2. 将多日数据拆分为 白天 & 夜间
    #    这里使用多日拆分版本，能应对跨午夜、多天数据
    day_df, night_df, full_df = split_day_night_multi_day(df, day_start='08:00', night_start='23:00')

    # 3. 分别计算统计指标
    day_stats = calc_statistics(day_df, label='白天')
    night_stats = calc_statistics(night_df, label='夜间')
    full_stats = calc_statistics(full_df, label='全天')

    # 4. 计算额外指标(包括晨峰血压)
    extra_indices = calc_extra_indices(
        day_stats, night_stats, full_df,
        morning_start='06:00', morning_end='08:00',
        pre_morning_start='03:00', pre_morning_end='06:00'
    )

    # 5. 生成折线图、饼图
    plot_file = generate_plot(full_df, output_png='result/blood_pressure_plot.png')
    six_pies_file = plot_bp_six_pies(day_df, night_df, full_df,
                                     thresholds=REF_THRESHOLDS,
                                     output_file="result/bp_6_pies.png")

    # 6. 生成 PDF 报告
    generate_pdf_report(day_stats, night_stats, full_stats, extra_indices,
                        full_df, day_df, night_df,
                        plot_file=plot_file,
                        pie_6_file=six_pies_file,
                        output_pdf='result/blood_pressure_report.pdf')


if __name__ == '__main__':
    main()
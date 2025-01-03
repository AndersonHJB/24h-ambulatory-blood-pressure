# -*- coding: utf-8 -*-
# @Time    : 2025/1/3 14:39
# @Author  : AI悦创
# @FileName: 24h.py
# @Software: PyCharm
# @Blog    ：https://bornforthis.cn/
# code is far away from bugs with the god animal protecting
#    I love animals. They taste delicious.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, time
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
rcParams['font.sans-serif'] = ['PingFang SC', 'SimHei', 'Songti SC']  # 可根据实际环境调整
rcParams['axes.unicode_minus'] = False  # 解决坐标轴负号显示问题

# 设定正常血压范围（可根据需要修改）
NORMAL_SBP_RANGE = (90, 140)  # SBP在此区间内视为正常
NORMAL_DBP_RANGE = (60, 90)   # DBP在此区间内视为正常

def mark_value(value, low, high):
    """
    判断value是否在[low, high]之间：
    - 若 value < low，返回 'value↓'
    - 若 value > high，返回 'value↑'
    - 否则，返回 str(value)
    """
    if value < low:
        return f"{value}↓"
    elif value > high:
        return f"{value}↑"
    else:
        return str(value)

def read_data_from_excel(file_path):
    """
    从Excel中读取数据，并返回DataFrame。
    假设表头为： 日期 | 时间 | 高压 | 低压 | 心率 | 情况
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

    # 合并日期和时间，转化为 datetime 类型，便于后续操作
    df['DateTime'] = df.apply(
        lambda row: datetime.strptime(str(row['Date']) + ' ' + str(row['Time']), '%Y-%m-%d %H:%M'),
        axis=1
    )

    # 按时间排序
    df.sort_values(by='DateTime', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 增加 脉压差（Pulse Pressure, PP） 列 (SBP - DBP)
    df['PP'] = df['SBP'] - df['DBP']

    return df

def split_day_night(df, day_start='08:00', night_start='23:00'):
    """
    将全天数据拆分为 白天(08:00 - 22:59) 和 夜间(23:00 - 次日07:59)。
    返回三个DataFrame: day_df, night_df, full_df
    """
    # 定义时间界限
    day_start_time = datetime.strptime(day_start, '%H:%M').time()
    night_start_time = datetime.strptime(night_start, '%H:%M').time()

    # 判断是否属于白天： day_start_time <= t < night_start_time
    def is_daytime(t):
        return (t >= day_start_time) and (t < night_start_time)

    # 判断是否属于夜间： t >= night_start_time or t < day_start_time
    def is_nighttime(t):
        return (t >= night_start_time) or (t < day_start_time)

    # 分别筛选，并使用 .copy() 保证返回的子集是副本
    day_df = df[df['DateTime'].dt.time.apply(is_daytime)].copy()
    night_df = df[df['DateTime'].dt.time.apply(is_nighttime)].copy()

    # 第三个返回值 full_df 就是原始 df
    return day_df, night_df, df

def calc_statistics(df, label='全天'):
    """
    计算收缩压、舒张压、平均压、脉率等的基本统计量，并包含：
    - 最大值、最大值发生时刻
    - 最小值、最小值发生时刻
    - 平均真实变异(ARV)
    返回一个字典。
    """
    # 如果数据为空，返回空统计
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

    # 计算平均动脉压（MAP）
    df['MAP'] = df['DBP'] + (df['SBP'] - df['DBP']) / 3.0

    # 最大和最小 SBP
    max_sbp_idx = df['SBP'].idxmax()
    min_sbp_idx = df['SBP'].idxmin()

    # 最大和最小 DBP
    max_dbp_idx = df['DBP'].idxmax()
    min_dbp_idx = df['DBP'].idxmin()

    # 函数：计算 ARV
    def average_real_variation(series):
        arr = series.dropna().values
        if len(arr) < 2:
            return None
        return np.mean(np.abs(np.diff(arr)))

    # 组装统计结果
    stats = {
        'label': label,
        'mean_sbp': round(df['SBP'].mean(), 2),
        'max_sbp': int(df.loc[max_sbp_idx, 'SBP']),
        'max_sbp_time': df.loc[max_sbp_idx, 'DateTime'].strftime('%H:%M:%S'),
        'min_sbp': int(df.loc[min_sbp_idx, 'SBP']),
        'min_sbp_time': df.loc[min_sbp_idx, 'DateTime'].strftime('%H:%M:%S'),

        'mean_dbp': round(df['DBP'].mean(), 2),
        'max_dbp': int(df.loc[max_dbp_idx, 'DBP']),
        'max_dbp_time': df.loc[max_dbp_idx, 'DateTime'].strftime('%H:%M:%S'),
        'min_dbp': int(df.loc[min_dbp_idx, 'DBP']),
        'min_dbp_time': df.loc[min_dbp_idx, 'DateTime'].strftime('%H:%M:%S'),

        'mean_map': round(df['MAP'].mean(), 2),
        'mean_hr': round(df['HR'].mean(), 2),

        'sbp_std': round(df['SBP'].std(), 2),
        'dbp_std': round(df['DBP'].std(), 2),

        # 血压负荷（示例：收缩压 >= 140 mmHg 的比例；舒张压 >= 90 mmHg 的比例）
        'sbp_load': f"{round((df['SBP'] >= 140).sum() / len(df) * 100, 2)}%",
        'dbp_load': f"{round((df['DBP'] >= 90).sum() / len(df) * 100, 2)}%",

        # 新增：平均真实变异(ARV)
        'sbp_arv': round(average_real_variation(df['SBP']), 2) if average_real_variation(df['SBP']) is not None else None,
        'dbp_arv': round(average_real_variation(df['DBP']), 2) if average_real_variation(df['DBP']) is not None else None,

        'count': len(df)
    }

    return stats

def filter_by_time_range(df, start_time_str, end_time_str):
    """
    根据指定的时段 [start_time, end_time)，返回该时间段内的所有数据。
    若 start_time > end_time，视为跨午夜。否则视为同日区间。
    """
    if df.empty:
        return df

    fmt = '%H:%M'
    start_time = datetime.strptime(start_time_str, fmt).time()
    end_time = datetime.strptime(end_time_str, fmt).time()

    def in_range(t):
        # 如果 start < end：同一天段
        if start_time < end_time:
            return start_time <= t < end_time
        else:
            # 若跨午夜
            return (t >= start_time) or (t < end_time)

    df_filtered = df[df['DateTime'].dt.time.apply(in_range)].copy()
    return df_filtered


def calc_morning_surge(full_df,
                       morning_start='06:00', morning_end='08:00',
                       pre_start='03:00', pre_end='06:00'):
    """
    计算“晨峰血压”：默认对比 03:00-06:00 与 06:00-08:00 两段数据的平均 SBP/DBP。
    返回 dict, 包含 sbp_morning_surge, dbp_morning_surge, 以及这两段的平均值。
    """
    if full_df.empty:
        return {}

    # 晨间时段
    morning_df = filter_by_time_range(full_df, morning_start, morning_end)
    # 凌晨对比时段
    pre_morning_df = filter_by_time_range(full_df, pre_start, pre_end)

    if morning_df.empty or pre_morning_df.empty:
        # 数据不足，返回空
        return {}

    sbp_morning_avg = morning_df['SBP'].mean()
    sbp_pre_morning_avg = pre_morning_df['SBP'].mean()
    dbp_morning_avg = morning_df['DBP'].mean()
    dbp_pre_morning_avg = pre_morning_df['DBP'].mean()

    sbp_morning_surge = sbp_morning_avg - sbp_pre_morning_avg
    dbp_morning_surge = dbp_morning_avg - dbp_pre_morning_avg

    return {
        'sbp_morning_surge': round(sbp_morning_surge, 2),
        'dbp_morning_surge': round(dbp_morning_surge, 2),
        'sbp_morning_avg': round(sbp_morning_avg, 2),
        'sbp_pre_morning_avg': round(sbp_pre_morning_avg, 2),
        'dbp_morning_avg': round(dbp_morning_avg, 2),
        'dbp_pre_morning_avg': round(dbp_pre_morning_avg, 2),
    }

def calc_extra_indices(day_stats, night_stats, full_df,
                       morning_start='06:00', morning_end='08:00',
                       pre_morning_start='03:00', pre_morning_end='06:00'):
    """
    在原有昼夜差值、下降率、极限差比值等基础上，额外计算更贴近临床定义的“晨峰血压”。
    """
    # 如果有任一为空，则直接返回空
    if day_stats['mean_sbp'] is None or night_stats['mean_sbp'] is None:
        return {}

    sbp_day = day_stats['mean_sbp']
    sbp_night = night_stats['mean_sbp']
    dbp_day = day_stats['mean_dbp']
    dbp_night = night_stats['mean_dbp']

    # 1) 昼夜差值
    sbp_diff = sbp_day - sbp_night
    dbp_diff = dbp_day - dbp_night

    # 2) 下降率 (Dip %) = (day_mean - night_mean) / day_mean * 100
    sbp_dip = (sbp_diff / sbp_day * 100) if sbp_day != 0 else None
    dbp_dip = (dbp_diff / dbp_day * 100) if dbp_day != 0 else None

    # 3) 昼夜比(极限差比值) = (night_mean / day_mean) * 100%
    sbp_ratio = (sbp_night / sbp_day * 100) if sbp_day != 0 else None
    dbp_ratio = (dbp_night / dbp_day * 100) if dbp_day != 0 else None

    # 4) 晨峰血压计算
    surge_info = calc_morning_surge(full_df,
                                    morning_start=morning_start,
                                    morning_end=morning_end,
                                    pre_start=pre_morning_start,
                                    pre_end=pre_morning_end)

    # 汇总
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

    return result

# ========== 这里是修改的重点：在饼图右下角添加一份图例说明，避免与饼图重叠 ==========
def plot_bp_six_pies(day_df, night_df, full_df, output_file="result/bp_6_pies.png"):
    """
    参考示例图片的布局，绘制6个饼图：
      1) 白天收缩压(设定参考范围)  2) 夜间收缩压(设定参考范围)  3) 全天收缩压
      4) 白天舒张压               5) 夜间舒张压               6) 全天舒张压

    每个饼图分为 高(H)/正常(N)/低(L) 三个区块。
    output_file: 保存的图片文件名

    在绘图后，会在图的底部(右下角)添加一个图例，标示这三种颜色的含义。
    """

    # 如果没有 result 文件夹，则创建
    if not os.path.exists("result"):
        os.makedirs("result")

    # 定义每个饼图的配置信息：(标题, 数据Series, (low, high))
    # 你可以根据自己临床或需求，分别设置白天/夜间/全天的参考区间。
    # 这里仅举例：
    configs = [
        ("白天 收缩压(90-135)", day_df['SBP'], (90, 135)),
        ("夜间 收缩压(80-120)", night_df['SBP'], (80, 120)),
        ("全天 收缩压(90-140)", full_df['SBP'], (90, 140)),
        ("白天 舒张压(60-85)", day_df['DBP'], (60, 85)),
        ("夜间 舒张压(50-70)", night_df['DBP'], (50, 70)),
        ("全天 舒张压(60-90)", full_df['DBP'], (60, 90)),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    axes = axes.ravel()  # 变成一维方便迭代

    # 为高(H)/正常(N)/低(L) 设置固定颜色和标签
    labels = ["H", "N", "L"]
    colors = ["tomato", "lightgreen", "lightskyblue"]  # 你可自行调整三种色块

    for i, (title, series, ref_range) in enumerate(configs):
        ax = axes[i]

        if series.empty:
            # 若该时段没有数据，直接空
            ax.set_title(f"{title}\n无数据")
            ax.axis('off')
            continue

        low, high = ref_range
        # 计算 H, N, L 三类计数
        count_H = (series > high).sum()
        count_L = (series < low).sum()
        count_N = len(series) - count_H - count_L

        # 注意：若都为0，会报错，这里做个保护
        total = count_H + count_N + count_L
        if total == 0:
            ax.set_title(f"{title}\n无数据")
            ax.axis('off')
            continue

        data = [count_H, count_N, count_L]
        # 绘制饼图
        patches, texts, autotexts = ax.pie(
            data,
            labels=labels,
            colors=colors,
            autopct=lambda pct: f"{pct:.1f}%" if pct > 0 else "",
            startangle=140
        )
        ax.set_title(f"{title}\n(N={total})", fontsize=10)

        # 可选：让文字更好看一点
        for text in texts:
            text.set_fontsize(9)
        for autot in autotexts:
            autot.set_fontsize(9)

    # ------- 新增：在右下角添加图例，避免与6个饼图重叠 -------
    from matplotlib.patches import Patch
    legend_patches = [
        Patch(color="tomato",      label="H(高)"),
        Patch(color="lightgreen",  label="N(正常)"),
        Patch(color="lightskyblue",label="L(低)")
    ]
    # 调整子图与图例的布局，避免重叠或被裁剪
    plt.subplots_adjust(bottom=0.15)
    # loc='lower right'：图例在整张图的右下角
    # bbox_to_anchor=(1, 0)：再精细微调，以免和坐标轴贴得太近
    fig.legend(
        handles=legend_patches,
        loc='lower right',
        bbox_to_anchor=(1, 0.02)  # 你可根据实际效果微调
    )
    # --------------------------------------------------

    plt.tight_layout()
    plt.savefig(output_file, dpi=100)
    plt.close()
    return output_file

def generate_plot(df, output_png='blood_pressure_plot.png'):
    """
    生成简单的可视化图表（时间-收缩压/舒张压折线图），并保存为PNG文件。
    """
    if df.empty:
        print("数据为空，无法生成图表。")
        return None

    # x轴为测量时间
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
    plt.tight_layout()
    plt.xticks(rotation=45)  # 旋转 x 轴刻度，防止重叠

    # 保存图表
    if not os.path.exists("result"):
        os.makedirs("result")
    plt.savefig(output_png, dpi=100)
    plt.close()
    return output_png

def generate_pdf_report(day_stats, night_stats, full_stats, extra_indices,
    df, day_df, night_df,
    plot_file='blood_pressure_plot.png',
    pie_6_file=None,
    output_pdf='report.pdf'):
    """
    使用reportlab生成PDF报告
    pie_6_file: 改进后的“六饼图”文件
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
    story.append(Paragraph("你好，我是黄家宝。", styles['Normal']))
    story.append(Paragraph("以下为基于所采集血压数据的统计分析，仅供个人家庭参考：", styles['Normal']))
    story.append(Spacer(1, 12))

    # ========== 血压汇总表展示 ==========
    table_data = []
    # 表头
    table_data.append(["统计项", "白天", "夜间", "全天"])

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
        f"{full_stats['sbp_arv']}" if full_stats['sbp_arv'] is not None else "N/A"
    ])
    table_data.append([
        "DBP平均真实变异(ARV)",
        f"{day_stats['dbp_arv']}" if day_stats['dbp_arv'] is not None else "N/A",
        f"{night_stats['dbp_arv']}" if night_stats['dbp_arv'] is not None else "N/A",
        f"{full_stats['dbp_arv']}" if full_stats['dbp_arv'] is not None else "N/A"
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
        text = f"""
        收缩压(白天): {extra_indices.get('sbp_day', 'N/A')} mmHg，夜间: {extra_indices.get('sbp_night', 'N/A')} mmHg，<br/>
        收缩压差值: {extra_indices.get('sbp_diff', 'N/A')} mmHg，下降率: {extra_indices.get('sbp_dip', 'N/A')}%，<br/>
        极限差比值(夜/昼): {extra_indices.get('sbp_ratio', 'N/A')}%，
        晨峰血压: {extra_indices.get('sbp_morning_surge', 'N/A')} mmHg<br/>
        <br/>
        舒张压(白天): {extra_indices.get('dbp_day', 'N/A')} mmHg，夜间: {extra_indices.get('dbp_night', 'N/A')} mmHg，<br/>
        舒张压差值: {extra_indices.get('dbp_diff', 'N/A')} mmHg，下降率: {extra_indices.get('dbp_dip', 'N/A')}%，<br/>
        极限差比值(夜/昼): {extra_indices.get('dbp_ratio', 'N/A')}%，
        晨峰血压: {extra_indices.get('dbp_morning_surge', 'N/A')} mmHg<br/>
        """
        story.append(Paragraph(text, styles['Normal']))
    story.append(Spacer(1, 20))

    # 在PDF中插入“六饼图”
    if pie_6_file and os.path.exists(pie_6_file):
        story.append(Paragraph("<b>血压高/正常/低分布（改进饼图）</b>", styles['Normal']))
        story.append(Spacer(1, 10))
        story.append(Image(pie_6_file, width=500, height=350))  # 可自行调大小
        story.append(Spacer(1, 20))
        story.append(PageBreak())

    # 若有折线图，则插入图
    if plot_file and os.path.exists(plot_file):
        story.append(Paragraph("<b>24小时趋势图</b>", styles['Normal']))
        story.append(Spacer(1, 10))
        story.append(Image(plot_file, width=400, height=250))
        story.append(Spacer(1, 20))

    # ===== 血压测量值明细表 =====
    story.append(Paragraph("<b>血压测量值明细</b>", styles['Normal']))

    # 表头
    detail_data = [["编号", "日期", "时间", "收缩压", "舒张压", "平均压", "脉率", "脉压差"]]
    for i, row in df.iterrows():
        sbp_str = mark_value(row['SBP'], NORMAL_SBP_RANGE[0], NORMAL_SBP_RANGE[1])
        dbp_str = mark_value(row['DBP'], NORMAL_DBP_RANGE[0], NORMAL_DBP_RANGE[1])

        date_str = row['DateTime'].strftime('%Y-%m-%d')
        time_str = row['DateTime'].strftime('%H:%M:%S')

        map_val = row.get('MAP', None)
        map_val = round(map_val, 2) if map_val is not None else ""

        detail_data.append([
            i + 1,          # 编号
            date_str,       # 日期
            time_str,       # 时间
            sbp_str,        # 收缩压(附带箭头)
            dbp_str,        # 舒张压(附带箭头)
            map_val,        # 平均压
            row['HR'],      # 脉率
            row['PP']       # 脉压差
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
    · 本报告基于所采集的人工测量数据进行简单分析，可能与真实的24h动态血压监测设备结果存在差异；<br/>
    · 若有异常结果，请结合临床或及时就医；<br/>
    · 更多指标或自定义分析，后续可根据需要扩展。<br/>
    """
    story.append(Paragraph(summary_text, styles['Normal']))

    # 生成 PDF
    doc.build(story)
    print(f"PDF 报告已生成: {output_pdf}")

def main():
    # 1. 读取Excel数据
    file_path = "data/blood_pressure-test1.xlsx"  # Excel文件名
    df = read_data_from_excel(file_path)

    # 2. 拆分白天、夜间数据
    day_df, night_df, full_df = split_day_night(df, day_start='08:00', night_start='23:00')

    # 3. 分别计算统计指标
    day_stats = calc_statistics(day_df, label='白天')
    night_stats = calc_statistics(night_df, label='夜间')
    full_stats = calc_statistics(full_df, label='全天')

    # 4. 计算额外指标（包含昼夜差值、下降率、晨峰血压等改进）
    extra_indices = calc_extra_indices(
        day_stats, night_stats, full_df,
        morning_start='06:00', morning_end='08:00',
        pre_morning_start='03:00', pre_morning_end='06:00'
    )

    # 5. 生成 24h 趋势图
    plot_file = generate_plot(full_df, output_png='result/blood_pressure_plot.png')

    # 6. 生成改进后的“六饼图”
    #    这里演示高/正常/低三分法，且白天/夜间各有不同参考范围
    six_pies_file = plot_bp_six_pies(day_df, night_df, full_df, output_file="result/bp_6_pies.png")

    # 7. 生成PDF报告
    generate_pdf_report(
        day_stats,
        night_stats,
        full_stats,
        extra_indices,
        full_df,
        day_df,
        night_df,
        plot_file=plot_file,
        pie_6_file=six_pies_file,
        output_pdf='result/blood_pressure_report.pdf'
    )

if __name__ == '__main__':
    main()
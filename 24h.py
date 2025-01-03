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
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os
from matplotlib import font_manager, rcParams
# 注册中文字体（宋体），以便在PDF和matplotlib图表中正常显示中文
pdfmetrics.registerFont(TTFont('SimSun', 'SimSun.ttf'))
rcParams['font.sans-serif'] = ['PingFang SC', 'SimHei', 'Songti SC']  # 这行可根据实际环境调整
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

    # 分别筛选
    day_df = df[df['DateTime'].dt.time.apply(is_daytime)]
    night_df = df[df['DateTime'].dt.time.apply(is_nighttime)]

    return day_df, night_df, df

def calc_statistics(df, label='全天'):
    """
    计算收缩压、舒张压、平均压、脉率等的基本统计量。
    返回一个字典，包含平均值、最大值及其出现时刻、最小值及其出现时刻、标准差、血压负荷等。
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
            'count': 0
        }

    # 平均动脉压（MAP）简单计算公式： MAP = DBP + (SBP - DBP) / 3
    # 也可用其它更精确的公式，这里仅做示例
    df['MAP'] = df['DBP'] + (df['SBP'] - df['DBP']) / 3.0

    mean_sbp = df['SBP'].mean()
    max_sbp_idx = df['SBP'].idxmax()
    min_sbp_idx = df['SBP'].idxmin()

    mean_dbp = df['DBP'].mean()
    max_dbp_idx = df['DBP'].idxmax()
    min_dbp_idx = df['DBP'].idxmin()

    stats = {
        'label': label,
        'mean_sbp': round(mean_sbp, 2),
        'max_sbp': int(df.loc[max_sbp_idx, 'SBP']),
        'max_sbp_time': df.loc[max_sbp_idx, 'DateTime'].strftime('%H:%M:%S'),
        'min_sbp': int(df.loc[min_sbp_idx, 'SBP']),
        'min_sbp_time': df.loc[min_sbp_idx, 'DateTime'].strftime('%H:%M:%S'),

        'mean_dbp': round(mean_dbp, 2),
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

        'count': len(df)
    }

    return stats

def calc_extra_indices(day_stats, night_stats):
    """
    计算额外指标：收缩压/舒张压的昼夜差值、下降率、极限差比值、晨峰血压等。
    可根据实际需求进行调整。
    """
    # 如果有任一为空，则返回空值
    if day_stats['mean_sbp'] is None or night_stats['mean_sbp'] is None:
        return {}

    sbp_day = day_stats['mean_sbp']
    sbp_night = night_stats['mean_sbp']
    dbp_day = day_stats['mean_dbp']
    dbp_night = night_stats['mean_dbp']

    # 差值
    sbp_diff = round(sbp_day - sbp_night, 2)
    dbp_diff = round(dbp_day - dbp_night, 2)

    # 下降率(Dip %) = (day_mean - night_mean) / day_mean * 100
    sbp_dip = round(sbp_diff / sbp_day * 100, 2) if sbp_day != 0 else 0
    dbp_dip = round(dbp_diff / dbp_day * 100, 2) if dbp_day != 0 else 0

    # 极限差比值(夜间/白天) * 100% （这里纯示例）
    sbp_ratio = round((sbp_night / sbp_day) * 100, 2) if sbp_day != 0 else None
    dbp_ratio = round((dbp_night / dbp_day) * 100, 2) if dbp_day != 0 else None

    # 晨峰血压：可根据自定义时段，如06:00-10:00与前夜均值对比，这里简化为(day - night)
    sbp_morning_surge = sbp_diff
    dbp_morning_surge = dbp_diff

    return {
        'sbp_day': sbp_day,
        'sbp_night': sbp_night,
        'sbp_diff': sbp_diff,
        'sbp_dip': sbp_dip,
        'sbp_ratio': sbp_ratio,
        'sbp_morning_surge': sbp_morning_surge,

        'dbp_day': dbp_day,
        'dbp_night': dbp_night,
        'dbp_diff': dbp_diff,
        'dbp_dip': dbp_dip,
        'dbp_ratio': dbp_ratio,
        'dbp_morning_surge': dbp_morning_surge,
    }

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
    plt.savefig(output_png, dpi=100)
    plt.close()
    return output_png

def generate_pdf_report(day_stats, night_stats, full_stats, extra_indices,
                        df, day_df, night_df, plot_file='blood_pressure_plot.png',
                        output_pdf='report.pdf'):
    """
    使用 reportlab 生成PDF报告，包括统计结果、图表以及血压测量值明细表。
    """
    doc = SimpleDocTemplate(output_pdf, pagesize=A4)
    styles = getSampleStyleSheet()

    # 设置 Normal 和 Title 样式字体为 SimSun
    styles['Normal'].fontName = 'SimSun'
    styles['Title'].fontName = 'SimSun'
    # 适当调大行距，避免中文字体可能出现的行间重叠
    styles['Normal'].leading = 16
    styles['Title'].leading = 20

    story = []

    # 标题
    story.append(Paragraph("24小时动态血压报告", styles['Title']))
    story.append(Spacer(1, 12))

    # 简要说明
    story.append(Paragraph("以下为基于所采集血压数据的统计分析，仅供参考：", styles['Normal']))
    story.append(Spacer(1, 12))

    # 组装三段统计表格: 白天、夜间、全天
    table_data = []
    table_data.append(["统计项", "白天", "夜间", "全天"])
    table_data.append([
        "平均SBP",
        f"{day_stats['mean_sbp']}" if day_stats['mean_sbp'] else "N/A",
        f"{night_stats['mean_sbp']}" if night_stats['mean_sbp'] else "N/A",
        f"{full_stats['mean_sbp']}" if full_stats['mean_sbp'] else "N/A"
    ])
    table_data.append([
        "平均DBP",
        f"{day_stats['mean_dbp']}" if day_stats['mean_dbp'] else "N/A",
        f"{night_stats['mean_dbp']}" if night_stats['mean_dbp'] else "N/A",
        f"{full_stats['mean_dbp']}" if full_stats['mean_dbp'] else "N/A"
    ])
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

    # 显示额外指标
    story.append(Paragraph("<b>额外指标：</b>", styles['Normal']))
    if extra_indices:
        text = f"""
        收缩压(白天): {extra_indices['sbp_day']} mmHg，夜间: {extra_indices['sbp_night']} mmHg，<br/>
        收缩压差值: {extra_indices['sbp_diff']} mmHg，下降率: {extra_indices['sbp_dip']}%，<br/>
        极限差比值(夜/昼): {extra_indices['sbp_ratio']}%，晨峰血压: {extra_indices['sbp_morning_surge']} mmHg<br/>
        <br/>
        舒张压(白天): {extra_indices['dbp_day']} mmHg，夜间: {extra_indices['dbp_night']} mmHg，<br/>
        舒张压差值: {extra_indices['dbp_diff']} mmHg，下降率: {extra_indices['dbp_dip']}%，<br/>
        极限差比值(夜/昼): {extra_indices['dbp_ratio']}%，晨峰血压: {extra_indices['dbp_morning_surge']} mmHg<br/>
        """
        story.append(Paragraph(text, styles['Normal']))
    story.append(Spacer(1, 20))

    # 若有折线图，则插入图
    if plot_file and os.path.exists(plot_file):
        story.append(Image(plot_file, width=400, height=250))
        story.append(Spacer(1, 20))

    # ===== 新增：血压测量值明细表（编号、日期、时间、收缩压、舒张压、平均压、脉率、脉压差） =====
    story.append(Paragraph("<b>血压测量值明细</b>", styles['Normal']))

    # 表头
    detail_data = [["编号", "日期", "时间", "收缩压", "舒张压", "平均压", "脉率", "脉压差"]]

    # 遍历所有行，这里可根据需求，只显示前N条或全部
    for i, row in df.iterrows():
        sbp_str = mark_value(row['SBP'], NORMAL_SBP_RANGE[0], NORMAL_SBP_RANGE[1])
        dbp_str = mark_value(row['DBP'], NORMAL_DBP_RANGE[0], NORMAL_DBP_RANGE[1])

        # 取日期和时间
        date_str = row['DateTime'].strftime('%Y-%m-%d')
        time_str = row['DateTime'].strftime('%H:%M:%S')

        # MAP 如果没有则显示空，否则保留两位小数
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
    - 本报告基于所采集的人工测量数据进行简单分析，可能与真实的24h动态血压监测设备结果存在差异；<br/>
    - 若有异常结果，请结合临床或及时就医；<br/>
    - 更多指标或自定义分析，后续可根据需要扩展。<br/>
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

    # 3. 计算统计指标
    day_stats = calc_statistics(day_df, label='白天')
    night_stats = calc_statistics(night_df, label='夜间')
    full_stats = calc_statistics(full_df, label='全天')

    # 4. 计算额外指标
    extra_indices = calc_extra_indices(day_stats, night_stats)

    # 5. 生成可视化图表
    plot_file = generate_plot(full_df, output_png='result/blood_pressure_plot.png')

    # 6. 生成PDF报告
    generate_pdf_report(day_stats, night_stats, full_stats, extra_indices,
                        full_df, day_df, night_df,
                        plot_file=plot_file,
                        output_pdf='result/blood_pressure_report.pdf')


if __name__ == '__main__':
    main()
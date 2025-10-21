import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import quote_plus
from datetime import datetime, timedelta
import time
import os
import re


host = '10.34.195.135'
port = 19030  # MySQL 默认端口
user = 'gdmo'  # 用户名
password = 'gdmo@123!!'  # 密码
database = 'tenant_4100002'  # 数据库名
table = 'ods_yczy'  # 表名

password_encoded = quote_plus(password)

engine = create_engine(f'mysql+pymysql://{user}:{password_encoded}@{host}:{port}/{database}')

save_dir = r'D:\xz2_yczy_data'
os.makedirs(save_dir, exist_ok=True)

def fetch_yczy_data(battery_sql_start_time: datetime,
                            battery_sql_end_time: datetime):

    sql = f"""
            SELECT *
            FROM {table}
            WHERE end_time BETWEEN '{battery_sql_start_time.strftime('%Y-%m-%d %H:%M:%S')}'
                      AND '{battery_sql_end_time.strftime('%Y-%m-%d %H:%M:%S')}'
            """

    df = pd.read_sql(sql, engine)
    start_record_time = battery_sql_start_time.strftime('%Y%m%d')
    end_record_time = battery_sql_end_time.strftime('%Y%m%d')

    csv_filename = os.path.join(save_dir, f'data_{start_record_time}_{end_record_time}.csv')
    df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
    print(
        f"{battery_sql_start_time}~{battery_sql_end_time} 范围内查询到 {len(df)} 条有效电芯数据")


def merge_and_split_by_product_quality():
    if 'save_dir' not in globals() and 'save_dir' not in locals():
        raise RuntimeError("未定义 save_dir，请先设置 save_dir 变量为文件夹路径")

    csv_pattern = re.compile(r"^data_\d{8}_\d{8}\.csv$")
    csv_files = [f for f in os.listdir(save_dir) if csv_pattern.match(f)]
    if not csv_files:
        print("未找到符合条件的 CSV 文件")
        return

    df_list = []
    for file in csv_files:
        file_path = os.path.join(save_dir, file)
        try:
            df_list.append(pd.read_csv(file_path, encoding='utf-8-sig'))
        except Exception as e:
            print(f"读取文件失败: {file_path} -> {e}")

    if not df_list:
        print("没有成功读取任何 CSV 文件")
        return

    merged_df = pd.concat(df_list, ignore_index=True)
    # 只保留 product_quality == 1
    if 'product_quality' not in merged_df.columns:
        print("合并后的数据没有 'product_quality' 字段，无法按质量过滤")
        return

    merged_df = merged_df[merged_df['product_quality'] == 1].reset_index(drop=True)
    print(f"合并并筛选 product_quality==1 后总行数: {len(merged_df)}")

    # merged_out = os.path.join(save_dir, "merged_yczy_data.csv")
    # merged_df.to_csv(merged_out, index=False, encoding='utf-8-sig')
    # print(f"已保存合并文件: {merged_out}")

    group_cols = ['technics_line_name', 'device_code', 'ryczy407']

    missing_cols = [c for c in group_cols if c not in merged_df.columns]
    if missing_cols:
        print(f"数据中缺少以下分组列，无法按要求分组: {missing_cols}")
        return

    before = len(merged_df)
    merged_df = merged_df.dropna(subset=group_cols).reset_index(drop=True)
    after = len(merged_df)
    dropped = before - after
    print(f"在分组前已去除 {dropped} 行（因为 {group_cols} 中存在 NaN），剩余 {after} 行用于分组")

    if merged_df.empty:
        print("删除 NaN 后没有数据可分组")
        return

    invalid_chars_re = re.compile(r'[\\/*?:"<>|]')
    max_name_len = 180

    for keys, group_df in merged_df.groupby(group_cols):
        # keys 是一个元组 (technics_line_name, device_code, ryczy407)
        line_name, device_code, ryczy = keys
        safe_line = invalid_chars_re.sub("_", str(line_name)).strip() or "unknown_line"
        safe_device = invalid_chars_re.sub("_", str(device_code)).strip() or "unknown_device"
        safe_ryczy = invalid_chars_re.sub("_", str(ryczy)).strip() or "unknown_ryczy"

        # 合并成文件名，并截断以防过长
        filename_base = f"merged_{safe_line}_{safe_device}_{safe_ryczy}"
        if len(filename_base) > max_name_len:
            filename_base = filename_base[:max_name_len]

        output_file = os.path.join(save_dir, f"{filename_base}.csv")
        try:
            group_df.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"已保存: {output_file} ({len(group_df)} 行)")
        except Exception as e:
            print(f"保存失败: {output_file} -> {e}")


if __name__ == '__main__':

    # total_start_date = datetime(2025, 9, 4)
    # total_end_date = datetime(2025, 10, 21)
    # current_day = total_start_date
    #
    # while current_day < total_end_date:
    #     next_day = current_day + timedelta(days=1)
    #     print(f"\n=== 正在处理时间段：{current_day} ~ {next_day} ===")
    #     fetch_yczy_data(battery_sql_start_time=current_day,
    #                      battery_sql_end_time=next_day)
    #
    #     current_day = next_day

    merge_and_split_by_product_quality()



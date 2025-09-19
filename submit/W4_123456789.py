# -*- coding: utf-8 -*-
"""
Pandas 基礎操作課堂練習：學生期中成績分析
"""

import pandas as pd

def load_and_explore_data(file_path):
    """任務一：讀取 CSV 並初步探索資料"""
    df = pd.read_csv(file_path, encoding='utf-8-sig')

    # 顯示前 5 筆資料
    print("--- 前五筆資料 ---")
    print(df.head())

    # 查看資料結構（欄位、型態、缺失值）
    print("\n--- 資料結構 ---")
    print(df.info())
    print("\n--- 缺失值統計 ---")
    print(df.isnull().sum())

    return df

def feature_engineering(df):
    """計算總分、平均分數與是否及格"""
    
    # 計算總分
    df["總分"] = df[["數學","英文","國文","自然","社會"]].sum(axis=1)

    # 計算平均分數
    df["平均"] = df[["數學","英文","國文","自然","社會"]].mean(axis=1)

    # 新增是否及格欄位 (平均 >= 60 即及格)
    df["是否及格"] = df["平均"] >= 60

    return df

def filter_and_analyze_data(df):
    """篩選資料與統計"""
    
    # 找出數學成績 < 60 的學生
    math_fail = df[df["數學"] < 60]
    print("\n--- 數學不及格學生 ---")
    print(math_fail)

    # 找出班級為 'A' 且英文 > 90 的學生
    a_high_english = df[(df["班級"]=="A") & (df["英文"]>90)]
    print("\n--- A班英文 > 90 的學生 ---")
    print(a_high_english)

    # 顯示統計摘要
    print("\n--- 統計摘要 ---")
    print(df.describe())

    # 找出總分最高的學生
    top_student = df.loc[df["總分"].idxmax()]
    print("\n--- 總分最高的學生 ---")
    print(top_student[["姓名","總分"]])

    return df

def save_results(df, output_file_path):
    """儲存為 CSV"""
    df.to_csv(output_file_path, index=False, encoding='utf-8-sig')
    print(f"\n結果已儲存至 {output_file_path}")

if __name__ == "__main__":
    INPUT_CSV = "grades.csv"
    OUTPUT_CSV = "grades_analyzed.csv"

    df = load_and_explore_data(INPUT_CSV)
    df = feature_engineering(df)
    df = filter_and_analyze_data(df)
    save_results(df, OUTPUT_CSV)

    print("\n完成所有分析任務")

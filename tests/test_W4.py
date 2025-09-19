# -*- coding: utf-8 -*-
import pytest
import pandas as pd
import importlib.util
from pathlib import Path

# -------------------------
# 取得學生 PR 新增檔案
# -------------------------
SUBMIT_DIR = Path(__file__).parent / "submit"

# 假設 PR 已驗證過只有一個檔案，且命名符合 W4_學號.py
student_files = list(SUBMIT_DIR.glob("W4_*.py"))
if not student_files:
    raise FileNotFoundError("❌ submit/ 目錄下沒有學生提交檔案")

student_file = student_files[0]

# 動態 import 學生的程式
spec = importlib.util.spec_from_file_location("student_submission", student_file)
student_submission = importlib.util.module_from_spec(spec)
spec.loader.exec_module(student_submission)

# -------------------------
# 測試用 DataFrame
# -------------------------
@pytest.fixture
def sample_df():
    data = {
        "姓名": ["Alice","Bob","Charlie","David","Eva"],
        "性別": ["F","M","M","M","F"],
        "班級": ["A","B","A","C","A"],
        "數學": [95,55,60,45,88],
        "英文": [88,70,92,60,95],
        "國文": [78,82,85,50,90],
        "自然": [90,65,80,40,85],
        "社會": [85,60,70,55,92],
    }
    return pd.DataFrame(data)

# -------------------------
# 功能測試
# -------------------------
def test_feature_engineering(sample_df):
    df = student_submission.feature_engineering(sample_df.copy())

    # 總分
    assert "總分" in df.columns
    assert df.loc[0, "總分"] == 95+88+78+90+85

    # 平均分數
    assert "平均" in df.columns
    assert df.loc[0, "平均"] == pytest.approx((95+88+78+90+85)/5)

    # 是否及格
    assert "是否及格" in df.columns
    assert df.loc[1, "是否及格"] is False
    assert df.loc[0, "是否及格"] is True

def test_filter_and_analyze_data(sample_df):
    df = student_submission.feature_engineering(sample_df.copy())
    df = student_submission.filter_and_analyze_data(df)

    # 數學不及格
    math_failed = df[df['數學']<60]
    assert len(math_failed) == 2

    # A班英文 > 90
    high_A = df[(df['班級']=='A') & (df['英文']>90)]
    assert len(high_A) == 2

    # 總分最高
    top_student = df.loc[df['總分'].idxmax()]
    assert top_student["姓名"] == "Alice"

def test_save_results(tmp_path, sample_df):
    df = student_submission.feature_engineering(sample_df.copy())
    output_file = tmp_path / "grades_test.csv"
    student_submission.save_results(df, output_file)

    # CSV 檔案存在
    assert output_file.exists()

    # 讀回檢查欄位
    df_read = pd.read_csv(output_file, encoding='utf-8-sig')
    for col in ["總分","平均","是否及格"]:
        assert col in df_read.columns

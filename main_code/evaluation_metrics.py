import pandas as pd
import numpy as np


def preprocess_errors(df):
    """将包含多列的错误条目拆分为多行"""
    # 统一分隔符并拆分
    df['column'] = df['column'].str.replace(';', ',').str.split(',')
    df = df.explode('column')

    # 清理空格并过滤空值
    df['column'] = df['column'].str.strip()
    df = df[df['column'].ne('')]

    # 转换为整数行号
    df['row'] = df['row'].astype(int)
    return df.drop_duplicates().reset_index(drop=True)


def calculate_metrics(true_errors, detected_errors, total_records):
    # 预处理数据
    true_errors = preprocess_errors(true_errors)
    detected_errors = preprocess_errors(detected_errors)

    # 创建索引集合
    true_set = set(zip(true_errors['row'], true_errors['column']))
    detected_set = set(zip(detected_errors['row'], detected_errors['column']))

    # 计算指标
    tp = len(true_set & detected_set)
    fp = len(detected_set - true_set)
    fn = len(true_set - detected_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "precision": round(precision, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn
    }



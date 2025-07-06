import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

def analyze_anomaly_detection_results(file_path):
    """
    分析时间序列异常检测结果，同时计算基于单个点和基于连续异常段两种规则的捕获情况。

    Args:
        file_path (str): CSV 文件的路径。文件应包含 'y', 'x', 'recon_x', 'ub', 'lb' 列。

    Returns:
        dict: 包含各项计算结果的字典。
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"错误：文件 '{file_path}' 未找到。请检查文件路径。")
        return None
    except Exception as e:
        print(f"读取文件时发生错误：{e}")
        return None

    # 1. 确保所需列存在
    required_columns = ['y', 'x', 'recon_x', 'day_ub', 'day_lb','hour_ub','hour_lb']
    if not all(col in df.columns for col in required_columns):
        missing_cols = [col for col in required_columns if col not in df.columns]
        print(f"错误：CSV 文件缺少以下必需列：{missing_cols}")
        return None

    # --- 基础统计 ---

    # 异常点总数 (y=1)
    total_anomalies = df['y'].sum()

    # 异常点占比
    anomaly_percentage = (total_anomalies / len(df)) * 100 if len(df) > 0 else 0

    # 判断每个点是否被边界捕获 (即 x 在 ub 或 lb 之外)
    df['is_captured_by_boundary'] = (((df['x'] < df['day_lb']) | (df['x'] > df['day_ub'])) & ((df['x'] < df['hour_lb']) | (df['x'] > df['hour_ub'])))
    # df['is_captured_by_boundary'] = (df['x'] > df['ub'])

    # --- 1. 基于单个点规则的捕获数量 ---
    # 只有当 y=1 且 x 在边界外时，才算被捕获
    count_anomalies_individually_captured = df[(df['y'] == 0) & df['is_captured_by_boundary']].shape[0]

    # 占比计算
    percentage_anomalies_individually_captured_of_total = \
        (count_anomalies_individually_captured / total_anomalies) * 100 if total_anomalies > 0 else 0

    # --- 2. 基于连续异常段规则的捕获数量 ---
    # 如果连续异常段中有一个点被捕获，则整段的异常点都算被捕获
    count_anomalies_captured_by_segment_rule = 0

    if total_anomalies > 0:
        # 识别连续的真实异常段
        df['segment_change'] = (df['y'] != df['y'].shift()).cumsum()

        # 遍历每个真实的异常段 (y=1的段)
        for segment_id, segment_df in df[df['y'] == 1].groupby('segment_change'):
            # 如果该连续异常段中存在任何一个点被边界捕获
            if segment_df['is_captured_by_boundary'].any():
                # 则该段中的所有异常点都算作被捕获
                count_anomalies_captured_by_segment_rule += len(segment_df)

    # 占比计算
    percentage_anomalies_captured_by_segment_rule_of_total = \
        (count_anomalies_captured_by_segment_rule / total_anomalies) * 100 if total_anomalies > 0 else 0

    # --- 预测准确性指标 ---

    # MAE (Mean Absolute Error)
    mae = mean_absolute_error(df['x'], df['recon_x'])

    # MSE (Mean Squared Error)
    mse = mean_squared_error(df['x'], df['recon_x'])

    results = {
        "总数据点数": len(df),
        "异常点总数 (y=1)": total_anomalies,
        "异常点占比 (%)": f"{anomaly_percentage:.2f}%",
        "--- 异常点捕获情况 (基于**单个点**的规则) ---": "",
        "被边界 (ub/lb) 单独捕获的异常点数量": count_anomalies_individually_captured,
        "被边界单独捕获的异常点占异常点总数的比重 (%)": f"{percentage_anomalies_individually_captured_of_total:.2f}%",
        "--- 异常点捕获情况 (基于**连续异常段**的规则) ---": "",
        "被边界 (ub/lb) 捕获的异常点数量 (基于'一个点被捕获则整段被捕获'的规则)": count_anomalies_captured_by_segment_rule,
        "被边界捕获的异常点占异常点总数的比重 (%) (基于段规则)": f"{percentage_anomalies_captured_by_segment_rule_of_total:.2f}%",
        "--- 预测模型性能 ---": "",
        "MAE (x vs recon_x)": f"{mae:.4f}",
        "MSE (x vs recon_x)": f"{mse:.4f}"
    }

    return results

if __name__ == "__main__":
    # --- 使用示例 ---
    # 假设你的CSV文件名为 'your_data.csv' 并且在当前目录下
    # 请将 'your_data.csv' 替换为你实际的文件路径
    csv_file = 'Yahoo/result.csv'

    analysis_results = analyze_anomaly_detection_results(csv_file)

    if analysis_results:
        print("\n--- 异常检测结果分析 ---")
        for key, value in analysis_results.items():
            print(f"{key}: {value}")

        print("\n--- 解释 ---")
        print("这个分析脚本同时提供了两种异常点捕获情况的评估：")
        print("1. **基于单个点规则**：")
        print("   - 这是一种严格的衡量方式，只有当一个真实异常点 ('y'=1) 的实际值 ('x') 确实超出了预测的正常范围 ('ub'/'lb') 时，才认为它被模型捕获。它直接反映了模型在每个时间点上的边界准确性。")
        print("2. **基于连续异常段规则**：")
        print("   - 这是一种更注重“事件”层面检测的衡量方式。如果一个连续的真实异常点序列中，哪怕只有一个点被模型成功识别（即超出了边界），那么该序列中的所有异常点都被认为是被捕获的。这在实际应用中很有意义，因为用户往往更关心是否能发现整个异常事件，而不是事件中的每个微小点。")
        print("\n您可以使用这些指标来评估您的模型在不同严格程度下的性能，并根据您的业务需求选择更合适的评估标准。")
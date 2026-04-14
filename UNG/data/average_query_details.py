import pandas as pd
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Calculate global average metrics from query details CSV.")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to the input query_details_repeatX.csv")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to save the averaged metrics CSV")
    args = parser.parse_args()

    if not os.path.exists(args.input_csv):
        print(f"❌ 找不到输入文件: {args.input_csv}")
        return

    # 读取明细数据
    df = pd.read_csv(args.input_csv)

    # 剔除对求平均没有实际统计意义的列
    if 'QueryID' in df.columns:
        df = df.drop(columns=['QueryID'])

    # 确保只对数值型列求平均，避免因潜在的字符串/布尔列报错
    numeric_df = df.select_dtypes(include='number')

    # 直接对所有剩余的数值列求全局平均值
    # .mean() 会返回一个 Series，.to_frame().T 会将其转换回单行的 DataFrame
    df_avg = numeric_df.mean().to_frame().T

    # 保存结果
    df_avg.to_csv(args.output_csv, index=False)
    print(f"✅ 已成功计算所有指标的全局平均值 (单行数据)，并保存至: {args.output_csv}")

if __name__ == "__main__":
    main()
import pandas as pd
import os

def process_csv_with_pandas(input_file, output_file, value, operation='sub'):
    """
    读取 CSV，对 'Average_Time_ms' 列进行加减乘除处理，并保存。

    参数:
    input_file (str): 输入路径
    output_file (str): 输出路径
    value (float): 用于计算的数值
    operation (str): 操作模式 -> 'add'(加), 'sub'(减), 'mul'(乘), 'div'(除)
    """
    try:
        # 1. 检查文件
        if not os.path.exists(input_file):
            print(f"错误：找不到输入文件 '{input_file}'")
            return

        # 2. 读取CSV
        df = pd.read_csv(input_file)

        # 3. 检查列
        target_col = 'Average_Time_ms'
        if target_col not in df.columns:
            print(f"错误：输入文件中没有找到 '{target_col}' 列。")
            return

        # 4. 转为数值型（处理可能存在的非数字字符）
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce')

        # 5. 根据 operation 执行不同的数学运算
        print(f"正在对 '{target_col}' 列执行 [{operation}] 操作，数值: {value}...")

        if operation == 'sub':  # 减法
            df[target_col] = df[target_col] - value
        elif operation == 'add':  # 加法
            df[target_col] = df[target_col] + value
        elif operation == 'mul':  # 乘法
            df[target_col] = df[target_col] * value
        elif operation == 'div':  # 除法
            if value == 0:
                print("错误：除数不能为 0！")
                return
            df[target_col] = df[target_col] / value
        else:
            print(f"错误：未知的操作模式 '{operation}'。请使用 'add', 'sub', 'mul', 或 'div'。")
            return

        # 6. 格式化保留4位小数 (可选，为了由浮点数转为整洁的字符串输出)
        df[target_col] = df[target_col].map('{:.4f}'.format)

        # 7. 保存文件
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"处理完成！已保存到 '{output_file}'")

    except pd.errors.EmptyDataError:
        print(f"错误：文件 '{input_file}' 是空的。")
    except Exception as e:
        print(f"发生未知错误: {e}")

# --- 主程序配置区 ---
if __name__ == "__main__":
    # 文件路径
    FILE_PATH = '/home/fengxiaoyao/FilterVector/FilterVectorResults/Music/Results/pre-filter/Index[M32_LB100_alpha1.2_C6_EP16_AN1511563_AM32_AMB64_AG80]_GT[GT_query_select_imp_A_B_C-weighted-sub-base-123456789_K20]_Search[Ls1000-Le40000-Lp1000_efsS300-efss300-efsf300-lt5000_K20_th100]/results/search_time_summary.csv'
    
    # 输入和输出可以是同一个文件（覆盖），也可以是不同文件
    INPUT_FILENAME = FILE_PATH
    OUTPUT_FILENAME = FILE_PATH 
    ADJUST_VALUE = 1.5      # 想操作的数值
    MODE = 'mul'              # 'add' = 加 (+), 'sub' = 减 (-), 'mul' = 乘 (*), 'div' = 除 (/)

    process_csv_with_pandas(INPUT_FILENAME, OUTPUT_FILENAME, ADJUST_VALUE, MODE)
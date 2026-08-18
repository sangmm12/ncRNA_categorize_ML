import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import os
import sys
import numpy as np
import argparse

module_path = '/home/hgq/BioData/hgq/NovelNaturalVectorAcceleration250414'
#module_path = '/home/hgq/BioData/hgq/NV_old'
if module_path not in sys.path:
    sys.path.append(module_path)

from common import calculate_values, calculate_k
from basicSetting import VALID_CHARS
from asymmetricNV import calculate_NV_16, calculate_NV_16_py, calculate_NV_64, calculate_NV_64_py
from asymmetricNV import calculate_NV_256, calculate_NV_256_py, calculate_NV_1024, calculate_NV_1024_py

#################################################
# Define functions for calculating moments and k-values
def calculate_moments_new(sequence, avg_positions, seq_len, nucleotide_counts, max_k):
    moments = {k: [0, 0, 0, 0] for k in range(2, max_k + 1)}
    for i, nt in enumerate(sequence):
        for j, k in enumerate("ACGT"):
            if nt == k:
                for n in range(2, max_k + 1):
                    moments[n][j] += (((i + 1) - avg_positions[j]) ** n) / (nucleotide_counts[j] ** n)
    return moments

def calculate_k(sequence, max_k):
    nucleotide_counts = [sequence.count("A"), sequence.count("C"), sequence.count("G"), sequence.count("T")]
    seq_len = len(sequence)
    avg_positions = []
    for nt in "ACGT":
        positions = [(i + 1) for i, base in enumerate(sequence) if base == nt]
        avg_positions.append(sum(positions) / len(positions) if positions else 0)

    moments = calculate_moments_new(sequence, avg_positions, seq_len, nucleotide_counts, max_k)
    nucleotide_vector = nucleotide_counts + avg_positions
    for k in range(2, max_k + 1):
        nucleotide_vector += moments[k]

    return nucleotide_vector

def process_seq(sequence):
    n, mu, D2 = calculate_values(sequence)
    NV_8 = list(n.values()) + list(mu.values())

    #NV_16   = calculate_NV_16_py(sequence, mu, n, abs_flag=False)
    #NV_64   = calculate_NV_64_py(sequence, mu, n, abs_flag=False)
    #NV_256  = calculate_NV_256_py(sequence, mu, n, abs_flag=False)
    #NV_1024 = calculate_NV_1024_py(sequence, mu, n, abs_flag=False)

    #NV_16   = calculate_NV_16_py(sequence, mu, n, kernel_flag=2) #square
    #NV_64   = calculate_NV_64_py(sequence, mu, n, kernel_flag=2)
    #NV_256  = calculate_NV_256_py(sequence, mu, n, kernel_flag=2)
    #NV_1024 = calculate_NV_1024_py(sequence, mu, n, kernel_flag=2)
    NV_16   = calculate_NV_16_py(sequence, mu, n, kernel_flag=1) #abs
    NV_64   = calculate_NV_64_py(sequence, mu, n, kernel_flag=1)
    NV_256  = calculate_NV_256_py(sequence, mu, n, kernel_flag=1)
    NV_1024 = calculate_NV_1024_py(sequence, mu, n, kernel_flag=1)

    NV_24   = NV_8 + NV_16
    NV_88   = NV_24 + NV_64
    NV_344  = NV_88 + NV_256
    NV_1368 = NV_344 + NV_1024
    return NV_1368

def clean_sequence(index, sequence, valid_chars):
    # 转换为大写并替换U为T
    sequence = sequence.upper().replace('U', 'T')

    cleaned_sequence = ''
    removed_chars = []  # 记录被移除的字符

    for char in sequence:
        if char not in valid_chars:
            if char != 'N':  # N不算无效字符
                removed_chars.append(char)
        else:
            cleaned_sequence += char

    # 打印被移除的无效字符
    #if removed_chars:
    #    print(f"Invalid characters found at {index} and removed: {set(removed_chars)}")

    return cleaned_sequence

def process_row(args):
    """处理单行数据，接收包含索引和行的元组"""
    index, row = args
    try:
        Seq = row['Sequence']
        cleaned_sequence = clean_sequence(index, Seq.upper(), VALID_CHARS)
        NV = process_seq(cleaned_sequence)
        return index, NV

    except TypeError as e:
        print(f"TypeError at row {index}: {e}")
        return index, None

    except AttributeError as e:
        print(f"AttributeError at row {index}: {e}")
        return index, None

    except Exception as e:
        print(f"Error processing row {index}: {e}")
        return index, None

def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='Process sequences and calculate NV vectors')
    parser.add_argument('--input', required=True, help='Input CSV file path')
    parser.add_argument('--output', required=True, help='Output CSV file path')
    args = parser.parse_args()
    
    # 读取数据
    print(f"Reading data from: {args.input}")
    data = pd.read_csv(args.input)
    
    print(data.head())
    print(f"Data shape: {data.shape}")
    print(f"Valid characters: {VALID_CHARS}")

    # 准备处理任务 - 保持顺序
    tasks = [(index, row) for index, row in data.iterrows()]
    
    # 处理数据（并行）- 使用map保持顺序
    nv_data = [None] * len(data)  # 预分配列表以保持顺序
    
    print("Processing rows...")
    with ProcessPoolExecutor() as executor:
        # 使用map保持顺序
        results = list(tqdm(executor.map(process_row, tasks), total=len(tasks), desc="Processing rows"))
    
    # 处理结果
    none_count = 0
    error_examples = []
    
    for index, result in results:
        if result is not None:
            nv_data[index] = (index, result)
        else:
            nv_data[index] = (index, None)
            none_count += 1
            error_examples.append((index, data.iloc[index].to_dict()))

    print(f"\n⚠️ Skipped {none_count} rows with None results.")

    # 显示错误示例
    if error_examples:
        print("\n😪 Example of problematic rows:")
        for i, (idx, row_data) in enumerate(error_examples[:5]):
            print(f"Row index: {idx}")
            print(row_data)
            print("-" * 40)

    # 分离有效和无效数据
    valid_data = [(idx, nv) for idx, nv in nv_data if nv is not None]
    invalid_indices = [idx for idx, nv in nv_data if nv is None]
    
    if invalid_indices:
        invalid_data = [(idx, data.iloc[idx]) for idx in invalid_indices]
        print(f"\n⚠️ Skipped {len(invalid_data)} rows with None results.")

        # 显示错误示例
        if invalid_data:
            print("😪 Example of problematic rows:")
            for i, (idx, row) in enumerate(invalid_data[:5]):
                print(f"Index: {idx}")
                print(row)
                print("-" * 50)

            # 保存失败的行到文件
            failed_df = pd.DataFrame([row for _, row in invalid_data])
            failed_path = os.path.splitext(args.output)[0] + '_failed.csv'
            failed_df.to_csv(failed_path, index=False)
            print(f"💾 Saved failed rows to {failed_path}")

    # 检查是否有有效结果
    if not valid_data:
        print("❌ No valid NV vectors were returned. Exiting.")
        sys.exit(1)

    # 提取NV值并创建DataFrame
    nv_values = [nv for _, nv in valid_data]
    nv_columns = ['V' + str(i + 1) for i in range(len(nv_values[0]))]
    nv_df = pd.DataFrame(nv_values, columns=nv_columns)

    print("NV DataFrame head:")
    print(nv_df.head())
    print(f"NV DataFrame shape: {nv_df.shape}")

    # 合并原始数据和NV数据
    # 注意：由于我们跳过了某些行，需要重新索引
    valid_indices = [idx for idx, nv in nv_data if nv is not None]
    processed_data = pd.concat([data.iloc[valid_indices].reset_index(drop=True), nv_df], axis=1)

    # 保存处理后的数据
    print(f"Saving processed data to: {args.output}")
    processed_data.to_csv(args.output, index=False)

    print("Processed data head:")
    print(processed_data.head())
    print(f"Processed data shape: {processed_data.shape}")
    print(f"✅ Processed data saved to {args.output}")

if __name__ == "__main__":
    main()

import pandas as pd

# 文件路径
input_file = 'rnacentral_active_with_NV1368.csv'
output_file = 'rnacentral_active_NV1368_lncRNA_less200.csv'

# 分块处理参数
chunksize = 100000  # 每次处理的行数，根据内存调整

print(f"[INFO] Start processing large file with chunksize={chunksize}")

# 用于统计的变量
total_removed = 0
total_processed = 0
chunk_count = 0

# 第一次写入，创建文件并写入header
first_chunk = True

# 分块读取和处理
for chunk in pd.read_csv(input_file, chunksize=chunksize):
    chunk_count += 1
    before_rows = len(chunk)
    total_processed += before_rows
    
    # === 对 lncRNA 进行长度过滤 ===
    # 确保 Length 可数值化
    chunk['Length'] = pd.to_numeric(chunk['Length'], errors='coerce')
    mask_lnc = (chunk['Label'] == 'lncRNA')
    mask_short = mask_lnc & (chunk['Length'] < 200)
    
    # 删除短lncRNA记录
    chunk = chunk[~mask_short]
    after_rows = len(chunk)
    removed_in_chunk = before_rows - after_rows
    total_removed += removed_in_chunk
    
    # 写入文件
    if first_chunk:
        chunk.to_csv(output_file, index=False)
        first_chunk = False
    else:
        chunk.to_csv(output_file, mode='a', header=False, index=False)
    
    print(f"[INFO] Chunk {chunk_count}: Removed {removed_in_chunk} rows, Remaining {after_rows} rows")

print(f"\n[INFO] Processing completed!")
print(f"[INFO] Total processed rows: {total_processed}")
print(f"[INFO] Total removed rows: {total_removed}")
print(f"[INFO] Final output saved to: {output_file}")

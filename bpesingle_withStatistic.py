#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from collections import Counter
from typing import List, Dict, Tuple

import pandas as pd
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

#file = "rnacentral_active.csv"
file = "rnacentral_active_top20_2w_seed42.csv"

VALID_CHARS = 'ACGT'

def clean_sequence(sequence):
    """Clean DNA sequence by converting to uppercase, replacing U with T, and removing invalid characters."""
    if isinstance(sequence, str):
        # 处理单个字符串
        cleaned_sequence = ''
        sequence = sequence.upper()
        
        for char in sequence:
            if char == 'U':
                cleaned_sequence += 'T'
            elif char in VALID_CHARS:
                cleaned_sequence += char
            elif char != 'N':
                print(f"Removed invalid character: {char}")
        
        return cleaned_sequence
    else:
        # 处理 pandas Series，对每个元素应用清理函数
        return sequence.apply(clean_sequence)

# 1) 收集特定文件夹的训练数据
def collect_sequences(folder_path: str) -> List[str]:
    sequences = []
    train_file = os.path.join(folder_path, f"{file}")
    if os.path.exists(train_file):
        df = pd.read_csv(train_file)
        seq = clean_sequence(df['Sequence'].astype(str))
        sequences.extend(seq.tolist())
    return sequences

# 2) 训练 BPE 分词器
def train_bpe(sequences: List[str], vocab_size: int = 1000, max_token_length: int = None) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    
    # 设置训练参数
    trainer_kwargs = {
        "vocab_size": vocab_size,
        "min_frequency": 2
    }
    
    # 如果设置了最大token长度，添加到训练参数中
    if max_token_length is not None:
        trainer_kwargs["max_token_length"] = max_token_length
    
    trainer = trainers.BpeTrainer(**trainer_kwargs)
    tokenizer.train_from_iterator(sequences, trainer=trainer)
    return tokenizer

# 统计：对训练序列编码，统计 token 频次/概率，并补齐 vocab 中从未出现的 token（count=0）
def profile_tokenizer(tokenizer: Tokenizer, sequences: List[str]) -> pd.DataFrame:
    counts = Counter()
    total_tokens = 0
    for seq in sequences:
        enc = tokenizer.encode(seq)
        toks = enc.tokens
        counts.update(toks)
        total_tokens += len(toks)

    if total_tokens == 0:
        raise ValueError("训练序列编码后没有任何 token，检查输入数据是否为空。")

    vocab = tokenizer.get_vocab()  # {token: id}

    rows = []
    for tok, cnt in counts.items():
        tok_id = vocab.get(tok, -1)
        prob = cnt / total_tokens
        rows.append((tok, tok_id, cnt, prob, len(tok)))  # 添加token长度
    df = pd.DataFrame(rows, columns=["token", "id", "count", "prob", "token_length"])

    # 补齐 vocab 中未出现的 token
    if len(vocab) > len(df):
        existing = set(df["token"])
        extra = [(tok, tid, 0, 0.0, len(tok)) for tok, tid in vocab.items() if tok not in existing]
        if extra:
            df = pd.concat([df, pd.DataFrame(extra, columns=["token", "id", "count", "prob", "token_length"])],
                           ignore_index=True)

    # 排序 + 累计占比
    df = df.sort_values("prob", ascending=False).reset_index(drop=True)
    df["rank_by_prob"] = df.index + 1
    df["cum_prob"] = df["prob"].cumsum()
    return df

def _pct(x: float) -> str:
    return f"{x*100:.4f}%"

def print_brief(size: int, df_stats: pd.DataFrame, max_token_length: int = None):
    n_tokens = len(df_stats)
    total_count = int(df_stats["count"].sum())
    min_prob = df_stats["prob"].min()
    median_prob = df_stats["prob"].median()
    p1 = df_stats["prob"].quantile(0.01)
    p5 = df_stats["prob"].quantile(0.05)
    
    # 添加token长度统计
    max_len = df_stats["token_length"].max()
    avg_len = df_stats["token_length"].mean()
    
    limit_info = f" (限制: {max_token_length})" if max_token_length is not None else ""
    print(f"\n===== Vocab Size = {size}{limit_info} =====")
    print(f"Token 总数: {n_tokens} | 总 token 数: {total_count}")
    print(f"Token 长度 - 最大: {max_len:.1f} | 平均: {avg_len:.2f}")
    print(f"最小概率: {_pct(min_prob)} | 中位数: {_pct(median_prob)}")
    print(f"P1分位: {_pct(p1)} | P5分位: {_pct(p5)}")

    tail = df_stats.sort_values("prob", ascending=True).head(10)
    print("[最稀有 TOP-10 token]")
    for _, r in tail.iterrows():
        print(f" id={int(r['id']):>4} | len={r['token_length']} | prob={_pct(r['prob'])} | token={repr(r['token'])}")

def save_vocab_txt(tokenizer: Tokenizer, out_path: str):
    vocab = tokenizer.get_vocab()
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Token\tID\tLength\n")
        for token, idx in sorted_vocab:
            f.write(f"{token}\t{idx}\t{len(token)}\n")

# 3) 写出 tokenized 文件
def process_file(input_file: str, output_file: str, tokenizer: Tokenizer):
    df = pd.read_csv(input_file)
    seq_ = clean_sequence(df['Sequence'].astype(str)) 
    tokenized = []
    for s in seq_:
        enc = tokenizer.encode(s)
        tokenized.append("|".join(enc.tokens))
    df['sequence'] = tokenized
    df.to_csv(output_file, index=False)
    print(f"[SAVE] {output_file} ({len(df)} rows)")

def parse_sizes(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/hgq/BioData/hgq/ncRNA/", help="根目录")
    ap.add_argument("--folder", default="rnacentral", help="数据子目录")
    ap.add_argument("--vocab-size", type=int, default=500, help="主分词器词表大小（用于导出）")
    ap.add_argument("--probe-sizes", type=str, default="", help="可选：逗号分隔多规模统计，如 '400,800,1000,1200,1500'")
    ap.add_argument("--limit", type=int, default=10, help="限制单个token的最大长度（字符数）")
    args = ap.parse_args()

    root_dir = args.root
    target_folder = args.folder
    output_dir = os.path.join(root_dir, f"{target_folder}_tokenized")
    #os.makedirs(output_dir, exist_ok=True)
    
    # 添加限制信息到输出目录名
    if args.limit is not None:
        output_dir = output_dir + f"_limit{args.limit}"
        os.makedirs(output_dir, exist_ok=True)

    print("收集训练数据...")
    folder_path = os.path.join(root_dir, target_folder)
    sequences = collect_sequences(folder_path)
    print(f"{folder_path}\n 共收集到 {len(sequences)} 条序列")
    if len(sequences) == 0:
        raise ValueError("未从训练集中收集到任何序列，请检查路径与数据。")

    # 显示限制信息
    if args.limit is not None:
        print(f"\n[INFO] 设置最大token长度限制: {args.limit} 字符")
    else:
        print(f"\n[INFO] 未设置token长度限制")

    # （A）可选：多词表规模探测，仅做统计，不导出 tokenized
    probe_sizes = parse_sizes(args.probe_sizes) if args.probe_sizes else []
    summary_rows = []
    for size in probe_sizes:
        print(f"\n[INFO] 训练探测分词器 (vocab_size={size}, limit={args.limit}) ...")
        tok_probe = train_bpe(sequences, vocab_size=size, max_token_length=args.limit)
        df_stats = profile_tokenizer(tok_probe, sequences)
        print_brief(size, df_stats, args.limit)

        stats_path = os.path.join(output_dir, f"bpe_stats_{size}.csv")
        df_stats.to_csv(stats_path, index=False)
        print(f"[SAVE] {stats_path} (rows={len(df_stats)})")

        row = {
            "vocab_size": size,
            "max_token_length": args.limit,
            "num_tokens": len(df_stats),
            "total_token_count": int(df_stats["count"].sum()),
            "max_token_len": df_stats["token_length"].max(),
            "avg_token_len": df_stats["token_length"].mean(),
            "min_prob": df_stats["prob"].min(),
            "median_prob": df_stats["prob"].median(),
            "p01_prob": df_stats["prob"].quantile(0.01),
            "p05_prob": df_stats["prob"].quantile(0.05),
            "coverage_top100": float(df_stats.sort_values("prob", ascending=False)["prob"].head(100).sum()) if len(df_stats) >= 100 else float(df_stats["prob"].sum()),
            "coverage_top200": float(df_stats.sort_values("prob", ascending=False)["prob"].head(200).sum()) if len(df_stats) >= 200 else float(df_stats["prob"].sum()),
        }
        summary_rows.append(row)

    if summary_rows:
        df_summary = pd.DataFrame(summary_rows)
        summary_path = os.path.join(output_dir, "bpe_size_summary.csv")
        df_summary.to_csv(summary_path, index=False)
        print(f"\n[SAVE] 多规模汇总: {summary_path}")
        # 控制台友好展示
        def _fmt_col(col):
            return df_summary[col].map(lambda x: _pct(x) if isinstance(x, float) and col.endswith('_prob') or 'coverage' in col else f"{x:.2f}" if isinstance(x, float) else x)
        show = df_summary.copy()
        for c in show.columns:
            if c in ["min_prob","median_prob","p01_prob","p05_prob","coverage_top100","coverage_top200"]:
                show[c] = _fmt_col(c)
            elif c in ["max_token_len", "avg_token_len"]:
                show[c] = _fmt_col(c)
        print(show.to_string(index=False))

    # （B）主模型：使用 vocab_size 导出 tokenized 与词表，并对其做同样的统计
    vocab_size = args.vocab_size
    print(f"\n[INFO] 训练主分词器 (vocab_size={vocab_size}, limit={args.limit}) 并导出 ...")
    tokenizer = train_bpe(sequences, vocab_size, max_token_length=args.limit)
    print("BPE 训练完成!")

    train_in = os.path.join(folder_path, f"{file}")
    train_out = os.path.join(output_dir, "train_tokenized.csv")
    if os.path.exists(train_in):
        process_file(train_in, train_out, tokenizer)

    # 保存词表
    vocab_file = os.path.join(output_dir, f"bpe_vocab_{vocab_size}.txt")
    save_vocab_txt(tokenizer, vocab_file)
    print(f"[SAVE] 词表: {vocab_file}")

    # 主模型的 token 统计
    print("\n[INFO] 统计主分词器 token 概率分布 ...")
    df_stats_main = profile_tokenizer(tokenizer, sequences)
    print_brief(vocab_size, df_stats_main, args.limit)
    stats_main_path = os.path.join(output_dir, f"bpe_stats_{vocab_size}.csv")
    df_stats_main.to_csv(stats_main_path, index=False)
    print(f"[SAVE] {stats_main_path} (rows={len(df_stats_main)})")

    print(f"\n处理完成! 结果已保存到: {output_dir}")
    if probe_sizes:
        print("- 多规模统计：bpe_stats_<size>.csv + bpe_size_summary.csv")
    print(f"- 主规模统计：{os.path.basename(stats_main_path)}")
    print(f"- 词表与 tokenized：{os.path.basename(vocab_file)}, {os.path.basename(train_out)}")
    
    # 显示最终的token长度统计
    max_len = df_stats_main["token_length"].max()
    print(f"- 最终token长度范围: 1 - {max_len} 字符")

if __name__ == "__main__":
    main()

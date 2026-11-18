#!/usr/bin/env python3
"""多进程并行helper(2026-09-10借鉴Sequoia-X工程): scanner逐股循环加速.

用法: scanner把逐股逻辑提成模块级函数work(code)->dict|None,然后:
    from scripts.core.parallel import process_map
    hits = [h for h in process_map(work, codes) if h]
macOS spawn安全: work必须在模块顶层定义(可pickle),返回纯dict。
名称/ST过滤等需要外部映射的逻辑放主进程做,worker只算数。
"""
import multiprocessing as mp
import os

N_PROC = min(8, os.cpu_count() or 4)


def process_map(work, codes, n_proc=None, chunk=64):
    """并行跑work(code),保序返回结果列表(None保留占位,调用方过滤)."""
    n_proc = n_proc or N_PROC
    if len(codes) < 500 or n_proc <= 1:  # 小任务不值得开进程
        return [work(c) for c in codes]
    with mp.Pool(n_proc) as pool:
        return pool.map(work, codes, chunksize=chunk)

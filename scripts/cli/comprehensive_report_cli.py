#!/usr/bin/env python3
"""综合报告数据收集编排器 — 顺序调用CLI工具，输出结构化JSON数据包。

用法:
    python scripts/cli/comprehensive_report_cli.py 603881
    python scripts/cli/comprehensive_report_cli.py 603881 --wave-from 2024-09-18
    python scripts/cli/comprehensive_report_cli.py 603881 --name 数据港

输出: JSON 数据包文件，包含所有工具的结构化输出。
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).parent.parent.parent)
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, 'scripts', 'cli')


def run_tool(name: str, cmd: list, timeout: int = 120) -> dict:
    """运行一个CLI工具，捕获输出。"""
    start = time.time()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            cwd=PROJECT_ROOT, env={**os.environ, 'PYTHONPATH': PROJECT_ROOT}
        )
        elapsed = round(time.time() - start, 1)
        success = result.returncode == 0
        return {
            'success': success,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'returncode': result.returncode,
            'duration_seconds': elapsed,
            'error': result.stderr[:500] if not success else None,
        }
    except subprocess.TimeoutExpired:
        elapsed = round(time.time() - start, 1)
        return {
            'success': False, 'stdout': '', 'stderr': '',
            'returncode': -1, 'duration_seconds': elapsed,
            'error': f'Timeout after {timeout}s',
        }
    except Exception as e:
        return {
            'success': False, 'stdout': '', 'stderr': '',
            'returncode': -1, 'duration_seconds': 0,
            'error': str(e),
        }


def extract_json_from_stdout(stdout: str) -> object:
    """从混合文本输出中提取最后一个JSON对象。"""
    # 方法：从末尾反向找第一个 {，然后尝试解析
    text = stdout.rstrip()
    idx = text.rfind('\n{')
    if idx >= 0:
        json_str = text[idx + 1:]
    elif text.startswith('{'):
        json_str = text
    else:
        return None
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None


def main():
    parser = argparse.ArgumentParser(description='综合报告数据收集编排器')
    parser.add_argument('symbol', help='股票代码（如 603881）')
    parser.add_argument('--name', help='股票名称（可选，自动从数据中提取）')
    parser.add_argument('--wave-from', dest='wave_from', default=None,
                        help='波浪分析起始日期 (YYYY-MM-DD)')
    parser.add_argument('--output-dir', default=None, help='输出目录')
    args = parser.parse_args()

    symbol = args.symbol.strip()
    name = args.name or ''
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    date_str = datetime.now().strftime('%Y-%m-%d')

    out_dir = args.output_dir or os.path.join(PROJECT_ROOT, 'reports', date_str)
    os.makedirs(out_dir, exist_ok=True)

    print(f'{"=" * 60}')
    print(f' 综合分析数据收集: {symbol}')
    print(f' 时间: {timestamp}')
    print(f'{"=" * 60}')

    tools_run = []
    tools_failed = []
    data_package = {
        'meta': {
            'symbol': symbol, 'name': name, 'timestamp': timestamp,
            'tools_run': tools_run, 'tools_failed': tools_failed,
            'wave_from': args.wave_from,
        },
        'tech_analysis': None,
        'chanlun': None,
        'wave_analysis': None,
        'volume': None,
        'news': None,
        'decision': None,
        'visual_report_path': None,
    }

    # ── Step 1: 技术分析（JSON输出，必须先运行以获取股票名称）──
    print('\n[1/7] 技术分析 (4D评分 + MACD双参数 + 基本面)...')
    r = run_tool('tech_analysis', [sys.executable,
                os.path.join(SCRIPTS_DIR, 'tech_analysis_cli.py'), symbol, '--json'],
                timeout=180)
    if r['success']:
        parsed = extract_json_from_stdout(r['stdout'])
        if parsed:
            data_package['tech_analysis'] = {'success': True, 'data': parsed}
            tools_run.append('tech_analysis')
            # 提取股票名称
            if not name and parsed.get('realtime', {}).get('name'):
                name = parsed['realtime']['name']
                data_package['meta']['name'] = name
            print(f'  ✓ 成功 ({r["duration_seconds"]}s) — {name} {parsed.get("price", "?")}元')
        else:
            data_package['tech_analysis'] = {'success': False, 'data': None, 'raw_stdout': r['stdout'][:2000]}
            tools_failed.append('tech_analysis')
            print(f'  ✗ JSON解析失败 ({r["duration_seconds"]}s)')
    else:
        data_package['tech_analysis'] = {'success': False, 'error': r['error']}
        tools_failed.append('tech_analysis')
        print(f'  ✗ 失败: {r["error"]}')

    # ── Step 2: 缠论分析 ──
    print('\n[2/7] 缠论分析 (笔/线段/中枢/买卖点)...')
    r = run_tool('chanlun', [sys.executable,
                os.path.join(SCRIPTS_DIR, 'chanlun_cli.py'), symbol,
                '--levels', 'daily,60min'],
                timeout=120)
    data_package['chanlun'] = {
        'success': r['success'], 'raw_text': r['stdout'],
        'duration_seconds': r['duration_seconds'],
    }
    (tools_run if r['success'] else tools_failed).append('chanlun')
    status = '✓' if r['success'] else '✗'
    print(f'  {status} {"成功" if r["success"] else r["error"][:100]} ({r["duration_seconds"]}s)')

    # ── Step 3: 量价分析 ──
    print('\n[3/7] 量价分析 (量价模式/OBV/MFI)...')
    r = run_tool('volume', [sys.executable,
                os.path.join(SCRIPTS_DIR, 'volume_cli.py'), symbol, '--json'],
                timeout=120)
    if r['success']:
        parsed = extract_json_from_stdout(r['stdout'])
        data_package['volume'] = {
            'success': True, 'data': parsed,
            'duration_seconds': r['duration_seconds'],
        }
        tools_run.append('volume')
        print(f'  ✓ 成功 ({r["duration_seconds"]}s)')
    else:
        data_package['volume'] = {'success': False, 'raw_text': r['stdout'], 'error': r['error']}
        tools_failed.append('volume')
        print(f'  ✗ 失败: {r["error"][:100]}')

    # ── Step 4: 新闻公告 ──
    print('\n[4/7] 新闻公告...')
    news_cmd = [sys.executable, os.path.join(SCRIPTS_DIR, 'news_cli.py'), symbol]
    if name:
        news_cmd.extend(['--name', name])
    r = run_tool('news', news_cmd, timeout=60)
    data_package['news'] = {
        'success': r['success'], 'raw_text': r['stdout'],
        'duration_seconds': r['duration_seconds'],
    }
    (tools_run if r['success'] else tools_failed).append('news')
    status = '✓' if r['success'] else '✗'
    print(f'  {status} {"成功" if r["success"] else r["error"][:100]} ({r["duration_seconds"]}s)')

    # ── Step 5: 交易决策 ──
    print('\n[5/7] 交易决策 (买入/卖出/持有)...')
    decision_cmd = [sys.executable, os.path.join(SCRIPTS_DIR, 'make_decision.py'),
                    symbol]
    if name:
        decision_cmd.append(name)
    decision_cmd.extend(['--format', 'md'])
    r = run_tool('decision', decision_cmd, timeout=180)
    data_package['decision'] = {
        'success': r['success'], 'raw_text': r['stdout'],
        'duration_seconds': r['duration_seconds'],
    }
    (tools_run if r['success'] else tools_failed).append('decision')
    status = '✓' if r['success'] else '✗'
    print(f'  {status} {"成功" if r["success"] else r["error"][:100]} ({r["duration_seconds"]}s)')

    # ── Step 6: 波浪分析（可选）──
    if args.wave_from:
        print(f'\n[6/7] 波浪分析 (从 {args.wave_from} 开始)...')
        r = run_tool('wave', [sys.executable,
                    os.path.join(SCRIPTS_DIR, 'wave_analysis_cli.py'),
                    symbol, '--from', args.wave_from],
                    timeout=120)
        if r['success']:
            parsed = extract_json_from_stdout(r['stdout'])
            data_package['wave_analysis'] = {
                'success': True, 'data': parsed,
                'duration_seconds': r['duration_seconds'],
            }
            tools_run.append('wave_analysis')
            print(f'  ✓ 成功 ({r["duration_seconds"]}s)')
        else:
            data_package['wave_analysis'] = {
                'success': False, 'error': r['error'],
                'duration_seconds': r['duration_seconds'],
            }
            tools_failed.append('wave_analysis')
            print(f'  ✗ 失败: {r["error"][:100]}')
    else:
        print('\n[6/7] 波浪分析 — 跳过（未提供 --wave-from 参数）')

    # ── Step 7: 可视化报告 ──
    print('\n[7/7] 可视化报告...')
    visual_cmd = [sys.executable, os.path.join(SCRIPTS_DIR, 'visual_report_cli.py'), symbol]
    if name:
        visual_cmd.extend(['--name', name])
    r = run_tool('visual', visual_cmd, timeout=120)
    if r['success']:
        # 从输出中提取文件路径
        for line in r['stdout'].split('\n'):
            if '.html' in line:
                data_package['visual_report_path'] = line.strip()
                break
        tools_run.append('visual_report')
        print(f'  ✓ 成功 ({r["duration_seconds"]}s)')
    else:
        tools_failed.append('visual_report')
        print(f'  ✗ 失败: {r["error"][:100]}')

    # ── 写入 JSON 数据包 ──
    pkg_filename = f'{name or symbol}{symbol}_data_package.json'
    pkg_path = os.path.join(out_dir, pkg_filename)
    with open(pkg_path, 'w', encoding='utf-8') as f:
        json.dump(data_package, f, ensure_ascii=False, indent=2)

    # ── 汇总 ──
    print(f'\n{"=" * 60}')
    print(f' 数据收集完成')
    print(f' 成功: {len(tools_run)}/{len(tools_run) + len(tools_failed)} 工具')
    if tools_failed:
        print(f' 失败: {", ".join(tools_failed)}')
    print(f' 数据包: {pkg_path}')
    print(f'{"=" * 60}')

    return pkg_path


if __name__ == '__main__':
    main()

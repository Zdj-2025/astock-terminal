#!/usr/bin/env python3
"""
通知模块 - 通过Server酱推送消息到微信

使用方法:
1. 注册 https://sct.ftqq.com 获取SendKey
2. 在 config/notification_config.json 中填入SendKey
3. 调用 Notifier.send(title, content) 发送通知

作者: AI Assistant
日期: 2026-04-08
"""
import json
import os
import requests
from datetime import datetime
from typing import Optional, List
import sys

sys.path.insert(0, '/mnt/d/claude-code/stock')


class Notifier:
    """
    统一通知模块

    支持: Server酱（微信推送）
    预留: 飞书、钉钉、邮件等通道
    """

    def __init__(self, config_path: str = 'config/notification_config.json'):
        self.config = self._load_config(config_path)
        self.enabled = self.config.get('enabled', False)
        self.channel = self.config.get('channel', 'serverchan')
        self.serverchan_key = self.config.get('serverchan', {}).get('sendkey', '')

        if self.enabled and self.serverchan_key:
            print(f"✅ 通知模块已启用（{self.channel}）")
        elif self.enabled:
            print(f"⚠️  通知已启用但SendKey为空，请在 {config_path} 中配置")
            self.enabled = False
        else:
            print(f"ℹ️  通知模块未启用")

    def _load_config(self, config_path: str) -> dict:
        """加载通知配置"""
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"⚠️  通知配置加载失败: {e}")
        return {}

    def send(self, title: str, content: str = '') -> bool:
        """
        发送通知

        参数:
            title: 通知标题
            content: 通知内容（支持Markdown）

        返回:
            是否发送成功
        """
        if not self.enabled:
            print(f"  [通知未启用] {title}")
            return False

        if self.channel == 'serverchan':
            return self._send_serverchan(title, content)
        else:
            print(f"⚠️  不支持的通知通道: {self.channel}")
            return False

    def send_alert(self, alert_type: str, symbol: str, name: str, details: str) -> bool:
        """
        发送预警通知（止损/止盈等）

        参数:
            alert_type: 预警类型 (stop_loss / take_profit / price_alert)
            symbol: 股票代码
            name: 股票名称
            details: 详细信息
        """
        icons = {
            'stop_loss': '🔴',
            'take_profit': '🟢',
            'price_alert': '🟡',
        }
        icon = icons.get(alert_type, '⚠️')

        title = f"{icon} {alert_type.upper()}: {symbol} {name}"
        content = f"**{name} ({symbol})**\n\n{details}\n\n---\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M')}"

        return self.send(title, content)

    def send_daily_summary(self, decisions: list) -> bool:
        """
        发送每日分析摘要

        参数:
            decisions: 决策结果列表
        """
        title = f"📊 每日持仓分析 ({datetime.now().strftime('%m-%d')})"

        lines = []
        buy_count = sum(1 for d in decisions if d.get('action') == 'buy')
        sell_count = sum(1 for d in decisions if d.get('action') == 'sell')
        hold_count = sum(1 for d in decisions if d.get('action') == 'hold')

        lines.append(f"**总持仓**: {len(decisions)}只 | 买入{buy_count} | 卖出{sell_count} | 观望{hold_count}\n")

        for d in decisions:
            action_icons = {'buy': '📈', 'sell': '📉', 'hold': '⏸️'}
            icon = action_icons.get(d.get('action', 'hold'), '⏸️')
            score = d.get('overall_score', 0)
            action_text = d.get('action_text', '观望')

            line = f"- {icon} **{d.get('symbol', '')} {d.get('name', '')}** "
            line += f"| 评分{score:.0f} | {action_text}"

            # ETF专用信息
            if d.get('is_etf') and d.get('trading_advice'):
                trade = d['trading_advice']
                if trade.get('action') != 'wait':
                    line += f"\n  → 交易: {trade.get('action_text', '')}"

            lines.append(line)

        lines.append(f"\n---\n⚠️ 仅供参考，不构成投资建议")

        return self.send(title, '\n'.join(lines))

    def _send_serverchan(self, title: str, content: str) -> bool:
        """通过Server酱发送"""
        try:
            url = f"https://sctapi.ftqq.com/{self.serverchan_key}.send"
            data = {
                'title': title,
                'desp': content
            }
            resp = requests.post(url, data=data, timeout=10)
            result = resp.json()

            if result.get('code') == 0:
                print(f"  ✅ 通知已发送: {title}")
                return True
            else:
                print(f"  ❌ 通知发送失败: {result.get('message', '未知错误')}")
                return False

        except Exception as e:
            print(f"  ❌ Server酱调用失败: {str(e)[:50]}")
            return False


# ── 测试 ──

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='发送测试通知')
    parser.add_argument('--init', action='store_true', help='创建配置文件模板')
    args = parser.parse_args()

    if args.init:
        # 创建配置模板
        config = {
            "enabled": False,
            "channel": "serverchan",
            "serverchan": {
                "sendkey": "YOUR_SENDKEY_HERE"
            },
            "schedule": {
                "daily_analysis_time": "09:45",
                "alert_check_interval_minutes": 30
            }
        }
        config_path = 'config/notification_config.json'
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"✅ 配置模板已创建: {config_path}")
        print(f"   请编辑此文件，填入你的SendKey并设置 enabled=true")

    elif args.test:
        notifier = Notifier()
        if notifier.enabled:
            notifier.send("测试通知", "这是一条来自A股分析系统的测试消息。\n\n如果你看到了，说明通知功能正常！✅")
        else:
            print("通知未启用。请先运行: python scripts/utils/notifier.py --init")
    else:
        parser.print_help()

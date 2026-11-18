#!/usr/bin/env python3
"""读图工具 — 2026-10-03立: 用glm-5.3-flash(多模态)读本地图片,绕过主模型无视觉的限制。

背景: 主模型glm-5.3收不了图块,此前读截图靠外部MCP;实测glm-5.3-flash(裸名,勿带[1m]——
flash不支持该后缀会报1211模型不存在)通过Anthropic兼容口可正常识图。
用法:
  python scripts/tools/read_image.py <图片路径> [问题]
  python scripts/tools/read_image.py /tmp/shot.jpg "提取总资产和持仓明细"
"""
import sys, json, base64, urllib.request

def read_image(path, question='详细描述这张图片的所有文字和数字'):
    s = json.load(open('/Users/river/.claude/settings.json'))
    env = s.get('env', {})
    tok = env.get('ANTHROPIC_AUTH_TOKEN', '')
    base = env.get('ANTHROPIC_BASE_URL', '')
    with open(path, 'rb') as f:
        img = base64.b64encode(f.read()).decode()
    media = 'image/png' if path.lower().endswith('.png') else 'image/jpeg'
    req = urllib.request.Request(base + '/v1/messages',
        data=json.dumps({'model': 'glm-5.3-flash', 'max_tokens': 3000,
            'messages': [{'role': 'user', 'content': [
                {'type': 'image', 'source': {'type': 'base64', 'media_type': media, 'data': img}},
                {'type': 'text', 'text': question}]}]}).encode(),
        headers={'x-api-key': tok, 'anthropic-version': '2023-06-01',
                 'content-type': 'application/json'})
    r = json.load(urllib.request.urlopen(req, timeout=120))
    parts = [b.get('text', '') for b in r.get('content', []) if b.get('type') == 'text']
    return '\n'.join(parts)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    q = sys.argv[2] if len(sys.argv) > 2 else '详细描述这张图片的所有文字和数字，逐项列出，不要遗漏数字'
    print(read_image(sys.argv[1], q))

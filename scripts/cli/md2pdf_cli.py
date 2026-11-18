"""Markdown → PDF 转换工具（带中文排版样式）

用法:
    python scripts/cli/md2pdf_cli.py reports/2026-05-31/报告.md
    python scripts/cli/md2pdf_cli.py reports/2026-05-31/报告.md --output /tmp/报告.pdf
    python scripts/cli/md2pdf_cli.py reports/2026-05-31/报告.md --theme dark
    python scripts/cli/md2pdf_cli.py reports/2026-05-31/*.md  # 批量转换
"""
import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

CSS_LIGHT = """
@page { size: A4; margin: 2cm 2.5cm; }
body { font-family: "PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif;
       font-size: 11pt; line-height: 1.8; color: #222; }
h1 { font-size: 20pt; color: #1a1a2e; border-bottom: 3px solid #e94560; padding-bottom: 8px; }
h2 { font-size: 15pt; color: #16213e; border-bottom: 1.5px solid #0f3460; padding-bottom: 5px; margin-top: 25px; }
h3 { font-size: 12.5pt; color: #0f3460; margin-top: 18px; }
h4 { font-size: 11pt; color: #533483; }
table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 9.5pt; }
th { background: #2d3436; color: white; padding: 6px 8px; text-align: left; }
td { border: 1px solid #ddd; padding: 5px 8px; }
tr:nth-child(even) { background: #f8f8f8; }
blockquote { border-left: 4px solid #e94560; padding: 8px 12px; color: #555; background: #fff5f5; border-radius: 0 4px 4px 0; }
code { background: #eef0f2; color: #333; padding: 1px 4px; border-radius: 3px; font-size: 9.5pt; }
pre { background: #f4f6f8; color: #222; padding: 14px 16px; border-radius: 6px;
      font-size: 9.5pt; border-left: 3px solid #0f3460; overflow-x: auto; line-height: 1.6; }
pre code { background: none; color: #222; padding: 0; }
hr { border: none; border-top: 2px solid #e94560; margin: 25px 0; }
strong { color: #1a1a2e; }
"""

CSS_DARK = """
@page { size: A4; margin: 2cm 2.5cm; }
body { font-family: "PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif;
       font-size: 11pt; line-height: 1.8; color: #d0d0d0; background: #1e1e2e; }
h1 { font-size: 20pt; color: #f5c2e7; border-bottom: 3px solid #f38ba8; padding-bottom: 8px; }
h2 { font-size: 15pt; color: #89b4fa; border-bottom: 1.5px solid #45475a; padding-bottom: 5px; margin-top: 25px; }
h3 { font-size: 12.5pt; color: #89dceb; margin-top: 18px; }
h4 { font-size: 11pt; color: #cba6f7; }
table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 9.5pt; }
th { background: #313244; color: #cdd6f4; padding: 6px 8px; text-align: left; }
td { border: 1px solid #45475a; padding: 5px 8px; }
tr:nth-child(even) { background: #1e1e2e; }
blockquote { border-left: 4px solid #f38ba8; padding: 8px 12px; color: #a6adc8; background: #181825; border-radius: 0 4px 4px 0; }
code { background: #313244; color: #a6e3a1; padding: 1px 4px; border-radius: 3px; font-size: 9.5pt; }
pre { background: #181825; color: #cdd6f4; padding: 14px 16px; border-radius: 6px;
      font-size: 9.5pt; border-left: 3px solid #89b4fa; overflow-x: auto; line-height: 1.6; }
pre code { background: none; color: #cdd6f4; padding: 0; }
hr { border: none; border-top: 2px solid #f38ba8; margin: 25px 0; }
strong { color: #f5c2e7; }
"""

THEMES = {
    "light": CSS_LIGHT,
    "dark": CSS_DARK,
}


def convert(md_path: str, pdf_path: str = None, theme: str = "light") -> str:
    """Convert a Markdown file to PDF with Chinese-friendly styling."""
    md_path = Path(md_path)
    if not md_path.exists():
        raise FileNotFoundError(f"File not found: {md_path}")

    if pdf_path is None:
        pdf_path = str(md_path.with_suffix('.pdf'))

    # Step 1: pandoc MD → HTML
    result = subprocess.run(
        ['pandoc', str(md_path), '--to', 'html5', '--standalone', '--metadata', 'title='],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"pandoc failed: {result.stderr}")
    html = result.stdout

    # Step 2: inject CSS
    css = THEMES.get(theme, CSS_LIGHT)
    html = html.replace('</head>', f'<style>{css}</style></head>')

    # Step 3: write temp HTML
    tmp_html = md_path.with_suffix('.tmp.html')
    tmp_html.write_text(html, encoding='utf-8')

    # Step 4: weasyprint HTML → PDF
    try:
        from weasyprint import HTML
        HTML(filename=str(tmp_html)).write_pdf(pdf_path)
    except OSError as e:
        if "cannot load library" in str(e):
            print("[ERROR] weasyprint 需要系统库。请运行：")
            print("  brew install pango glib")
            sys.exit(1)
        raise
    finally:
        tmp_html.unlink(missing_ok=True)

    size_kb = Path(pdf_path).stat().st_size / 1024
    print(f"[OK] {md_path.name} -> {Path(pdf_path).name} ({size_kb:.0f}KB)")
    return pdf_path


def main():
    parser = argparse.ArgumentParser(description="Markdown → PDF 转换工具（中文排版）")
    parser.add_argument("input", nargs="+", help="Markdown 文件路径（支持多个）")
    parser.add_argument("--output", "-o", help="输出 PDF 路径（仅单个文件时有效）")
    parser.add_argument("--theme", choices=["light", "dark"], default="light", help="配色主题（默认 light）")
    args = parser.parse_args()

    if args.output and len(args.input) > 1:
        print("[ERROR] --output 仅在转换单个文件时可用")
        sys.exit(1)

    for md_file in args.input:
        pdf_file = args.output if args.output else None
        convert(md_file, pdf_file, theme=args.theme)


if __name__ == "__main__":
    main()

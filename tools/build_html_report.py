"""将 Markdown 报告转换为独立 HTML（图片 base64 内嵌）。

生成的 HTML 文件可在任何电脑上双击用浏览器打开，无需额外软件。

用法：
    python tools/build_html_report.py
"""
import base64
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
MD_PATH = _ROOT / "docs" / "full_report.md"
IMG_DIR = _ROOT / "docs" / "images"
OUT_PATH = _ROOT / "docs" / "full_report.html"


def md_to_html(md_text: str) -> str:
    """极简 Markdown → HTML 转换（支持标题/表格/图片/代码块/列表/粗体）。"""
    lines = md_text.split("\n")
    html_lines = []
    in_code = False
    in_table = False
    in_list = False

    for line in lines:
        # 代码块
        if line.strip().startswith("```"):
            if in_code:
                html_lines.append("</code></pre>")
                in_code = False
            else:
                lang = line.strip()[3:]
                html_lines.append(f'<pre><code class="{lang}">')
                in_code = True
            continue
        if in_code:
            html_lines.append(line.replace("<", "&lt;").replace(">", "&gt;"))
            continue

        # 空行
        if not line.strip():
            if in_table:
                html_lines.append("</table>")
                in_table = False
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            html_lines.append("")
            continue

        # 表格分隔行
        if re.match(r"^\s*\|[\s\-:|]+\|\s*$", line):
            continue

        # 表格行
        if line.strip().startswith("|") and line.strip().endswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if not in_table:
                html_lines.append('<table class="report-table">')
                tag = "th"
                in_table = True
            else:
                tag = "td"
            row = "".join(f"<{tag}>{_inline(c)}</{tag}>" for c in cells)
            html_lines.append(f"<tr>{row}</tr>")
            continue

        if in_table:
            html_lines.append("</table>")
            in_table = False

        # 标题
        m = re.match(r"^(#{1,6})\s+(.+)$", line)
        if m:
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            level = len(m.group(1))
            html_lines.append(f"<h{level}>{_inline(m.group(2))}</h{level}>")
            continue

        # 水平线
        if re.match(r"^---+\s*$", line):
            html_lines.append("<hr>")
            continue

        # 图片
        m = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", line.strip())
        if m:
            alt, src = m.group(1), m.group(2)
            b64 = _embed_image(src)
            if b64:
                html_lines.append(
                    f'<figure><img src="{b64}" alt="{alt}" style="max-width:100%">'
                    f'<figcaption>{alt}</figcaption></figure>'
                )
            else:
                html_lines.append(f'<p><em>[图片未找到: {src}]</em></p>')
            continue

        # 列表项
        m = re.match(r"^(\s*)[-*]\s+(.+)$", line)
        if m:
            if not in_list:
                html_lines.append("<ul>")
                in_list = True
            html_lines.append(f"<li>{_inline(m.group(2))}</li>")
            continue

        # 引用
        if line.strip().startswith(">"):
            text = line.strip()[1:].strip()
            html_lines.append(f"<blockquote>{_inline(text)}</blockquote>")
            continue

        # 普通段落
        if in_list:
            html_lines.append("</ul>")
            in_list = False
        html_lines.append(f"<p>{_inline(line)}</p>")

    if in_table:
        html_lines.append("</table>")
    if in_list:
        html_lines.append("</ul>")

    return "\n".join(html_lines)


def _inline(text: str) -> str:
    """处理行内格式：粗体/代码/链接。"""
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"`(.+?)`", r"<code>\1</code>", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', text)
    # checkbox
    text = text.replace("- [ ]", "☐").replace("- [x]", "☑")
    return text


def _embed_image(src: str) -> str | None:
    """将图片文件转为 base64 data URI。"""
    img_path = IMG_DIR / Path(src).name
    if not img_path.exists():
        # 尝试从 src 相对路径找
        img_path = _ROOT / "docs" / src
    if not img_path.exists():
        print(f"  [WARN] Image not found: {src}")
        return None
    data = img_path.read_bytes()
    b64 = base64.b64encode(data).decode()
    ext = img_path.suffix.lower()
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "gif": "image/gif"}.get(ext[1:], "image/png")
    return f"data:{mime};base64,{b64}"


# ── 主流程 ────────────────────────────────────────────────────────────
print("Building HTML report...")

md_text = MD_PATH.read_text(encoding="utf-8")
body = md_to_html(md_text)

html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>井中探测数据处理与反演子系统 — 综合汇报</title>
<style>
  body {{
    font-family: "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", sans-serif;
    max-width: 1100px;
    margin: 0 auto;
    padding: 20px 40px;
    line-height: 1.7;
    color: #1a1a2e;
    background: #fafbfc;
  }}
  h1 {{ color: #0d47a1; border-bottom: 3px solid #0d47a1; padding-bottom: 10px; }}
  h2 {{ color: #1565c0; border-bottom: 1px solid #ddd; padding-bottom: 6px; margin-top: 40px; }}
  h3 {{ color: #1976d2; margin-top: 25px; }}
  .report-table {{
    border-collapse: collapse;
    width: 100%;
    margin: 12px 0;
    font-size: 14px;
  }}
  .report-table th {{
    background: #e3f2fd;
    border: 1px solid #bbdefb;
    padding: 8px 12px;
    text-align: left;
    font-weight: bold;
  }}
  .report-table td {{
    border: 1px solid #e0e0e0;
    padding: 6px 12px;
  }}
  .report-table tr:nth-child(even) {{ background: #f5f5f5; }}
  code {{
    background: #e8eaf6;
    padding: 2px 6px;
    border-radius: 3px;
    font-family: Consolas, "Courier New", monospace;
    font-size: 13px;
  }}
  pre {{
    background: #263238;
    color: #eeffff;
    padding: 16px;
    border-radius: 6px;
    overflow-x: auto;
    font-size: 13px;
    line-height: 1.5;
  }}
  pre code {{ background: none; color: inherit; padding: 0; }}
  figure {{
    text-align: center;
    margin: 20px 0;
    padding: 10px;
    background: white;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
  }}
  figure img {{ border-radius: 4px; }}
  figcaption {{
    color: #666;
    font-size: 13px;
    margin-top: 8px;
    font-style: italic;
  }}
  blockquote {{
    border-left: 4px solid #1976d2;
    margin: 12px 0;
    padding: 8px 16px;
    background: #e3f2fd;
    color: #0d47a1;
  }}
  hr {{ border: none; border-top: 2px solid #e0e0e0; margin: 30px 0; }}
  strong {{ color: #b71c1c; }}
  ul {{ padding-left: 24px; }}
  li {{ margin: 4px 0; }}
  @media print {{
    body {{ max-width: 100%; padding: 10px; }}
    figure {{ page-break-inside: avoid; }}
    h2 {{ page-break-before: always; }}
    h1 + * {{ page-break-before: avoid; }}
  }}
</style>
</head>
<body>
{body}
<hr>
<p style="text-align:center; color:#999; font-size:12px;">
  Generated on 2026-03-26 | 井中探测数据处理与反演子系统 v0.3
</p>
</body>
</html>"""

OUT_PATH.write_text(html, encoding="utf-8")
size_kb = OUT_PATH.stat().st_size / 1024
print(f"Done: {OUT_PATH} ({size_kb:.0f} KB)")
print(f"双击即可在浏览器中打开。")

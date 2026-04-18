#!/usr/bin/env python3
"""
build_index.py — 掃描 output/ 產生 index.json（文章索引清單）。
GitHub Pages 無法列目錄，所以需要這份索引讓前端知道有哪些文章。
每次新增文章後執行一次：python build_index.py
"""

import json
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "output"
INDEX_PATH = Path(__file__).parent / "index.json"


def build_index():
    index = {}
    for date_dir in sorted(OUTPUT_DIR.iterdir()):
        if not date_dir.is_dir():
            continue
        date_str = date_dir.name
        slugs = []
        for f in sorted(date_dir.glob("*.json")):
            raw = None
            for enc in ["utf-8", "utf-8-sig"]:
                try:
                    raw = json.loads(f.read_text(encoding=enc))
                    break
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
            if raw is None:
                continue
            # 支援 {title, title_zh, data} 物件格式
            if not isinstance(raw, dict) or "data" not in raw:
                continue
            title_en = raw["title"]
            title_zh = raw["title_zh"]
            data = raw["data"]
            slugs.append({
                "slug": f.stem,
                "title": title_en,
                "title_zh": title_zh,
            })
        if slugs:
            index[date_str] = slugs
    return index


if __name__ == "__main__":
    idx = build_index()
    INDEX_PATH.write_text(json.dumps(idx, ensure_ascii=False, indent=2), encoding="utf-8")
    total = sum(len(v) for v in idx.values())
    print(f"✅ index.json — {len(idx)} 個日期、{total} 篇文章")

import json
json_path = r'D:\article-daily-paper\output\20260416\20260416-Anthropogenic Regional Adaptation in Multimodal Vision-Language Model.json'
md_path = r'D:\article-daily-paper\output\20260416\20260416-Anthropogenic Regional Adaptation in Multimodal Vision-Language Model.md'
with open(json_path, encoding='utf-8') as f:
    data = json.load(f)
lines = []
for para in data:
    for sentence in para:
        lines.append(sentence)
        lines.append('')
    lines.append('----')
with open(md_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print('完成')

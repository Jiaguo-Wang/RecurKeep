"""Verify the identities of release files, standalone archives, and original copied files without starting the runtime."""
import hashlib
import argparse
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archives-only',action='store_true',help='仅校验两个原始独立归档，不要求解压数据或生成结果')
    args=parser.parse_args()
    manifest=json.loads((ROOT/'RELEASE_MANIFEST.json').read_text())
    items=[item for item in manifest['files'] if not args.archives_only or item['path'].startswith('archives/')]
    for item in items:
        path=ROOT/item['path']
        if not path.is_file():
            parser.error('缺少校验文件：'+item['path']+'；归档准备说明见 artifacts/README.md')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:
            raise ValueError('发布文件发生变化：'+item['path'])
    print(json.dumps({'status':'PASS','verified_files':len(items)},ensure_ascii=False))


if __name__=='__main__':main()

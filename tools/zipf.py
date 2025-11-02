import os
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED



def list_all_files(root: Path, exts=None):
    files = []
    for p in root.rglob("*"):
        if p.is_file():
            if exts is None or p.suffix.lower() in exts:
                files.append(p)
    return sorted(files)

def split_evenly(items, k):
    n = len(items)
    base, rem = divmod(n, k)
    result = []
    start = 0
    for i in range(k):
        size = base + (1 if i < rem else 0)
        result.append(items[start:start+size])
        start += size
    return result

def make_zip(zip_path: Path, files, base_dir: Path):
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(zip_path, "w", ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(f, arcname=f.relative_to(base_dir))

def split_and_zip(folder_path, prefix, k=20, skip_empty=True, exts=None):
    root = Path(folder_path).expanduser().resolve()
    files = list_all_files(root, exts=exts)
    print(f"[INFO] 根目录: {root}")
    print(f"[INFO] 文件总数: {len(files)}；目标份数: {k}")

    chunks = split_evenly(files, k)

    made = 0
    for i, chunk in enumerate(chunks, 1):
        if skip_empty and len(chunk) == 0:
            continue
        zip_name = f"{prefix}_{(made+1) if skip_empty else i}.zip"
        make_zip(Path(zip_name), chunk, base_dir=root)
        print(f"已生成：{zip_name}（{len(chunk)} 个文件）")
        made += 1

    if made == 0:
        print("[WARN] 没有可打包的文件。")
    else:
        print(f"[DONE] 共生成 {made} 个压缩包。")

if __name__ == "__main__":
    # ====== 可改动区 ======
    FOLDER = "/home/pxl416/PeixiLiu/px_proj/px_data/csl-daily-frames-512x512/sentence"  # 根目录
    PREFIX = "CSL_Daily"  # 压缩包前缀
    K = 20  # 份数
    SKIP_EMPTY = True  # 是否跳过空压缩包（True 推荐）
    EXTS = None  # 只打某些后缀，如 {".mp4", ".jpg"}；None 表示全部文件
    # ====== 可改动区 ======

    split_and_zip(FOLDER, PREFIX, k=K, skip_empty=SKIP_EMPTY, exts=EXTS)

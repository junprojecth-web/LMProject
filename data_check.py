from pathlib import Path
from collections import Counter
import pandas as pd

# ============================================================
# 설정
# ============================================================

DATASET_DIRS = {
    "PlantVillage": Path("./PlantVillage"),
    "PlantDoc": Path("./PlantDoc-Dataset"),
    "PlantSeg": Path("./PlantSeg"),
    "Kaggle_1": Path(r"C:\Users\human-12\.cache\kagglehub\datasets\salmasyed1360"),
    "Kaggle_2": Path(r"C:\Users\human-12\.cache\kagglehub\datasets\weitianqi"),
}

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png",
    ".bmp", ".webp", ".JPG", ".JPEG", ".PNG"
}


# ============================================================
# 이미지 파일 찾기
# ============================================================

def find_images(folder):

    images = []

    if not folder.exists():
        print(f"[폴더 없음] {folder}")
        return images

    for path in folder.rglob("*"):

        if path.is_file():
            if path.suffix.lower() in {
                ".jpg", ".jpeg", ".png",
                ".bmp", ".webp"
            }:
                images.append(path)

    return images


# ============================================================
# 데이터셋별 진단
# ============================================================

all_results = []

for dataset_name, dataset_path in DATASET_DIRS.items():

    print("\n" + "=" * 70)
    print(dataset_name)
    print(dataset_path)
    print("=" * 70)

    images = find_images(dataset_path)

    print(f"발견된 이미지 : {len(images):,}장")

    if len(images) == 0:
        continue

    # --------------------------------------------------------
    # 상위 폴더 구조 확인
    # --------------------------------------------------------

    folder_counter = Counter()

    for img in images:

        try:
            relative = img.relative_to(dataset_path)

            parts = relative.parts

            if len(parts) >= 2:
                folder_counter[parts[0]] += 1
            else:
                folder_counter["ROOT"] += 1

        except Exception:
            pass

    print("\n[상위 폴더별 이미지 수]")

    for folder, count in folder_counter.most_common(30):
        print(f"{folder:<50} {count:>6,}")

    # --------------------------------------------------------
    # 실제 이미지 경로 저장
    # --------------------------------------------------------

    for img in images:

        relative_path = img.relative_to(dataset_path)

        all_results.append({
            "dataset": dataset_name,
            "image_path": str(img),
            "relative_path": str(relative_path),
            "filename": img.name,
            "parent_folder": img.parent.name
        })


# ============================================================
# 결과 CSV
# ============================================================

df = pd.DataFrame(all_results)

df.to_csv(
    "all_image_inventory.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 70)
print("전체 이미지 진단 완료")
print("=" * 70)

print(f"전체 이미지 : {len(df):,}장")

if len(df) > 0:

    print("\n데이터셋별 이미지 수")
    print(
        df.groupby("dataset")
        .size()
        .sort_values(ascending=False)
    )

    print("\n결과 파일")
    print("all_image_inventory.csv")

else:

    print("""
이미지가 한 장도 발견되지 않았습니다.

이 경우 다음을 확인해야 합니다.

1. 현재 Python 실행 위치
2. PlantVillage 폴더 위치
3. PlantDoc-Dataset 폴더 위치
4. PlantSeg 폴더 위치
5. Kaggle 경로
6. 이미지가 실제로 존재하는지
""")
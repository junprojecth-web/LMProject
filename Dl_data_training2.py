
import os
import re
import shutil
import pandas as pd
from pathlib import Path
import shutil
from pathlib import Path

# ============================================================
# 1. 기본 설정
# ============================================================

BASE_DIR = Path(r"C:\Users\human-12\Desktop\MLProject")

# 5개 데이터셋 위치
DATASET_DIRS = {
    "PlantVillage": BASE_DIR / "PlantVillage-Dataset" / "raw",
    "PlantDoc": BASE_DIR / "PlantDoc-Dataset",
   #"PlantSeg": BASE_DIR / "PlantSeg",
    
    # Kaggle 데이터셋
    "Kaggle1": Path(
    r"C:\Users\human-12\.cache\kagglehub\datasets\salmasyed1360\plant-diseases-100k-labelled-images"
),

    "Kaggle2": Path(
        r"C:\Users\human-12\.cache\kagglehub\datasets\weitianqi\plantseg\versions\1\plantsegv2\images\all"
    ),
}

# 결과 폴더
OUTPUT_DIR = BASE_DIR / "DL_processed_dataset"

# 실제 이미지 저장 폴더
IMAGE_DIR = OUTPUT_DIR / "images"

# 결과 CSV
CSV_FILE = OUTPUT_DIR / "dl_image_metadata.csv"

# 지원하는 이미지 확장자
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".JPG",
    ".JPEG",
    ".PNG",
}


# ============================================================
# 2. 상태 분류 키워드
# ============================================================

# ------------------------------------------------------------
# 정상
# ------------------------------------------------------------

NORMAL_KEYWORDS = [
    "healthy",
    "health",
    "normal",
    "healthy leaf",
    "healthy leaves",
    "fresh",
    "good",
    "normal leaf",
    "normal leaves",
    "healthy plant",
]


# ------------------------------------------------------------
# 병충해
# ------------------------------------------------------------

DISEASE_KEYWORDS = [
    # disease
    "disease",
    "diseased",
    "infection",
    "infected",
    "infect",

    # fungus
    "fungus",
    "fungal",
    "mildew",
    "powdery mildew",
    "downy mildew",
    "rust",
    "blight",
    "anthracnose",
    "botrytis",
    "scab",
    "canker",
    "rot",
    "black rot",
    "brown rot",
    "gray mold",
    "grey mold",

    # bacteria
    "bacterial",
    "bacteria",
    "bacterial spot",
    "bacterial wilt",

    # virus
    "virus",
    "viral",
    "mosaic",
    "yellow mosaic",
    "leaf curl",
    "virus disease",

    # pest
    "pest",
    "pests",
    "insect",
    "insects",
    "aphid",
    "aphids",
    "mite",
    "mites",
    "thrips",
    "whitefly",
    "white fly",
    "mealybug",
    "scale insect",
    "caterpillar",
    "worm",
    "borer",
    "leaf miner",
    "leafminer",

    # common disease names
    "early blight",
    "late blight",
    "septoria",
    "downy",
    "alternaria",
    "fusarium",
    "verticillium",
    "phytophthora",
    "fire blight",
    "apple scab",
    "cedar apple rust",
]


# ------------------------------------------------------------
# 잎 이상 / 환경 스트레스
# ------------------------------------------------------------

LEAF_ABNORMAL_KEYWORDS = [
    # discoloration
    "yellow",
    "yellowing",
    "chlorosis",
    "chlorotic",
    "discoloration",
    "discoloured",
    "discolored",

    # dry / burn
    "dry",
    "dried",
    "dryness",
    "wilt",
    "wilting",
    "wilted",
    "burn",
    "burned",
    "burnt",
    "leaf burn",
    "sunburn",
    "scorch",
    "leaf scorch",

    # damage
    "damage",
    "damaged",
    "injury",
    "injured",
    "necrosis",
    "necrotic",
    "dead leaf",
    "dead leaves",

    # shape
    "curl",
    "curled",
    "leaf curl",
    "rolling",
    "rolled",
    "deformation",
    "deformed",
    "distortion",

    # spots
    "spot",
    "spots",
    "leaf spot",
    "leaf spots",

    # stress
    "stress",
    "stressed",
    "heat stress",
    "cold stress",
    "drought stress",
    "water stress",
    "temperature stress",

    # nutrient
    "nutrient deficiency",
    "deficiency",
    "nitrogen deficiency",
    "iron deficiency",
]


# ============================================================
# 3. 텍스트 정규화
# ============================================================

def normalize_text(text):
    """
    폴더명/파일명을 비교하기 쉽게 정규화합니다.
    """

    text = str(text).lower()

    # URL encoding 일부 제거
    text = text.replace("%20", " ")

    # 특수문자를 공백으로 변경
    text = re.sub(r"[_\-]+", " ", text)

    # 여러 공백 제거
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# 4. 상태 자동 분류
# ============================================================

def classify_state(folder_path, file_name):
    """
    폴더명 + 파일명을 보고
    정상 / 병충해 / 잎 이상 / 기타
    중 하나로 자동 분류합니다.
    """

    folder_text = normalize_text(folder_path)
    file_text = normalize_text(file_name)

    # 폴더명을 더 중요하게 사용
    combined_text = folder_text + " " + file_text

    # --------------------------------------------------------
    # 1순위 : 병충해
    # --------------------------------------------------------

    for keyword in DISEASE_KEYWORDS:

        if keyword in combined_text:
            return "병충해", f"disease_keyword:{keyword}"

    # --------------------------------------------------------
    # 2순위 : 정상
    # --------------------------------------------------------

    for keyword in NORMAL_KEYWORDS:

        if keyword in combined_text:
            return "정상", f"normal_keyword:{keyword}"

    # --------------------------------------------------------
    # 3순위 : 잎 이상 / 스트레스
    # --------------------------------------------------------

    for keyword in LEAF_ABNORMAL_KEYWORDS:

        if keyword in combined_text:
            return "잎 이상", f"abnormal_keyword:{keyword}"

    # --------------------------------------------------------
    # 4순위 : 기타
    # --------------------------------------------------------

    return "기타", "no_matching_keyword"


# ============================================================
# 5. 이미지 파일 재귀 탐색
# ============================================================

def scan_dataset(dataset_name, dataset_path):

    print()
    print("=" * 70)
    print(f"[{dataset_name}] 탐색 시작")
    print(f"경로 : {dataset_path}")
    print("=" * 70)

    records = []

    if not dataset_path.exists():

        print("경고: 폴더가 존재하지 않습니다.")
        return records

    image_count = 0

    # 모든 하위 폴더 재귀 탐색
    for file_path in dataset_path.rglob("*"):

        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in {
            ext.lower() for ext in IMAGE_EXTENSIONS
        }:
            continue

        image_count += 1

        # 데이터셋 내부에서의 상대 경로
        relative_path = file_path.relative_to(dataset_path)

        # 부모 폴더
        parent_folder = str(relative_path.parent)

        # 실제 가장 가까운 폴더명
        folder_name = file_path.parent.name

        # 상태 분류
        state, reason = classify_state(
            parent_folder,
            file_path.name
        )

        records.append({
            "dataset": dataset_name,
            "original_path": str(file_path),
            "relative_path": str(relative_path),
            "folder_name": folder_name,
            "file_name": file_path.name,
            "state": state,
            "classification_reason": reason,
        })

    print(f"이미지 발견 : {image_count:,}장")

    return records


# ============================================================
# 6. 데이터셋 전체 스캔
# ============================================================

all_records = []

for dataset_name, dataset_path in DATASET_DIRS.items():

    records = scan_dataset(
        dataset_name,
        dataset_path
    )

    all_records.extend(records)


# ============================================================
# 7. DataFrame 생성
# ============================================================

df = pd.DataFrame(all_records)

if df.empty:

    print()
    print("이미지를 하나도 찾지 못했습니다.")
    print("DATASET_DIRS의 폴더 경로를 확인해주세요.")
    exit()


# ============================================================
# 8. 결과 폴더 생성
# ============================================================

IMAGE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 9. 통계 출력
# ============================================================

print()
print("=" * 70)
print("전체 이미지 탐색 결과")
print("=" * 70)

print(
    df.groupby("dataset")
    .size()
    .sort_values(ascending=False)
)


print()
print("=" * 70)
print("상태별 이미지 수")
print("=" * 70)

print(
    df["state"]
    .value_counts()
)


print()
print("=" * 70)
print("데이터셋 × 상태")
print("=" * 70)

print(
    pd.crosstab(
        df["dataset"],
        df["state"]
    )
)


# ============================================================
# 10. 자동 분류 결과 저장
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    CSV_FILE,
    index=False,
    encoding="utf-8-sig"
)


print()
print("=" * 70)
print("1차 자동 분류 완료")
print("=" * 70)

print(f"CSV 파일 : {CSV_FILE}")


# ============================================================
# 11. 폴더별 분류 현황
# ============================================================

folder_summary = (
    df.groupby(
        [
            "dataset",
            "folder_name",
            "state",
            "classification_reason"
        ]
    )
    .size()
    .reset_index(name="image_count")
    .sort_values(
        ["dataset", "image_count"],
        ascending=[True, False]
    )
)

folder_summary_file = (
    OUTPUT_DIR / "folder_classification_summary.csv"
)

folder_summary.to_csv(
    folder_summary_file,
    index=False,
    encoding="utf-8-sig"
)


print(
    f"폴더별 분류 결과 : {folder_summary_file}"
)


# ============================================================
# 12. 애매한 데이터 확인
# ============================================================

unknown_df = df[
    df["state"] == "기타"
].copy()

unknown_file = (
    OUTPUT_DIR / "manual_review_required.csv"
)

unknown_df.to_csv(
    unknown_file,
    index=False,
    encoding="utf-8-sig"
)


print()
print("=" * 70)
print("수동 확인 필요 데이터")
print("=" * 70)

print(
    f"기타 이미지 : {len(unknown_df):,}장"
)

print(
    f"확인 파일 : {unknown_file}"
)


# ============================================================
# 13. 실제 학습용 폴더 생성
# ============================================================

for state in [
    "정상",
    "병충해",
    "잎 이상",
    "기타"
]:

    (IMAGE_DIR / state).mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# 14. 이미지 복사
# ============================================================
#
# 주의:
# 처음에는 전체 이미지를 복사하지 않고
# metadata CSV만 확인하는 것을 권장합니다.
#
# 아래 COPY_IMAGES = False 로 시작하세요.
#
# 결과를 확인한 뒤 True로 변경하면
# 실제 학습 폴더를 생성합니다.
# ============================================================

COPY_IMAGES = False


if COPY_IMAGES:

    print()
    print("=" * 70)
    print("학습용 이미지 복사 시작")
    print("=" * 70)

    for index, row in df.iterrows():

        source = Path(row["original_path"])

        state = row["state"]

        if not source.exists():
            continue

        # 파일명이 중복될 수 있으므로
        # 데이터셋 이름과 index를 붙입니다.
        destination_name = (
            f"{row['dataset']}_{index:07d}"
            f"{source.suffix.lower()}"
        )

        destination = (
            IMAGE_DIR
            / state
            / destination_name
        )

        try:

            shutil.copy2(
                source,
                destination
            )

        except Exception as e:

            print(
                f"복사 실패: {source}"
            )
            print(e)

    print()
    print("이미지 복사 완료")


# ============================================================
# 15. 최종 결과
# ============================================================

print()
print("=" * 70)
print("DL 데이터 전처리 진단 완료")
print("=" * 70)

print()
print("생성 파일")
print(f"1. {CSV_FILE}")
print(f"2. {folder_summary_file}")
print(f"3. {unknown_file}")

print()
print("중요:")
print("먼저 CSV 결과를 확인한 후 COPY_IMAGES = True로 변경하세요.")
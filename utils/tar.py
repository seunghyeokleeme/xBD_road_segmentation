import os
import tarfile

def extract_tar(tar_path, extract_to_path):
    """tar 압축풀기"""
    with tarfile.open(tar_path, 'r') as tar:
        tar.extractall(extract_to_path)
    print(f"{extract_to_path} 경로에 압축을 해제합니다.")

def compress_to_tar(output_filename, source_path):
    # output_filename: 생성될 .tar 파일명 (예: "archive.tar")
    # source_path: 압축할 파일 또는 디렉토리 경로
    with tarfile.open(output_filename, "w") as tar:
        # source_path의 basename을 사용하면 압축 파일 내에서 해당 파일/디렉토리명이 그대로 사용됩니다.
        tar.add(source_path, arcname=os.path.basename(source_path))
    print(f"{output_filename} 파일이 생성되었습니다.")

# 사용 예제
# if __name__ == "__main__":
#     train_source = "./train"  # 압축할 파일 또는 디렉토리 경로를 입력하세요.
#     hold_source = "./hold"  # 압축할 파일 또는 디렉토리 경로를 입력하세요.
#     test_source = "./test"  # 압축할 파일 또는 디렉토리 경로를 입력하세요.
#     train_output = "road_train_images_targets.tar"                   # 생성될 tar 파일명
#     hold_output = "road_hold_images_targets.tar"                   # 생성될 tar 파일명
#     test_output = "road_test_images_targets.tar"                   # 생성될 tar 파일명
#     compress_to_tar(train_output, train_source)
#     compress_to_tar(hold_output, hold_source)
#     compress_to_tar(test_output, test_source)

# extract_tar('./road_train_images_targets.tar', './')
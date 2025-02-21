import os
import shutil

# 디렉토리 생성
def CreateDirectory(filePath):
    try:
        if not os.path.exists(filePath):
            os.makedirs(filePath)
            print(f"[LOG] Success to create the directory on {filePath}.")
    except OSError:
        print("[LOG] Error: Failed to create the directory.")
        

# 디렉토리내 폴더를 제외한 모든파일 삭제  
def DeleteAllFiles(filePath):
    if os.path.exists(filePath):
        for file in os.scandir(filePath):
            try:
                os.remove(file.path.replace('\\', '/'))
            # 폴더는 지울 수 없음    
            except:
                pass
        print(f"[LOG] Remove All File in {filePath}")
    else:
        print("[LOG] Directory Not Found")



#특정 확장자 파일제거
def DeleteExtensionFiles(filePath, exts = ['pkl', 'csv']):
    for path in filePath:
        if any(ext in path for ext in exts):
            os.remove(path)
        print(f"[LOG] Remove {exts} All File")
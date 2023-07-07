import os
import glob

# 현재 디렉토리의 모든 .cpp 파일을 검색합니다.
for filename in glob.glob('./cublas/*.cpp'):
    with open(filename, 'r') as file:
        lines = file.readlines()

    fix = 0
    # 각 줄을 검사합니다.
    for i in range(len(lines)):
        # 'Value ret;'이 포함된 줄을 찾았다면, 아래 3번째 줄을 'if (!ret)'으로 변경합니다.
        if 'Value ret;' in lines[i]:
            lines[i] = "    Value ret = nullptr;\n"
            if i + 3 < len(lines):
                if (lines[i+3] != "    else\n"):
                    print(filename)
                    print(lines[i+3])
                else:
                    fix = 1
                    lines[i + 3] = '    if (!ret)\n'

    # 변경된 내용을 다시 파일에 씁니다.
    if (fix == 1):
        with open(filename, 'w') as file:
            file.writelines(lines)
    fix = 0


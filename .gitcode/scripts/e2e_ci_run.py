"""
e2e_ci_run.py — CI 流水线端到端回归编排入口

在流水线（或本机）按顺序执行三段回归，任一段失败即整体失败：
  1. e2e_driver.py  ：注册登录 → 上传三类数据集 → 六类模型训练到终态
  2. e2e_onetest.py ：每类模型拉取本地模板 → /api/run_test 一键测试
  3. e2e_export.py  ：标准导出包下载校验 → 删除含旁车清理

用法：
    python .gitcode/scripts/e2e_ci_run.py
环境要求：项目依赖已安装（torch 等）；脚本会自行启动/复用 5000 端口服务。
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(os.path.dirname(HERE))  # .gitcode/scripts 的上两级 = 项目根
STEPS = ['e2e_driver.py', 'e2e_onetest.py', 'e2e_export.py']


def main():
    failures = []
    for step in STEPS:
        print(f'\n===== [CI] {step} =====', flush=True)
        proc = subprocess.run([sys.executable, os.path.join(HERE, step)],
                              cwd=PROJECT)
        if proc.returncode != 0:
            failures.append(step)
            print(f'\n[CI] {step} FAILED (exit={proc.returncode})', flush=True)
            break
        print(f'[CI] {step} PASSED', flush=True)

    if failures:
        print(f'\n[CI] E2E REGRESSION FAILED: {failures}')
        return 1
    print('\n[CI] E2E REGRESSION ALL PASSED')
    return 0


if __name__ == '__main__':
    sys.exit(main())

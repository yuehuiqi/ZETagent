"""
本地调试启动脚本（debug_runner.py）

使用方式：
    cd D:\ZET\2agent\codePro
    python debug_runner.py

注意：提交时不要包含此文件，agent.py 中不包含任何硬编码 Key。
"""

import os
import sys
import logging
import warnings

# ============================================================
#   1. 设置 API Key（调试用）
#   ⚠️ 若 Key 失效请到火山引擎控制台重新获取：
#      https://console.volcengine.com/ark → API Key 管理
# ============================================================
os.environ["VLM_API_KEY"] = "ark-22e3203f-6be0-498b-85f2-99f48bfb36ee-50063"

# ============================================================
#   2. 屏蔽 agent_base.py 产生的"调试模式"警告噪声
# ============================================================
warnings.filterwarnings("ignore", category=UserWarning, message=".*调试模式.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*EVAL_MODE.*")


# ============================================================
#   3. 配置整洁日志（抑制 httpx/openai/agent_base 的冗余输出）
# ============================================================
class _CleanFilter(logging.Filter):
    """过滤 agent_base/test_runner 产生的无关日志行"""
    _NOISY = [
        "调试模式", "API配置", "提示：提交时", "API_URL", "MODEL_ID",
        "主办方统一", "仅供自测", "Token usage:", "[API调用]",
        "=====", "-----", "EVAL_MODE", "选手", "提交阶段",
    ]

    def filter(self, record: logging.LogRecord) -> bool:
        # 只过滤 WARNING 级别的噪声（保留 ERROR 和重要 INFO）
        if record.levelno == logging.WARNING:
            msg = record.getMessage()
            if any(kw in msg for kw in self._NOISY):
                return False
        # DEBUG 日志只保留我们自己模块的
        if record.levelno == logging.DEBUG:
            if record.name not in ("agent", "utils.output_parser", "utils.prompt_engine"):
                return False
        return True


class _ColorFormatter(logging.Formatter):
    """带颜色和对齐的日志格式（Windows CMD 用 colorama）"""
    GREY   = "\033[90m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    CYAN   = "\033[96m"
    RESET  = "\033[0m"
    BOLD   = "\033[1m"

    LEVEL_COLORS = {
        logging.DEBUG:    GREY,
        logging.INFO:     "",       # 默认颜色
        logging.WARNING:  YELLOW,
        logging.ERROR:    RED,
        logging.CRITICAL: RED + BOLD,
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self.LEVEL_COLORS.get(record.levelno, "")
        time = self.formatTime(record, "%H:%M:%S")
        name = record.name.split(".")[-1][:12].ljust(12)
        msg  = record.getMessage()
        level = record.levelname[0]   # I / W / E / D
        return f"{self.GREY}{time}{self.RESET} {color}[{level}]{self.RESET} {self.GREY}{name}{self.RESET} {msg}"


def _setup_logging():
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # 清空已有 handler
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_ColorFormatter())
    handler.addFilter(_CleanFilter())
    root.addHandler(handler)

    # 抑制第三方库冗余日志
    for lib in ("httpx", "httpcore", "openai._base_client", "openai.http_client"):
        logging.getLogger(lib).setLevel(logging.ERROR)


_setup_logging()


# ============================================================
#   4. 切换到 codePro 目录后直接运行 test_runner（同进程）
# ============================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.insert(0, script_dir)

import runpy


print(f"\n{'='*60}")
print(f"  GUI Agent 本地测试 | Key: ...{os.environ['VLM_API_KEY'][-8:]}")
print(f"{'='*60}\n")

try:
    runpy.run_path("test_runner.py", run_name="__main__")
except SystemExit as e:
    sys.exit(e.code)

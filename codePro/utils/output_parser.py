"""
鲁棒输出解析模块 v4.0（Robust Output Parser）

v4.0 核心改进：
1. "动作行"锚定解析（v3继承）
2. 末级兜底：Desperate CLICK —— 从全文提取任意坐标对（修复 CLICK→SCROLL 误触发）
3. OPEN app名截断修复：正则改为贪婪+最小匹配，支持含"TV"等后缀
4. 兜底顺序：精确解析 → 全文 → Desperate CLICK → SCROLL
"""

import re
import json
import logging
from typing import Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)


# ============================================================
#   提取"动作行"
# ============================================================

def _extract_action_line(raw: str) -> str:
    """提取"动作：..."行内容"""
    patterns = [
        r"动作[：:]\s*(.+?)(?:\n|$)",
        r"Action[：:]\s*(.+?)(?:\n|$)",
        r"action[：:]\s*(.+?)(?:\n|$)",
    ]
    for pat in patterns:
        m = re.search(pat, raw, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return raw   # 找不到动作行，返回全文


# ============================================================
#   动作行级解析器（高置信度）
# ============================================================

def _parse_click(line: str) -> Optional[Tuple[str, Dict]]:
    patterns = [
        r"CLICK:\[\[(\d+)[,\s]+(\d+)\]\]",
        r"CLICK\s*:\s*\[\[\s*(\d+)\s*[,，]\s*(\d+)\s*\]\]",
        r"click\s*\(\s*point\s*=\s*['\"]<point>\s*(\d+)\s+(\d+)\s*</point>['\"]",
    ]
    for pat in patterns:
        m = re.search(pat, line, re.IGNORECASE)
        if m:
            x, y = int(m.group(1)), int(m.group(2))
            if 0 <= x <= 1000 and 0 <= y <= 1000:
                return "CLICK", {"point": [x, y]}
    return None


def _parse_type(line: str) -> Optional[Tuple[str, Dict]]:
    patterns = [
        r"TYPE:\['(.+?)'\]",
        r'TYPE:\["(.+?)"\]',
        r"TYPE:\[(.+?)\]",
        r"type\s*\(\s*(?:content|text)\s*=\s*['\"](.+?)['\"]",
    ]
    for pat in patterns:
        m = re.search(pat, line, re.IGNORECASE)
        if m:
            text = m.group(1).strip()
            if text:
                return "TYPE", {"text": text}
    return None


def _parse_scroll(line: str) -> Optional[Tuple[str, Dict]]:
    patterns = [
        r"SCROLL:\[\[(\d+)[,\s]+(\d+)\]\s*[,，]\s*\[(\d+)[,\s]+(\d+)\]\]",
        r"SCROLL\s*:?\s*\[?\[(\d+)[,\s]+(\d+)\]?\s*[,，]\s*\[?(\d+)[,\s]+(\d+)\]?",
        r"scroll.*?<point>\s*(\d+)\s+(\d+)\s*</point>.*?<point>\s*(\d+)\s+(\d+)\s*</point>",
    ]
    for pat in patterns:
        m = re.search(pat, line, re.IGNORECASE)
        if m:
            x1, y1, x2, y2 = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
            if all(0 <= v <= 1000 for v in [x1, y1, x2, y2]):
                return "SCROLL", {"start_point": [x1, y1], "end_point": [x2, y2]}
    return None


def _parse_open(line: str) -> Optional[Tuple[str, Dict]]:
    """OPEN 解析：支持含 TV/tv 等后缀的长名称"""
    # 通用词黑名单
    blacklist = {"应用", "app", "APP", "应用程序", "软件", "程序", "该应用",
                 "这个应用", "目标应用", "Ӧ", "\x00"}
                 
    # 规范化：修复模型输出被截断的情况
    NORMALIZE = {
        "去哪": "去哪儿",
        "去哪旅": "去哪儿",
        "去哪旅行": "去哪儿",
        "腾讯": "腾讯视频",
    }
                 
    patterns = [
        r"OPEN:\['(.+?)'\]",
        r'OPEN:\["(.+?)"\]',
        r"open\s*\(\s*app_name\s*=\s*['\"](.+?)['\"]",
        r"OPEN:\[(.+?)\]",
    ]
    for pat in patterns:
        m = re.search(pat, line, re.IGNORECASE)
        if m:
            raw_name = m.group(1).strip().strip("'\"[] ")
            if (raw_name
                    and 1 <= len(raw_name) <= 20
                    and raw_name not in blacklist
                    and not any(c in raw_name for c in ["\x00", "\xff", "\xfe"])):
                # 规范化处理（修复截断的app名）
                normalized = NORMALIZE.get(raw_name, raw_name)
                return "OPEN", {"app_name": normalized}
    return None


def _parse_complete(line: str) -> Optional[Tuple[str, Dict]]:
    """严格 COMPLETE 匹配（仅匹配动作行格式）"""
    if re.search(r"COMPLETE:\s*\[\s*\]", line, re.IGNORECASE):
        return "COMPLETE", {}
    if re.search(r"complete\s*\(\s*\)", line, re.IGNORECASE):
        return "COMPLETE", {}
    return None


# ============================================================
#   末级兜底：Desperate CLICK（从全文提取任意坐标对）
# ============================================================

def _desperate_click(raw: str) -> Optional[Tuple[str, Dict]]:
    """
    最后手段：从输出中提取任意看起来像坐标的数字对

    用于处理"点击[500, 300]"、"坐标是500,300"等非标准格式

    只有当模型确实没有标准动作输出时才使用（防止误触发）
    """
    # 模式1: [xxx, yyy] 格式（两位或三位数，不接受单个数字如[5,3]）
    bracket_pat = r"\[(\d{2,3})\s*[,，]\s*(\d{2,3})\]"
    # 模式2: (xxx, yyy) 格式
    paren_pat = r"\((\d{2,3})\s*[,，]\s*(\d{2,3})\)"
    # 模式3: 明确说"坐标 xxx yyy"
    coord_pat = r"(?:坐标|coordinate|position|点击)\D{0,5}(\d{2,3})\D{1,5}(\d{2,3})"

    for pat in [bracket_pat, paren_pat, coord_pat]:
        matches = re.findall(pat, raw)
        for m in matches:
            try:
                x, y = int(m[0]), int(m[1])
                if 10 <= x <= 990 and 10 <= y <= 990:  # 更严格范围，排除边界噪声
                    logger.warning(f"[Parser] Desperate CLICK: ({x}, {y})")
                    return "CLICK", {"point": [x, y]}
            except (ValueError, IndexError):
                continue
    return None


# ============================================================
#   主解析函数 v4.0
# ============================================================

def parse_output(raw_output: str) -> Tuple[str, Dict[str, Any]]:
    """
    多策略分级解析 v4.0

    解析优先级：
    1. 从"动作行"提取 → 严格解析
    2. 全文扫描 COMPLETE:[] 标记
    3. 全文宽松解析（CLICK/TYPE/SCROLL/OPEN）
    4. Desperate CLICK（从任意坐标对提取）
    5. 中文语义 COMPLETE（非常低优先级）
    6. 安全兜底 → SCROLL 向下
    """
    if not raw_output or not raw_output.strip():
        logger.warning("[Parser] Empty output → SCROLL fallback")
        return "SCROLL", {"start_point": [500, 700], "end_point": [500, 300]}

    raw = raw_output.strip()

    # ---- 第1级：动作行精确解析 ----
    action_line = _extract_action_line(raw)
    line_upper = action_line.upper()

    action_parsers_line = []
    if "COMPLETE" in line_upper:
        action_parsers_line.append(_parse_complete)
    if "CLICK" in line_upper:
        action_parsers_line.append(_parse_click)
    if "TYPE" in line_upper:
        action_parsers_line.append(_parse_type)
    if "SCROLL" in line_upper:
        action_parsers_line.append(_parse_scroll)
    if "OPEN" in line_upper:
        action_parsers_line.append(_parse_open)

    for parser in action_parsers_line:
        result = parser(action_line)
        if result:
            logger.info(f"[Parser] ActionLine → {result[0]}: {result[1]}")
            return result

    # ---- 第2级：全文 COMPLETE:[] 精确标记 ----
    if re.search(r"COMPLETE:\[\]", raw, re.IGNORECASE):
        logger.info("[Parser] Global COMPLETE:[] found")
        return "COMPLETE", {}

    # ---- 第3级：全文宽松解析 ----
    raw_upper = raw.upper()
    global_candidates = []
    if "CLICK" in raw_upper:
        global_candidates.append(_parse_click)
    if "TYPE" in raw_upper:
        global_candidates.append(_parse_type)
    if "SCROLL" in raw_upper:
        global_candidates.append(_parse_scroll)
    if "OPEN" in raw_upper:
        global_candidates.append(_parse_open)

    for parser in global_candidates:
        result = parser(raw)
        if result:
            logger.warning(f"[Parser] GlobalFallback → {result[0]}: {result[1]}")
            return result

    # ---- 第4级：Desperate CLICK（从任意坐标对提取）----
    result = _desperate_click(raw)
    if result:
        return result

    # ---- 第5级：中文语义 COMPLETE（严格限定场景）----
    if re.search(r"任务(?:已经?)?(?:全部)?完成|所有步骤.*完成|操作(?:全部)?完成", raw):
        logger.warning("[Parser] CN semantic COMPLETE (low confidence)")
        return "COMPLETE", {}

    # ---- 第6级：安全兜底 → SCROLL ----
    logger.error(f"[Parser] All failed → SCROLL. Raw[:100]: {raw[:100]}")
    return "SCROLL", {"start_point": [500, 700], "end_point": [500, 300]}


# ============================================================
#   校验 & 辅助
# ============================================================

def validate_output(action: str, parameters: Dict[str, Any]) -> Tuple[bool, str]:
    """校验动作格式"""
    if action not in {"CLICK", "TYPE", "SCROLL", "OPEN", "COMPLETE"}:
        return False, f"Invalid action: {action}"

    if action == "CLICK":
        pt = parameters.get("point")
        if not isinstance(pt, list) or len(pt) != 2:
            return False, f"CLICK needs point=[x,y], got {pt}"
        if not all(isinstance(v, (int, float)) and 0 <= v <= 1000 for v in pt):
            return False, f"CLICK point out of range: {pt}"

    elif action == "TYPE":
        if not parameters.get("text"):
            return False, "TYPE needs non-empty text"

    elif action == "SCROLL":
        for key in ("start_point", "end_point"):
            pt = parameters.get(key)
            if not isinstance(pt, list) or len(pt) != 2:
                return False, f"SCROLL needs {key}=[x,y], got {pt}"

    elif action == "OPEN":
        if not parameters.get("app_name"):
            return False, "OPEN needs non-empty app_name"

    return True, ""


def build_retry_hint(action: str, parameters: Dict, error_msg: str) -> str:
    """构建重试错误提示"""
    return (
        f"⚠️ 上次输出格式错误（{error_msg}），请严格按格式输出：\n"
        "  动作：CLICK:[[x, y]]\n"
        "  动作：TYPE:['内容']\n"
        "  动作：SCROLL:[[x1,y1],[x2,y2]]\n"
        "  动作：OPEN:['应用名']\n"
        "  动作：COMPLETE:[]\n"
        "「动作：」必须单独一行，格式完全一致。"
    )

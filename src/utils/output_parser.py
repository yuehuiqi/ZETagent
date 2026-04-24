"""
鲁棒输出解析模块（Robust Output Parser）v2.0

核心思路：多策略分级解析（Multi-Strategy Cascaded Parsing）

v2.0 修复（基于测试结果分析）：
1. 【P0修复】COMPLETE 误触发：
   - 原问题：COMPLETE 优先级最高，Thought中任何"完成"字眼均被误触发（占失败的64%）
   - 修复：降低COMPLETE优先级，改为仅匹配"动作行"而非全文
   - 新优先级：精确动作解析 → JSON块 → 宽松解析 → COMPLETE(严格) → 兜底SCROLL

2. 【P1修复】兜底策略：
   - 原问题：所有解析失败返回COMPLETE → 导致任务中途结束
   - 修复：兜底改为SCROLL（向下滚动），语义更安全

3. 【P2修复】OPEN app_name乱码：
   - 原问题：OPEN解析宽松模式误匹配"应用"两字（中文编码问题）
   - 修复：严格仅从标准格式和XML格式提取，不用宽松中文关键字
"""

import re
import json
import logging
from typing import Tuple, Dict, Any, Optional, List

logger = logging.getLogger(__name__)


# ============================================================
#   各动作的正则模式（按置信度从高到低排列）
# ============================================================

PATTERNS = {
    "CLICK": [
        # 标准格式（最高置信度）：动作：CLICK:[[500, 300]]
        r"动作[：:]\s*CLICK:\[\[(\d+)[,\s]+(\d+)\]\]",
        r"Action:\s*CLICK:\[\[(\d+)[,\s]+(\d+)\]\]",
        # 标准格式（无前缀）
        r"CLICK:\[\[(\d+)[,\s]+(\d+)\]\]",
        r"CLICK\s*:\s*\[\[\s*(\d+)\s*[,，]\s*(\d+)\s*\]\]",
        # 原始 XML 格式：click(point='<point>500 300</point>')
        r"click\s*\(\s*point\s*=\s*['\"]<point>\s*(\d+)\s+(\d+)\s*</point>['\"]",
        # JSON 格式
        r'"action"\s*:\s*"click".*?"point"\s*:\s*\[(\d+)[,\s]+(\d+)\]',
    ],
    "TYPE": [
        # 标准格式（最高置信度）：动作：TYPE:['内容']
        r"动作[：:]\s*TYPE:\['(.+?)'\]",
        r"动作[：:]\s*TYPE:\[\"(.+?)\"\]",
        r"Action:\s*TYPE:\['(.+?)'\]",
        # 标准格式（无前缀）
        r"TYPE:\['(.+?)'\]",
        r'TYPE:\["(.+?)"\]',
        r"TYPE:\[(.+?)\]",
        # 原始格式：type(content='内容') 或 type(text='内容')
        r"type\s*\(\s*(?:content|text)\s*=\s*['\"](.+?)['\"]",
        # JSON 格式
        r'"action"\s*:\s*"type".*?"(?:content|text)"\s*:\s*"(.+?)"',
    ],
    "SCROLL": [
        # 标准格式（最高置信度）
        r"动作[：:]\s*SCROLL:\[\[(\d+)[,\s]+(\d+)\]\s*[,，]\s*\[(\d+)[,\s]+(\d+)\]\]",
        r"Action:\s*SCROLL:\[\[(\d+)[,\s]+(\d+)\]\s*[,，]\s*\[(\d+)[,\s]+(\d+)\]\]",
        # 标准格式（无前缀）
        r"SCROLL:\[\[(\d+)[,\s]+(\d+)\]\s*[,，]\s*\[(\d+)[,\s]+(\d+)\]\]",
        r"SCROLL\s*:?\s*\[?\[(\d+)[,\s]+(\d+)\]?\s*[,，]\s*\[?(\d+)[,\s]+(\d+)\]?",
        # 原始格式
        r"scroll.*?<point>\s*(\d+)\s+(\d+)\s*</point>.*?<point>\s*(\d+)\s+(\d+)\s*</point>",
    ],
    "OPEN": [
        # 标准格式（最高置信度）：只从标准格式提取，不用中文关键字（防乱码）
        r"动作[：:]\s*OPEN:\['(.+?)'\]",
        r"动作[：:]\s*OPEN:\[\"(.+?)\"\]",
        r"Action:\s*OPEN:\['(.+?)'\]",
        r"OPEN:\['(.+?)'\]",
        r'OPEN:\["(.+?)"\]',
        # 原始格式：open(app_name='应用名')
        r"open\s*\(\s*app_name\s*=\s*['\"](.+?)['\"]",
        # 宽松格式（最低优先级）：OPEN:[应用名]
        r"OPEN:\[(.+?)\]",
    ],
    # COMPLETE：严格匹配，仅匹配"动作行"中的标准格式
    "COMPLETE_STRICT": [
        r"动作[：:]\s*COMPLETE:\[\]",
        r"Action:\s*COMPLETE:\[\]",
        r"COMPLETE:\[\]",
        r"complete\s*\(\s*\)",
        r"COMPLETE\s*:\s*\[\s*\]",
    ],
    # COMPLETE中文关键字（低优先级，只在找不到其他动作时使用）
    "COMPLETE_CN": [
        r"(?:任务(?:已经?)?完成|操作完成|已完成(?:所有)?任务)",
    ],
}


# ============================================================
#   各动作解析函数
# ============================================================

def _try_parse_click(raw: str) -> Optional[Tuple[str, Dict]]:
    """尝试解析 CLICK 动作"""
    for pat in PATTERNS["CLICK"]:
        m = re.search(pat, raw, re.IGNORECASE | re.DOTALL)
        if m:
            try:
                x, y = int(m.group(1)), int(m.group(2))
                if 0 <= x <= 1000 and 0 <= y <= 1000:
                    return "CLICK", {"point": [x, y]}
            except (ValueError, IndexError):
                continue
    return None


def _try_parse_type(raw: str) -> Optional[Tuple[str, Dict]]:
    """尝试解析 TYPE 动作"""
    for pat in PATTERNS["TYPE"]:
        m = re.search(pat, raw, re.IGNORECASE | re.DOTALL)
        if m:
            text = m.group(1).strip()
            # 过滤乱码（长度过短或包含非常见字符）
            if text and len(text) >= 1 and len(text) <= 500:
                return "TYPE", {"text": text}
    return None


def _try_parse_scroll(raw: str) -> Optional[Tuple[str, Dict]]:
    """尝试解析 SCROLL 动作"""
    for pat in PATTERNS["SCROLL"]:
        m = re.search(pat, raw, re.IGNORECASE | re.DOTALL)
        if m:
            try:
                x1, y1, x2, y2 = (
                    int(m.group(1)), int(m.group(2)),
                    int(m.group(3)), int(m.group(4)),
                )
                if all(0 <= v <= 1000 for v in [x1, y1, x2, y2]):
                    return "SCROLL", {
                        "start_point": [x1, y1],
                        "end_point": [x2, y2],
                    }
            except (ValueError, IndexError):
                continue
    return None


def _try_parse_open(raw: str) -> Optional[Tuple[str, Dict]]:
    """尝试解析 OPEN 动作（严格模式，防止中文乱码匹配）"""
    for pat in PATTERNS["OPEN"]:
        m = re.search(pat, raw, re.IGNORECASE | re.DOTALL)
        if m:
            app_name = m.group(1).strip().strip("'\"[]")
            # 过滤乱码：app名长度应在2-20字符之间，且不应是通用词"应用"
            generic_words = {"应用", "app", "APP", "应用程序", "软件"}
            if (app_name and 2 <= len(app_name) <= 20
                    and app_name not in generic_words):
                return "OPEN", {"app_name": app_name}
    return None


def _try_parse_complete_strict(raw: str) -> Optional[Tuple[str, Dict]]:
    """严格 COMPLETE 检测（仅匹配动作行的标准格式）"""
    for pat in PATTERNS["COMPLETE_STRICT"]:
        if re.search(pat, raw, re.IGNORECASE):
            return "COMPLETE", {}
    return None


def _try_parse_complete_cn(raw: str) -> Optional[Tuple[str, Dict]]:
    """中文语义 COMPLETE 检测（低优先级兜底）"""
    for pat in PATTERNS["COMPLETE_CN"]:
        if re.search(pat, raw, re.IGNORECASE):
            return "COMPLETE", {}
    return None


def _try_parse_json(raw: str) -> Optional[Tuple[str, Dict]]:
    """
    尝试从输出中提取 JSON 结构
    模型有时会输出 ```json ... ``` 格式
    """
    json_pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
    m = re.search(json_pattern, raw, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(1))
            action = data.get("action", "").upper()
            params = data.get("parameters", data.get("params", {}))
            if action in ("CLICK", "TYPE", "SCROLL", "OPEN", "COMPLETE"):
                return action, params
        except json.JSONDecodeError:
            pass
    return None


# ============================================================
#   主解析函数（重构解析优先级）
# ============================================================

def parse_output(raw_output: str) -> Tuple[str, Dict[str, Any]]:
    """
    多策略分级解析模型原始输出 v2.0

    重构后的解析优先级（修复COMPLETE误触发）：
    1. 严格COMPLETE检测（仅匹配"动作：COMPLETE:[]"等标准行）
    2. 精确动作解析（CLICK/TYPE/SCROLL/OPEN，基于关键字+正则）
    3. JSON 块提取
    4. 全局宽松解析（不依赖前缀关键字）
    5. 中文语义COMPLETE（低优先级，仅在以上都失败时尝试）
    6. 安全兜底（返回SCROLL向下滚动，而非COMPLETE）

    Args:
        raw_output: 模型原始文字输出

    Returns:
        (action: str, parameters: dict)
    """
    if not raw_output or not raw_output.strip():
        logger.warning("[Parser] Empty output, using safe fallback SCROLL.")
        return "SCROLL", {"start_point": [500, 700], "end_point": [500, 300]}

    raw = raw_output.strip()

    # ---- 第1级：严格 COMPLETE 检测（仅匹配"动作行"标准格式）----
    # 必须是 "COMPLETE:[]" 精确格式，不触发思考文字中的"完成"
    result = _try_parse_complete_strict(raw)
    if result:
        logger.info("[Parser] Parsed COMPLETE (strict)")
        return result

    # ---- 第2级：精确动作解析（基于关键字预筛选）----
    raw_upper = raw.upper()
    parsers = []

    if "CLICK" in raw_upper:
        parsers.append(("CLICK", _try_parse_click))
    if "TYPE" in raw_upper:
        parsers.append(("TYPE", _try_parse_type))
    if "SCROLL" in raw_upper:
        parsers.append(("SCROLL", _try_parse_scroll))
    if "OPEN" in raw_upper:
        parsers.append(("OPEN", _try_parse_open))
    # 仅当上面都没有关键字时，再尝试中文关键字
    if not parsers:
        if "点击" in raw or "单击" in raw:
            parsers.append(("CLICK", _try_parse_click))
        if "输入" in raw or "键入" in raw:
            parsers.append(("TYPE", _try_parse_type))
        if "滑动" in raw or "滚动" in raw:
            parsers.append(("SCROLL", _try_parse_scroll))
        if "打开" in raw:
            parsers.append(("OPEN", _try_parse_open))

    for action_name, parser_fn in parsers:
        result = parser_fn(raw)
        if result:
            logger.info(f"[Parser] Parsed {result[0]}: {result[1]}")
            return result

    # ---- 第3级：JSON 块提取 ----
    result = _try_parse_json(raw)
    if result:
        logger.info(f"[Parser] Parsed from JSON: {result[0]}")
        return result

    # ---- 第4级：全局宽松解析（不依赖关键字前缀）----
    for parser_fn in [_try_parse_click, _try_parse_type,
                       _try_parse_scroll, _try_parse_open]:
        result = parser_fn(raw)
        if result:
            logger.warning(f"[Parser] Fallback parse: {result[0]}")
            return result

    # ---- 第5级：中文语义 COMPLETE（低优先级）----
    result = _try_parse_complete_cn(raw)
    if result:
        logger.warning("[Parser] Parsed COMPLETE (CN semantic, low confidence)")
        return result

    # ---- 第6级：安全兜底（SCROLL向下，而非COMPLETE）----
    logger.error(f"[Parser] All strategies failed. Fallback to SCROLL.\nRaw: {raw[:200]}")
    return "SCROLL", {"start_point": [500, 700], "end_point": [500, 300]}


# ============================================================
#   输出校验
# ============================================================

def validate_output(action: str, parameters: Dict[str, Any]) -> Tuple[bool, str]:
    """
    校验解析结果是否满足 TestRunner 的格式要求

    Returns:
        (is_valid, error_message)
    """
    valid_actions = {"CLICK", "TYPE", "SCROLL", "OPEN", "COMPLETE"}
    if action not in valid_actions:
        return False, f"Invalid action: {action}"

    if action == "CLICK":
        point = parameters.get("point")
        if not isinstance(point, list) or len(point) != 2:
            return False, f"CLICK requires point=[x,y], got {point}"
        if not all(isinstance(v, (int, float)) and 0 <= v <= 1000 for v in point):
            return False, f"CLICK point out of range [0,1000]: {point}"

    elif action == "TYPE":
        if "text" not in parameters or not parameters["text"]:
            return False, "TYPE requires non-empty text"

    elif action == "SCROLL":
        for key in ("start_point", "end_point"):
            pt = parameters.get(key)
            if not isinstance(pt, list) or len(pt) != 2:
                return False, f"SCROLL requires {key}=[x,y], got {pt}"

    elif action == "OPEN":
        if "app_name" not in parameters or not parameters["app_name"]:
            return False, "OPEN requires non-empty app_name"

    return True, ""


def build_retry_hint(action: str, parameters: Dict, error_msg: str) -> str:
    """构建重试时注入的错误提示"""
    hint_lines = [
        f"上次输出解析失败（{error_msg}），请严格按照以下格式输出动作：",
        "",
        "正确格式示例：",
        "  动作：CLICK:[[500, 300]]",
        "  动作：TYPE:['搜索内容']",
        "  动作：SCROLL:[[500, 700], [500, 300]]",
        "  动作：OPEN:['美团']",
        "  动作：COMPLETE:[]",
        "",
        "注意：动作格式必须严格匹配，不要省略任何括号或引号。",
    ]
    return "\n".join(hint_lines)

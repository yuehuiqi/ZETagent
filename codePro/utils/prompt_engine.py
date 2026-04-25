"""
提示词工程模块 v4.0（Prompt Engineering）

v4.0 核心改进：
1. detect_app_from_instruction()：从指令文本解析精确 App 名（覆盖乱码问题）
2. step1 OPEN 强制提示：当 instruction_app 已知时，第一步明确指出需要 OPEN
3. CLICK 坐标经验规则细化（修复 y 偏高问题）
4. 精简 Few-Shot，保证 Token 效率
"""

import re
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


# ============================================================
#   从指令中提取 App 名（关键新增功能）
# ============================================================

# 已知 App 名称列表（中文全名 + 常见别名）
# 注意：别名必须把较长的字符串排在前面，防止前缀截断
KNOWN_APPS = {
    "美团": ["美团外卖", "美团", "外卖"],
    "百度地图": ["百度地图", "百度导航"],
    "哔哩哔哩": ["哔哩哔哩", "B站", "bilibili", "哔哩"],
    "抖音": ["抖音", "douyin"],
    "快手": ["快手", "kuaishou"],
    "爱奇艺": ["爱奇艺", "iqiyi", "奇艺"],
    "芒果TV": ["芒果TV", "芒果tv", "芒果", "MangoTV"],
    "腾讯视频": ["腾讯视频", "腾讯"],
    "喜马拉雅": ["喜马拉雅", "喜马"],
    "去哪儿": ["去哪儿旅行", "去哪儿", "qunar"],
}


def detect_app_from_instruction(instruction: str) -> Optional[str]:
    """
    从用户指令文本中提取目标 App 名称

    优先级：精确匹配已知App名 > 正则提取

    Args:
        instruction: 用户任务指令字符串

    Returns:
        App名称字符串，如 "美团"、"百度地图"，未识别则返回 None
    """
    if not instruction:
        return None

    # 优先精确匹配已知 App
    for canonical, aliases in KNOWN_APPS.items():
        for alias in aliases:
            if alias in instruction:
                logger.info(f"[AppDetect] Matched '{alias}' → '{canonical}'")
                return canonical

    # 正则提取：通常格式是"去XXX"、"打开XXX"、"在XXX上"
    patterns = [
        r"去([^\s，。、！？去打开找]{2,6}(?:TV|tv|Map)?)[上里中\s，。]",
        r"打开([^\s，。、！？]{2,8}(?:TV|tv)?)",
        r"在([^\s，。、！？]{2,8}(?:TV|tv)?)(?:上|里|中|App|应用)",
    ]
    for pat in patterns:
        m = re.search(pat, instruction)
        if m:
            name = m.group(1).strip()
            logger.info(f"[AppDetect] Regex extracted: '{name}'")
            return name

    return None


# ============================================================
#   系统提示词
# ============================================================

SYSTEM_PROMPT = """你是一个专业的安卓手机 GUI 操作 Agent。根据用户指令和当前截图，输出下一步的精确操作。

## 坐标系统
- 所有坐标归一化到 [0, 1000]，x=左→右，y=上→下
- 坐标必须落在目标元素的**视觉中心**

## 动作格式（严格匹配）
```
动作：CLICK:[[x, y]]              ← 点击坐标
动作：TYPE:['内容']               ← 在输入框输入文字
动作：SCROLL:[[x1,y1],[x2,y2]]    ← 从起点滑动到终点
动作：OPEN:['应用名']             ← 打开指定App
动作：COMPLETE:[]                 ← 任务全部完成
```

## 必须遵守的输出格式
```
思考：[分析当前界面，确定下一步操作目标]
动作：[上面某种格式]
```

## 坐标参考规则
- **顶部状态栏/搜索框**：y 在 [60, 140] 之间，不要点到 y<60 的区域
- **顶部右侧图标**（搜索、设置、更多）：x=[880,970], y=[70,120]
- **顶部左侧图标**（返回、菜单）：x=[30,120], y=[70,120]
- **搜索结果列表第1项**：y 约在 [140, 210] 之间（通常第一条结果在此区域，不要偏高）
- **搜索结果列表第2项**：y 约在 [240, 310] 之间
- **底部标签栏**：y 在 [880, 980] 之间
- **底部结算/确认按钮**：y 在 [880, 960] 之间

## 关键规则
1. 元素点击必须精确，y坐标不能偏高（不要点状态栏），要点元素**中心**
2. OPEN 的 App 名必须与手机桌面图标一致（中文精确名称，不加"App"后缀）
3. 搜索框点击后，下一步直接 TYPE 输入，不需要再次点击
4. COMPLETE 只在界面显示"成功/已完成/订单提交"等明确完成状态时输出
5. 找不到目标 → 先 SCROLL:[[500,700],[500,300]] 向下查找
6. TYPE 直接输入内容，不加任何前缀

"""


# ============================================================
#   User Prompt 构建
# ============================================================

# 每个条目: {"scenario": ..., "screenshot_desc": ..., "thought": ..., "action": ...}
APP_FEW_SHOT_DB: Dict[str, List[Dict]] = {

    "美团": [
        {
            "scenario": "美团首页，需要点击外卖入口",
            "screenshot_desc": "屏幕显示美团APP首页，顶部有搜索栏，下方有'外卖'、'美食'等图标网格",
            "thought": "当前在美团首页，需要进入外卖功能。外卖图标通常在首页功能网格的左上角位置，约在 x=[50,200], y=[155,235]",
            "action": "CLICK:[[100, 195]]",
        },
        {
            "scenario": "美团外卖搜索页，搜索框为空",
            "screenshot_desc": "顶部有一个放大镜图标和搜索输入框，框内为空或显示提示文字",
            "thought": "需要点击搜索框激活输入，搜索框在顶部 y≈[95,130] 区域，横向占满屏幕宽度",
            "action": "CLICK:[[500, 112]]",
        },
        {
            "scenario": "美团搜索结果列表，需要选择目标商家",
            "screenshot_desc": "列表显示多家商家，需要找到目标店铺并点击",
            "thought": "搜索结果以卡片列表展示，每张卡片高约80单位。需要识别目标店铺名称所在的卡片并点击其中心区域",
            "action": "CLICK:[[500, 192]]",
        },
        {
            "scenario": "商品详情页，需要点击'加入购物车'按钮",
            "screenshot_desc": "页面右侧有橙色圆形加号按钮（+），通常在 x=[900,980] 区域",
            "thought": "需要将商品加入购物车，加号按钮位于商品行右侧，颜色为美团橙色",
            "action": "CLICK:[[941, 680]]",
        },
        {
            "scenario": "购物车结算页，需要点击'去结算'",
            "screenshot_desc": "底部有橙色'去结算'按钮，显示商品总价",
            "thought": "任务要求提交订单，底部结算按钮在 y=[880,940] 区域右侧",
            "action": "CLICK:[[833, 910]]",
        },
    ],

    "百度地图": [
        {
            "scenario": "百度地图首页，需要搜索目的地",
            "screenshot_desc": "顶部有搜索栏，显示'搜索地点、公交、地铁'等提示文字",
            "thought": "需要点击搜索框输入目的地，搜索框在顶部 y=[55,100] 区域",
            "action": "CLICK:[[400, 78]]",
        },
        {
            "scenario": "百度地图搜索结果页，需要选择正确的POI",
            "screenshot_desc": "显示搜索结果列表，每项包含名称、地址、距离",
            "thought": "搜索结果以列表展示，需要找到与任务匹配的地点并点击。通常第一条结果在 y=[150,230] 区域",
            "action": "CLICK:[[500, 190]]",
        },
        {
            "scenario": "百度地图POI详情页，需要点击'导航'",
            "screenshot_desc": "底部有'路线'、'导航'、'到这去'等按钮",
            "thought": "需要开始导航，'导航'按钮通常在底部操作栏的中间位置",
            "action": "CLICK:[[500, 920]]",
        },
    ],

    "哔哩哔哩": [
        {
            "scenario": "B站首页，需要搜索视频",
            "screenshot_desc": "顶部有放大镜搜索图标，右侧有摄像机图标",
            "thought": "点击顶部搜索图标进入搜索页",
            "action": "CLICK:[[920, 55]]",
        },
        {
            "scenario": "B站搜索结果页，需要选择目标视频",
            "screenshot_desc": "显示视频卡片列表，包含封面、标题、UP主、播放量",
            "thought": "在搜索结果中找到目标视频，视频卡片第一条在 y=[150,320] 区域",
            "action": "CLICK:[[500, 235]]",
        },
    ],

    "抖音": [
        {
            "scenario": "抖音首页，需要进入搜索",
            "screenshot_desc": "顶部右上角有放大镜图标，页面为视频流",
            "thought": "需要点击右上角搜索图标",
            "action": "CLICK:[[940, 55]]",
        },
        {
            "scenario": "抖音搜索框激活状态，需要输入关键词",
            "screenshot_desc": "顶部搜索框处于激活（光标闪烁）状态",
            "thought": "搜索框已激活，直接输入关键词",
            "action": "TYPE:['关键词']",
        },
    ],

    "快手": [
        {
            "scenario": "快手首页，需要进入搜索",
            "screenshot_desc": "顶部有快手logo和搜索图标",
            "thought": "点击顶部搜索图标",
            "action": "CLICK:[[940, 60]]",
        },
    ],

    "爱奇艺": [
        {
            "scenario": "爱奇艺首页，顶部有搜索框",
            "screenshot_desc": "顶部显示搜索框，内有提示文字",
            "thought": "点击搜索框进入搜索",
            "action": "CLICK:[[450, 60]]",
        },
    ],

    "芒果TV": [
        {
            "scenario": "芒果TV首页，需要搜索内容",
            "screenshot_desc": "顶部右侧有搜索图标",
            "thought": "点击搜索图标进入搜索页",
            "action": "CLICK:[[930, 55]]",
        },
    ],

    "腾讯视频": [
        {
            "scenario": "腾讯视频首页，顶部有搜索框",
            "screenshot_desc": "顶部搜索框显示'搜索影视、综艺'等提示",
            "thought": "点击搜索框",
            "action": "CLICK:[[420, 55]]",
        },
    ],

    "喜马拉雅": [
        {
            "scenario": "喜马拉雅首页，需要搜索内容",
            "screenshot_desc": "顶部有放大镜图标和搜索框",
            "thought": "点击搜索框",
            "action": "CLICK:[[400, 60]]",
        },
    ],

    # 通用兜底示例（任何 App 都适用）
    "通用": [
        {
            "scenario": "界面有明显的搜索框（顶部）",
            "screenshot_desc": "顶部有搜索框或搜索图标",
            "thought": "搜索框通常在顶部 y=[40,150] 区域，点击激活",
            "action": "CLICK:[[500, 90]]",
        },
        {
            "scenario": "需要向下滚动查看更多内容",
            "screenshot_desc": "当前页面内容不完整，需要向下滚动",
            "thought": "向下滑动查看更多，从屏幕中下部向上滑",
            "action": "SCROLL:[[500, 700], [500, 300]]",
        },
        {
            "scenario": "任务已完成",
            "screenshot_desc": "界面显示操作成功、订单提交、任务完成等状态",
            "thought": "任务目标已达成，输出完成信号",
            "action": "COMPLETE:[]",
        },
    ],
}


def _detect_app(instruction: str, history_actions: List[Dict]) -> str:
    """
    根据指令和历史动作推断当前操作的 App 类型

    优先级：
    1. 历史中是否有 OPEN 动作（最可靠）
    2. 指令中的关键词匹配
    3. 返回 '通用'
    """
    # 从历史 OPEN 动作推断
    for act in history_actions:
        if act.get("action") == "OPEN":
            app_name = act.get("parameters", {}).get("app_name", "")
            if app_name:
                for key in APP_FEW_SHOT_DB:
                    if key in app_name or app_name in key:
                        return key

    # 从指令关键词推断
    app_keywords = {
        "美团": ["美团", "外卖", "骑手"],
        "百度地图": ["百度地图", "导航", "地图", "路线"],
        "哔哩哔哩": ["B站", "哔哩", "bilibili"],
        "抖音": ["抖音", "douyin"],
        "快手": ["快手", "kuaishou"],
        "爱奇艺": ["爱奇艺", "iqiyi"],
        "芒果TV": ["芒果", "mango"],
        "腾讯视频": ["腾讯视频", "腾讯", "WeTV"],
        "喜马拉雅": ["喜马拉雅", "喜马"],
    }
    for app, keywords in app_keywords.items():
        for kw in keywords:
            if kw in instruction:
                return app

    return "通用"


def build_few_shot_block(app_name: str, max_examples: int = 2) -> str:
    """
    构建适合当前 App 的 Few-Shot 示例文本块

    Args:
        app_name: 推断出的 App 名称
        max_examples: 最多注入的示例数量

    Returns:
        格式化的示例文本（插入到 user prompt 中）
    """
    examples = APP_FEW_SHOT_DB.get(app_name, []) + APP_FEW_SHOT_DB.get("通用", [])
    examples = examples[:max_examples]

    if not examples:
        return ""

    lines = ["## 操作示例参考（来自同类 App）"]
    for i, ex in enumerate(examples, 1):
        lines.append(f"\n**示例 {i}**：{ex['scenario']}")
        lines.append(f"界面描述：{ex['screenshot_desc']}")
        lines.append(f"思考：{ex['thought']}")
        lines.append(f"动作：{ex['action']}")

    return "\n".join(lines)


def build_user_prompt(
    instruction: str,
    step_count: int,
    task_plan: Optional[str],
    history_summary: str,
    recent_actions_text: str,
    app_name: str,
    instruction_app: Optional[str] = None,
    retry_hint: str = "",
) -> str:
    """
    构建动态 User Prompt

    instruction_app: 从指令中提取的精确 App 名，用于 step1 OPEN 提示
    """
    parts = []

    parts.append(f"## 用户任务\n{instruction}\n")

    # ★ 关键：step 1 时，若知道 App 名，强制提示第一步要 OPEN
    if step_count == 1 and instruction_app:
        parts.append(
            f"## ⚠️ 第一步必须执行\n"
            f"当前在手机桌面（或应用未打开）。任务需要的App是：**{instruction_app}**\n"
            f"第一步动作必须是：\n"
            f"```\n"
            f"动作：OPEN:['{instruction_app}']\n"
            f"```\n"
        )
    elif step_count == 1:
        parts.append(
            "## 提示\n"
            "这是任务第一步。如果当前界面是手机桌面，请先用 OPEN:['应用名'] 打开目标应用。\n"
        )

    if task_plan:
        parts.append(f"## 任务规划（参考）\n{task_plan}\n")

    if history_summary:
        parts.append(f"## 历史摘要\n{history_summary}\n")

    if recent_actions_text:
        parts.append(f"## 最近操作\n{recent_actions_text}\n")

    parts.append(f"## 当前步骤\n第 {step_count} 步\n")

    # 注入 Few-Shot (前 4 步注入)
    if step_count <= 4:
        few_shot = build_few_shot_block(app_name, max_examples=3)
        if few_shot:
            parts.append(few_shot + "\n")

    if retry_hint:
        parts.append(f"## ⚠️ 上次格式错误，请修正\n{retry_hint}\n")

    parts.append(
        "## 输出要求\n"
        "请严格按照以下格式输出（「动作：」必须单独一行，完全匹配格式）：\n"
        "```\n"
        "思考：[分析界面，确定目标元素位置]\n"
        "动作：ACTION:[[params]]\n"
        "```"
    )

    return "\n".join(parts)


# ============================================================
#   Messages 构建
# ============================================================

def build_messages(
    system_prompt: str,
    user_text: str,
    image_b64_url: str,
    history_image_messages: List[Dict],
) -> List[Dict]:
    """构建完整 OpenAI messages 列表"""
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history_image_messages)
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": user_text},
            {"type": "image_url", "image_url": {"url": image_b64_url}},
        ],
    })
    return messages


# ============================================================
#   App 类型检测（向后兼容）
# ============================================================

def _detect_app(instruction: str, history_actions: List[Dict]) -> str:
    """从指令/历史动作推断 App 类型（兼容旧代码）"""
    # 先从历史 OPEN 推断
    for act in history_actions:
        if act.get("action") == "OPEN":
            app = act.get("parameters", {}).get("app_name", "")
            if app:
                return app

    result = detect_app_from_instruction(instruction)
    return result or "通用"

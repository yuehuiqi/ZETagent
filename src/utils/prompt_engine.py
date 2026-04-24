"""
提示词工程模块（Prompt Engineering）

核心设计思路：
1. System Prompt 固定化，充分利用 KV Cache 降低 Token 开销
2. 动态 User Prompt 携带任务进度、历史摘要、少样本示例
3. 针对不同 App 类型注入专属 Few-Shot 示例，提升领域适配性
4. Chain-of-Thought 结构引导模型输出高质量推理过程
"""

import logging
import re
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# ============================================================
#   系统级提示词（固定，可被 KV Cache）
# ============================================================

SYSTEM_PROMPT = """你是一个专业的安卓手机 GUI 操作 Agent，负责根据用户指令和当前截图，输出精确的下一步操作。

## 坐标系统
- 所有坐标使用归一化坐标，范围 [0, 1000]
- x=0 为屏幕最左，x=1000 为屏幕最右
- y=0 为屏幕顶部，y=1000 为屏幕底部
- 坐标必须落在目标 UI 元素的【内部中心】，不要点到边缘

## 可用动作
| 动作 | 输出格式 | 说明 |
|------|----------|------|
| CLICK | `CLICK:[[x, y]]` | 单点点击，x/y 为归一化坐标 |
| TYPE | `TYPE:['文本内容']` | 在当前激活的输入框中键入文字 |
| SCROLL | `SCROLL:[[x1, y1], [x2, y2]]` | 从起点滑动到终点 |
| OPEN | `OPEN:['精确应用名']` | 打开指定 App，名称必须与手机上显示的一致 |
| COMPLETE | `COMPLETE:[]` | 任务已全部完成时输出 |

## ⚠️ 严格输出格式（必须遵守）
每次回复必须包含两部分：
```
思考：[用中文分析当前界面状态，判断已完成的步骤，确定下一步目标]
动作：ACTION:[[params]]
```

**格式错误示例（禁止）**：
- ❌ 动作：click [[500, 300]]  （必须大写）
- ❌ 动作：CLICK(500, 300)     （必须双层方括号格式）
- ❌ 在"思考"中写"任务完成"   （这不会触发COMPLETE，必须在"动作"行写）

**格式正确示例**：
- ✅ 动作：CLICK:[[500, 300]]
- ✅ 动作：TYPE:['搜索内容']
- ✅ 动作：OPEN:['美团']
- ✅ 动作：COMPLETE:[]

## 关键操作规则
1. CLICK 坐标要精确落在目标元素中心（按钮、图标、文字区域内部）
2. 顶部搜索框/导航栏通常在 y=[40, 150]；底部TabBar在 y=[900, 1000]
3. 右上角图标（搜索、更多）通常在 x=[880, 980], y=[40, 100]
4. COMPLETE 仅在界面明确显示"操作成功/订单提交成功/完成"时输出
5. 若找不到目标元素，先 SCROLL:[[500, 700], [500, 300]] 向下滚动查找
6. OPEN 的应用名必须是完整精确的中文名（如"美团"、"百度地图"），不要加"App"后缀
"""


# ============================================================
#   App 专属 Few-Shot 示例库
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
    retry_hint: str = "",
) -> str:
    """
    构建动态 User Prompt

    Args:
        instruction: 原始用户指令
        step_count: 当前步骤编号
        task_plan: 首步生成的任务规划文本（可为 None）
        history_summary: 早期历史的文字摘要
        recent_actions_text: 最近几步的动作记录
        app_name: 当前 App 名称
        retry_hint: 重试时附加的错误提示

    Returns:
        user prompt 字符串
    """
    parts = []

    # 任务目标
    parts.append(f"## 用户任务\n{instruction}\n")

    # 任务规划（来自 TaskPlanner，首步生成）
    if task_plan:
        parts.append(f"## 任务规划（参考）\n{task_plan}\n")

    # 历史摘要
    if history_summary:
        parts.append(f"## 早期操作摘要\n{history_summary}\n")

    # 最近动作
    if recent_actions_text:
        parts.append(f"## 最近执行的操作\n{recent_actions_text}\n")

    # 当前步骤
    parts.append(f"## 当前步骤\n第 {step_count} 步\n")

    # App 专属示例（当步骤较少时注入，步骤多了则节省 Token）
    if step_count <= 3:
        few_shot = build_few_shot_block(app_name, max_examples=2)
        if few_shot:
            parts.append(few_shot + "\n")

    # 重试提示
    if retry_hint:
        parts.append(f"## ⚠️ 注意\n{retry_hint}\n")

    # 最终指令
    parts.append(
        "## 请根据上方截图分析当前界面，并输出下一步操作\n"
        "严格按照格式：\n"
        "```\n"
        "思考：[分析界面元素，判断下一步]\n"
        "动作：[ACTION:[[params]]]\n"
        "```"
    )

    return "\n".join(parts)


def build_messages(
    system_prompt: str,
    user_text: str,
    image_b64_url: str,
    history_image_messages: List[Dict],
) -> List[Dict]:
    """
    构建完整的 OpenAI messages 列表

    消息顺序：
    1. system（固定）
    2. 历史图文交互（可选）
    3. user（当前截图 + 指令）

    Args:
        system_prompt: 系统提示词
        user_text: 当前步骤的用户提示文字
        image_b64_url: 当前截图的 base64 URL
        history_image_messages: 来自 ContextManager 的历史消息列表

    Returns:
        messages list
    """
    messages = [{"role": "system", "content": system_prompt}]

    # 注入压缩后的历史消息（仅最近几轮的图文对话）
    messages.extend(history_image_messages)

    # 当前步骤：文字 + 截图
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": user_text},
            {"type": "image_url", "image_url": {"url": image_b64_url}},
        ],
    })

    return messages

"""
上下文管理模块（Context Manager）

核心机制：Token-Aware Dynamic Context Window
- 维护完整历史动作记录（纯文字，极低 Token 消耗）
- 对图像历史实施滑动窗口截断（仅保留最近 N 张）
- 早期历史以文字摘要形式压缩，保留关键信息
- 实时追踪 Token 消耗预算，动态调整历史长度
"""

import logging
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)

# 图像历史窗口大小（保留最近 N 轮的截图，每轮 = 1张图 + 1条助手回复）
DEFAULT_IMAGE_WINDOW = 3

# 每张图像估算的 Token 消耗（保守估计，720px 宽图约 1500 Token）
TOKENS_PER_IMAGE_ESTIMATE = 1500


class ContextManager:
    """
    Token 感知的动态上下文窗口管理器

    职责：
    1. 记录每步的完整历史（动作类型、参数、是否成功）
    2. 生成用于 Prompt 的历史摘要文本
    3. 管理图像历史的滑动窗口，控制 Token 开销
    4. 提供对 TaskRunner 注入的 history_messages 的解析接口
    """

    def __init__(self, max_image_window: int = DEFAULT_IMAGE_WINDOW):
        """
        Args:
            max_image_window: 保留多少轮的截图用于上下文
        """
        self.max_image_window = max_image_window
        self._step_records: List[Dict[str, Any]] = []   # 纯文字历史
        self._image_messages: List[Dict[str, Any]] = []  # 图文交互历史
        self._total_image_tokens: int = 0

    def reset(self):
        """在每个测试用例开始前由 Agent.reset() 调用"""
        self._step_records.clear()
        self._image_messages.clear()
        self._total_image_tokens = 0
        logger.debug("[ContextManager] Reset.")

    def add_step(
        self,
        step: int,
        action: str,
        parameters: Dict[str, Any],
        is_valid: bool,
        raw_output: str = "",
        screenshot_b64: Optional[str] = None,
    ):
        """
        记录一步执行结果

        Args:
            step: 步骤编号
            action: 动作类型（CLICK/TYPE/SCROLL/OPEN/COMPLETE）
            parameters: 动作参数
            is_valid: 该步骤是否被 Checker 验证通过（自测阶段可用）
            raw_output: 模型原始输出（调试用）
            screenshot_b64: 当前步的截图 base64（用于历史图像窗口）
        """
        record = {
            "step": step,
            "action": action,
            "parameters": parameters,
            "is_valid": is_valid,
        }
        self._step_records.append(record)

        # 维护图像历史滑动窗口
        if screenshot_b64:
            # 添加用户消息（截图）
            user_img_msg = {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": screenshot_b64}}
                ],
            }
            # 添加助手回复（动作文字）
            assistant_msg = {
                "role": "assistant",
                "content": self._format_action_text(action, parameters),
            }
            self._image_messages.append(user_img_msg)
            self._image_messages.append(assistant_msg)
            self._total_image_tokens += TOKENS_PER_IMAGE_ESTIMATE

    def get_history_image_messages(self) -> List[Dict]:
        """
        获取用于注入 messages 的历史图文消息（截断至最近 N 轮）

        每轮 = 1 条 user(image) + 1 条 assistant(text)
        保留最近 max_image_window 轮 = 最近 2*max_image_window 条消息
        """
        max_msgs = self.max_image_window * 2
        return self._image_messages[-max_msgs:] if self._image_messages else []

    def get_history_summary(self) -> str:
        """
        生成早期历史的文字摘要（用于 Prompt 中的"历史操作摘要"部分）

        策略：当步骤数 > max_image_window 时，将早期步骤压缩为文字
        """
        cutoff = max(0, len(self._step_records) - self.max_image_window)
        early_records = self._step_records[:cutoff]

        if not early_records:
            return ""

        lines = []
        for r in early_records:
            action_str = self._format_action_text(r["action"], r["parameters"])
            status = "✓" if r.get("is_valid") else "?"
            lines.append(f"  步骤{r['step']} [{status}]: {action_str}")

        return "\n".join(lines)

    def get_recent_actions_text(self, n: int = 3) -> str:
        """
        获取最近 n 步的动作描述文字（用于提示词中的"最近操作"部分）
        """
        recent = self._step_records[-n:] if self._step_records else []
        if not recent:
            return ""
        lines = []
        for r in recent:
            action_str = self._format_action_text(r["action"], r["parameters"])
            lines.append(f"  步骤{r['step']}: {action_str}")
        return "\n".join(lines)

    def get_step_count(self) -> int:
        """返回已记录的步骤总数"""
        return len(self._step_records)

    def has_completed(self) -> bool:
        """检查历史中是否已有 COMPLETE 动作"""
        return any(r["action"] == "COMPLETE" for r in self._step_records)

    def get_last_action(self) -> Optional[Dict]:
        """获取最后一步的动作记录"""
        return self._step_records[-1] if self._step_records else None

    @staticmethod
    def _format_action_text(action: str, parameters: Dict[str, Any]) -> str:
        """将动作和参数格式化为可读文字"""
        if action == "CLICK":
            pt = parameters.get("point", [])
            return f"CLICK 坐标 {pt}"
        elif action == "TYPE":
            text = parameters.get("text", "")
            return f"TYPE '{text}'"
        elif action == "SCROLL":
            s = parameters.get("start_point", [])
            e = parameters.get("end_point", [])
            return f"SCROLL 从{s}到{e}"
        elif action == "OPEN":
            app = parameters.get("app_name", "")
            return f"OPEN '{app}'"
        elif action == "COMPLETE":
            return "COMPLETE（任务完成）"
        else:
            return f"{action} {parameters}"

    @staticmethod
    def parse_runner_history(
        history_messages: List[Dict],
        history_actions: List[Dict],
    ) -> Tuple[str, List[Dict]]:
        """
        解析由 TestRunner 注入的 history_messages 和 history_actions

        TestRunner 的格式：
        - history_messages: [..., {role: user, content: [{type: image_url, ...}]},
                                  {role: assistant, content: "Action: CLICK(...)"}]
        - history_actions: [{step, action, parameters, raw_output, is_valid}, ...]

        Returns:
            (summary_text, recent_image_messages)
        """
        # 从 history_actions 生成摘要文字
        lines = []
        for act in history_actions:
            step = act.get("step", "?")
            action = act.get("action", "")
            params = act.get("parameters", {})
            action_str = ContextManager._format_action_text(action, params)
            lines.append(f"  步骤{step}: {action_str}")
        summary = "\n".join(lines) if lines else ""

        # 从 history_messages 提取最近图片消息（每轮2条：user+assistant）
        # 只保留最近 DEFAULT_IMAGE_WINDOW 轮
        max_msgs = DEFAULT_IMAGE_WINDOW * 2
        recent_imgs = history_messages[-max_msgs:] if history_messages else []

        return summary, recent_imgs

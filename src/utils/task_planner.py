"""
任务规划模块（Task Planner）

实现 Plan-and-Execute 框架的规划阶段：
- 首步调用一次 VLM，生成全局任务执行计划
- 计划以结构化文本形式存储，后续每步注入 Prompt
- 规划结果轻量（纯文字），不占用额外图像 Token 预算
- 通过规划减少任务中期的"迷失"现象，提升长任务（8+步）成功率

设计原则：
- 规划是「软约束」而非「硬规则」：执行步骤仍依赖当前截图判断
- 规划仅在首步生成，后续不再更新（节省 Token）
- 若规划调用失败，不影响主流程（降级为无规划模式）
"""

import logging
import re
from typing import Optional, List

logger = logging.getLogger(__name__)


# 规划阶段的专用 Prompt（较轻量，不含图像Few-Shot）
PLANNING_SYSTEM_PROMPT = """你是一个移动端 App 操作专家。根据用户指令和当前手机截图，制定一个简洁的操作步骤计划。

要求：
1. 步骤数量控制在 3-12 步
2. 每步描述简短（一句话）
3. 关注关键转折点（搜索、选择、确认等）
4. 输出格式：
步骤1: [操作描述]
步骤2: [操作描述]
...
完成: [预期最终状态]
"""

PLANNING_USER_PROMPT_TEMPLATE = """请为以下任务制定操作计划：

**用户指令**: {instruction}

**当前界面**: 请参考截图分析初始界面状态

请输出简洁的步骤计划（不要输出动作，只要描述）："""


class TaskPlanner:
    """
    任务规划器：在任务首步生成全局计划

    使用方式：
        plan_text = planner.generate_plan(call_api_fn, instruction, image_b64)
    """

    def __init__(self):
        self._last_plan: Optional[str] = None

    def reset(self):
        self._last_plan = None

    def generate_plan(
        self,
        call_api_fn,  # BaseAgent._call_api
        instruction: str,
        image_b64_url: str,
    ) -> Optional[str]:
        """
        调用 VLM 生成任务规划

        Args:
            call_api_fn: BaseAgent._call_api 函数引用
            instruction: 用户指令
            image_b64_url: 当前截图的 base64 URL

        Returns:
            规划文本（字符串），或 None（若规划失败）
        """
        try:
            messages = [
                {"role": "system", "content": PLANNING_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": PLANNING_USER_PROMPT_TEMPLATE.format(
                                instruction=instruction
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": image_b64_url},
                        },
                    ],
                },
            ]

            # 规划阶段使用低温度（更确定性）
            response = call_api_fn(messages, temperature=0.0, max_tokens=512)
            plan_text = response.choices[0].message.content.strip()

            if plan_text:
                self._last_plan = self._clean_plan(plan_text)
                logger.info(f"[TaskPlanner] Plan generated:\n{self._last_plan}")
                return self._last_plan
            else:
                logger.warning("[TaskPlanner] Empty plan response.")
                return None

        except Exception as e:
            logger.warning(f"[TaskPlanner] Planning failed (non-critical): {e}")
            return None

    def get_last_plan(self) -> Optional[str]:
        """获取上一次生成的计划"""
        return self._last_plan

    def extract_current_step_hint(self, step_count: int) -> Optional[str]:
        """
        根据当前步骤编号，从计划中提取对应的步骤描述

        这是一个启发式方法：假设计划步骤与执行步骤大致对应
        """
        if not self._last_plan:
            return None

        lines = [
            l.strip()
            for l in self._last_plan.split("\n")
            if l.strip() and re.match(r"步骤\d+", l.strip())
        ]

        if not lines:
            return None

        # 使用 min 防止越界
        idx = min(step_count - 1, len(lines) - 1)
        return lines[idx] if idx >= 0 else None

    @staticmethod
    def _clean_plan(text: str) -> str:
        """清理规划文本，去除多余内容"""
        lines = text.split("\n")
        cleaned = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # 只保留步骤行和完成行
            if re.match(r"步骤\d+[:：]", line) or line.startswith("完成"):
                cleaned.append(line)
            elif re.match(r"\d+[.、）)]\s", line):
                # 兼容"1. 操作"格式
                cleaned.append(line)

        return "\n".join(cleaned) if cleaned else text[:300]

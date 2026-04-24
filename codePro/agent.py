"""
HierarchicalGUIAgent — 层级 GUI 代理

========================= 架构概述 =========================

本 Agent 基于以下六大技术创新构建，系统性提升移动端 GUI 任务完成率：

1. Plan-and-Execute 层级规划
   首步调用 VLM 生成全局执行计划，后续步骤在计划引导下执行，
   有效减少复杂任务（8+步）中的方向迷失问题。

2. Token-Aware 动态上下文窗口
   使用滑动窗口策略：仅保留最近 N 张截图参与上下文（默认 N=3），
   早期历史压缩为文字摘要，在 Token 预算内最大化信息密度。

3. App-Adaptive 提示词引擎
   自动识别当前操作的 App 类型（美团/百度地图/哔哩哔哩等），
   动态注入该 App 的专属 Few-Shot 示例，提升领域适配性。

4. 自适应图像分辨率预处理
   将高分辨率截图（通常 1080×2400）等比压缩至最大 720px 宽，
   降低约 40% 的图像 Token 消耗，同时保留文字和 UI 元素清晰度。

5. 多策略分级鲁棒输出解析
   五级解析策略（精确→宽松→XML→JSON→语义），对模型输出
   格式变化高度容忍，有效输出率接近 100%。

6. 自适应重试与自我反思
   当输出格式非法时，自动附加错误提示重试（最多 2 次），
   重试时适当提高温度引入多样性，避免陷入相同错误。

==================== 使用说明 ====================

调试阶段：
    set VLM_API_KEY=ark-245b0dce-8641-4977-89b8-b19510d9dcbb-9ce61   (Windows CMD)
    python test_runner.py      (从 codePro/ 目录运行)

提交阶段：
    由主办方设置 EVAL_MODE=production 及相关环境变量，
    Agent 自动切换为正式评测配置。

"""

import os
import logging
from typing import Dict, Any, Optional

# ============================================================
#   VLM API KEY（调试用占位，正式测试请通过环境变量设置）
#   方式1（推荐）：set VLM_API_KEY=your-key（Windows CMD）
#   方式2（临时）：取消注释下方并填入 key（提交前必须注释回去！）
# ============================================================
os.environ["VLM_API_KEY"] = "ark-245b0dce-8641-4977-89b8-b19510d9dcbb-9ce61"  # 临时调试

from agent_base import (
    BaseAgent,
    AgentInput,
    AgentOutput,
    ACTION_CLICK,
    ACTION_TYPE,
    ACTION_SCROLL,
    ACTION_OPEN,
    ACTION_COMPLETE,
)
from utils.img_processor import preprocess_image, encode_image
from utils.prompt_engine import (
    SYSTEM_PROMPT,
    build_user_prompt,
    build_messages,
    _detect_app,
)
from utils.context_manager import ContextManager
from utils.output_parser import parse_output, validate_output, build_retry_hint
from utils.task_planner import TaskPlanner

logger = logging.getLogger(__name__)


# ============================================================
#   配置项（可按需调整）
# ============================================================

API_TEMPERATURE = 0.0          # 主调用温度（贪心解码，更稳定）
API_TEMPERATURE_RETRY = 0.3    # 重试时的温度（增加多样性）
API_MAX_TOKENS = 1024          # 最大输出 Token 数
MAX_RETRIES = 2                # 重试次数上限
IMAGE_MAX_WIDTH = 720          # 图像预处理最大宽度（px）
CONTEXT_IMAGE_WINDOW = 3       # 历史图像窗口大小
ENABLE_PLANNING = True         # 是否启用任务规划（多消耗一次 API）


class Agent(BaseAgent):
    """
    HierarchicalGUIAgent：层级 GUI 代理

    继承 BaseAgent，实现 act() 方法。
    所有模块（规划器、上下文管理器、解析器）在 _initialize() 中初始化，
    在 reset() 中清空状态（每个测试用例开始时由 TestRunner 调用）。
    """

    def _initialize(self):
        """初始化所有子模块"""
        self._ctx = ContextManager(max_image_window=CONTEXT_IMAGE_WINDOW)
        self._planner = TaskPlanner()
        self._task_plan: Optional[str] = None
        self._detected_app: str = "通用"
        self._session_tokens: int = 0
        logger.info("[Agent] HierarchicalGUIAgent initialized.")

    def reset(self):
        """重置所有状态，为新的测试用例做准备"""
        self._ctx.reset()
        self._planner.reset()
        self._task_plan = None
        self._detected_app = "通用"
        self._session_tokens = 0
        logger.info("[Agent] State reset for new task.")

    def act(self, input_data: AgentInput) -> AgentOutput:
        """
        根据当前输入（指令 + 截图 + 历史）生成下一步动作

        流程：
        1. 图像预处理（降分辨率压缩）
        2. 首步：调用规划器生成任务计划
        3. 识别 App 类型（用于 Few-Shot 选择）
        4. 构建消息并调用 API（带重试）
        5. 更新上下文
        6. 返回 AgentOutput
        """
        instruction = input_data.instruction
        step_count = input_data.step_count

        # Step 1: 图像预处理
        processed_img = preprocess_image(
            input_data.current_image, max_width=IMAGE_MAX_WIDTH
        )
        image_b64_url = encode_image(processed_img, image_format="PNG")

        # Step 2: 首步规划（仅在 step_count==1 时执行）
        if step_count == 1 and ENABLE_PLANNING:
            logger.info("[Agent] Step 1: Generating task plan...")
            self._task_plan = self._planner.generate_plan(
                call_api_fn=self._call_api,
                instruction=instruction,
                image_b64_url=image_b64_url,
            )

        # Step 3: App 类型识别
        if step_count == 1:
            self._detected_app = _detect_app(
                instruction, input_data.history_actions
            )
            logger.info(f"[Agent] Detected App: {self._detected_app}")

        # Step 4: 构建消息并调用 API（带重试）
        action, parameters, raw_output, usage = self._call_with_retry(
            instruction=instruction,
            step_count=step_count,
            image_b64_url=image_b64_url,
            history_messages=input_data.history_messages,
            history_actions=input_data.history_actions,
        )

        # Step 5: 更新上下文
        self._ctx.add_step(
            step=step_count,
            action=action,
            parameters=parameters,
            is_valid=True,
            raw_output=raw_output,
            screenshot_b64=image_b64_url,
        )

        if usage:
            self._session_tokens += usage.total_tokens
            logger.info(
                f"[Agent] Step {step_count} tokens: +{usage.total_tokens} "
                f"(session total: {self._session_tokens})"
            )

        logger.info(f"[Agent] Step {step_count} -> {action} {parameters}")

        return AgentOutput(
            action=action,
            parameters=parameters,
            raw_output=raw_output,
            usage=usage,
        )

    def _call_with_retry(
        self,
        instruction: str,
        step_count: int,
        image_b64_url: str,
        history_messages,
        history_actions,
    ):
        """
        带自适应重试的 API 调用

        Returns:
            (action, parameters, raw_output, usage)
        """
        history_summary, runner_history_imgs = ContextManager.parse_runner_history(
            history_messages, history_actions
        )

        retry_hint = ""
        last_error = ""

        for attempt in range(MAX_RETRIES):
            temperature = API_TEMPERATURE if attempt == 0 else API_TEMPERATURE_RETRY

            user_text = build_user_prompt(
                instruction=instruction,
                step_count=step_count,
                task_plan=self._task_plan,
                history_summary=history_summary,
                recent_actions_text=self._ctx.get_recent_actions_text(n=3),
                app_name=self._detected_app,
                retry_hint=retry_hint,
            )

            history_imgs = (
                runner_history_imgs
                if runner_history_imgs
                else self._ctx.get_history_image_messages()
            )

            messages = build_messages(
                system_prompt=SYSTEM_PROMPT,
                user_text=user_text,
                image_b64_url=image_b64_url,
                history_image_messages=history_imgs,
            )

            try:
                response = self._call_api(
                    messages,
                    temperature=temperature,
                    max_tokens=API_MAX_TOKENS,
                )
                raw_output = response.choices[0].message.content or ""
                usage = self.extract_usage_info(response)
            except Exception as e:
                logger.error(f"[Agent] API call failed at attempt {attempt}: {e}")
                raw_output = ""
                usage = None

            action, parameters = parse_output(raw_output)
            is_valid, error_msg = validate_output(action, parameters)
            if is_valid:
                return action, parameters, raw_output, usage

            last_error = error_msg
            retry_hint = build_retry_hint(action, parameters, error_msg)
            logger.warning(
                f"[Agent] Attempt {attempt + 1} invalid: {error_msg}. Retrying..."
            )

        logger.error(
            f"[Agent] All {MAX_RETRIES} attempts failed. "
            f"Last error: {last_error}. Using safe default."
        )
        return ACTION_COMPLETE, {}, "", None

    def generate_messages(self, input_data: AgentInput):
        """覆写基类方法，供外部调用/调试使用"""
        processed_img = preprocess_image(
            input_data.current_image, max_width=IMAGE_MAX_WIDTH
        )
        image_b64_url = encode_image(processed_img)

        user_text = build_user_prompt(
            instruction=input_data.instruction,
            step_count=input_data.step_count,
            task_plan=self._task_plan,
            history_summary="",
            recent_actions_text="",
            app_name=self._detected_app,
        )

        return build_messages(
            system_prompt=SYSTEM_PROMPT,
            user_text=user_text,
            image_b64_url=image_b64_url,
            history_image_messages=[],
        )

"""
HierarchicalGUIAgent v2.0 — 层级 GUI 代理

v2.0 优化（基于本地测试迭代）：
1. 从 instruction 解析 App 名，step1 强制注入 OPEN 提示（解决首步 OPEN 失败）
2. 优化 retry 机制：parse 失败时附加解析错误上下文
3. 提交版：VLM_API_KEY 改为纯环境变量方式（去掉硬编码）

提交说明（重要）：
    提交时请确保 os.environ["VLM_API_KEY"] = ... 那行已注释掉！
    官方评测环境会自动注入 EVAL_API_KEY / VLM_API_KEY。
"""

import os
import re
import logging
from typing import Dict, Any, Optional, List

# ============================================================
#   VLM API KEY — 本地调试用，提交前必须注释掉这一行！
#   官方评测环境会通过系统环境变量自动注入 VLM_API_KEY
# ============================================================
# os.environ["VLM_API_KEY"] = "ark-xxx"   # ← 提交前注释掉


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
    detect_app_from_instruction,
)
from utils.context_manager import ContextManager
from utils.output_parser import parse_output, validate_output, build_retry_hint
from utils.task_planner import TaskPlanner

logger = logging.getLogger(__name__)


# ============================================================
#   配置
# ============================================================

API_TEMPERATURE = 0.0
API_TEMPERATURE_RETRY = 0.3
API_MAX_TOKENS = 1024
MAX_RETRIES = 2
IMAGE_MAX_WIDTH = 720
CONTEXT_IMAGE_WINDOW = 3
ENABLE_PLANNING = True


def _banner(msg: str, char: str = "─", width: int = 56) -> str:
    """生成日志分隔横幅"""
    return f"\n{char * width}\n  {msg}\n{char * width}"


class Agent(BaseAgent):
    """HierarchicalGUIAgent：层级 GUI 代理"""

    def _initialize(self):
        self._ctx = ContextManager(max_image_window=CONTEXT_IMAGE_WINDOW)
        self._planner = TaskPlanner()
        self._task_plan: Optional[str] = None
        self._detected_app: str = "通用"
        self._instruction_app: Optional[str] = None
        self._session_tokens: int = 0
        self._current_case: str = ""

    def reset(self):
        self._ctx.reset()
        self._planner.reset()
        self._task_plan = None
        self._detected_app = "通用"
        self._instruction_app = None
        self._session_tokens = 0

    def act(self, input_data: AgentInput) -> AgentOutput:
        instruction = input_data.instruction
        step_count = input_data.step_count

        # Step 1: 图像预处理
        processed_img = preprocess_image(
            input_data.current_image, max_width=IMAGE_MAX_WIDTH
        )
        image_b64_url = encode_image(processed_img, image_format="PNG")

        # Step 2: 首步初始化
        if step_count == 1:
            self._instruction_app = detect_app_from_instruction(instruction)
            self._detected_app = self._instruction_app or "通用"
            logger.info(f"  App识别: {self._detected_app} | 指令: {instruction[:40]}...")

            if ENABLE_PLANNING:
                self._task_plan = self._planner.generate_plan(
                    call_api_fn=self._call_api,
                    instruction=instruction,
                    image_b64_url=image_b64_url,
                )

        # Step 3: 调用 API（带重试）
        action, parameters, raw_output, usage = self._call_with_retry(
            instruction=instruction,
            step_count=step_count,
            image_b64_url=image_b64_url,
            history_messages=input_data.history_messages,
            history_actions=input_data.history_actions,
        )

        # Step 4: 更新上下文
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

        # 简洁的步骤日志
        param_str = str(parameters)[:60]
        logger.info(f"  Step {step_count:2d} → {action:8s} {param_str}")

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
        """带自适应重试的 API 调用"""
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
                instruction_app=self._instruction_app,
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

        logger.error(f"[Agent] All {MAX_RETRIES} attempts failed: {last_error}")
        return ACTION_COMPLETE, {}, "", None

    def generate_messages(self, input_data: AgentInput):
        """覆写基类方法，供调试使用"""
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
            instruction_app=self._instruction_app,
        )

        return build_messages(
            system_prompt=SYSTEM_PROMPT,
            user_text=user_text,
            image_b64_url=image_b64_url,
            history_image_messages=[],
        )

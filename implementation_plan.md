# 2026中兴捧月大赛「智能代理」赛道 — 分析与高分方案

## 一、比赛任务解读

### 核心任务
构建一个**移动端 GUI Agent**，该 Agent 接收：
- 用户自然语言指令（如"帮我在美团订一份外卖"）
- 当前手机屏幕截图（Android 手机，多种机型/App）

并输出下一步要执行的操作动作（CLICK、TYPE、SCROLL、OPEN、COMPLETE）及其参数。

### 本质是什么
这是一个**多模态视觉-语言推理任务（VLM + GUI Grounding）**：
- 模型要**看懂**当前界面（定位 UI 元素）
- 模型要**理解**用户意图和任务进度
- 模型要**规划**下一步操作（正确的动作类型 + 精准的坐标）

### 关键约束
| 约束 | 说明 |
|------|------|
| 模型固定 | 主办方使用 `doubao-seed-1-6-vision-250815`，选手无法换模型 |
| API 固定 | 只能通过 `_call_api()` 调用，禁止自建客户端 |
| 坐标归一化 | 输出坐标必须是 [0, 1000] 的归一化坐标 |
| Token 限制 | 全程总 Token 限制 1,200,000 |
| 最大步数 | 每个任务最多 45 步 |
| 提交大小 | 解压后总大小 ≤ 20MB |
| Python | 3.10.12 |

### 评分公式
```
总得分 = 打榜得分×60% + Agent代码×20% + 设计方案及创新×20%
```

**关键洞察**：
- 打榜（60%）：每个任务只有全部步骤正确才得 1 分，错一步全 0。因此**每步精准度**极为重要
- 代码（20%）：代码质量、可读性、工程设计
- 方案创新（20%）：算法设计文档的创新性描述

---

## 二、测试数据分析

根据现有 11 个测试用例，覆盖的应用包括：
- 美团外卖（订餐流程）
- 百度地图（导航/搜索）
- 哔哩哔哩（视频播放）
- 抖音（短视频操作）
- 快手（短视频操作）
- 爱奇艺（视频播放）
- 芒果TV
- 腾讯视频
- 喜马拉雅（音频）
- 趣...

**高频动作类型**：OPEN → CLICK × N → TYPE → CLICK → COMPLETE
**挑战点**：
1. 搜索框点击需精准定位
2. 列表滚动选择需理解内容
3. 多步骤依赖（前步错 → 后步全错）

---

## 三、核心问题分析

### 基础 Agent 的缺陷（需要解决的问题）

| 问题 | 现象 | 影响 |
|------|------|------|
| 无历史感知 | 每步只看当前截图，不知道做过什么 | 重复操作、忘记任务进度 |
| 无任务规划 | 没有宏观计划，走一步看一步 | 迷失在复杂任务中 |
| 提示词简陋 | 没有精确的坐标输出引导 | 输出格式错误、坐标不准 |
| 无错误恢复 | 一步错则任务中断 | 得分归零 |
| 图像原尺寸 | 直接传大图，浪费 Token | Token 超限 |
| 输出解析脆弱 | 正则不鲁棒 | 格式合法但解析失败 |

---

## 四、高分 Agent 框架设计方案

### 架构总览

```
输入: instruction + screenshot + history
         │
    ┌────▼────────────────────────────────────────────┐
    │              HierarchicalGUIAgent                │
    │                                                  │
    │  ┌─────────────┐   ┌──────────────────────────┐ │
    │  │ TaskPlanner │   │    ContextManager         │ │
    │  │ (首步规划)  │   │  (历史消息/动作压缩管理)  │ │
    │  └──────┬──────┘   └──────────────┬───────────┘ │
    │         │                         │             │
    │  ┌──────▼─────────────────────────▼──────────┐  │
    │  │         PromptEngine                       │  │
    │  │  (结构化Prompt + 少样本示例 + 错误恢复提示) │  │
    │  └──────────────────────┬────────────────────┘  │
    │                         │                        │
    │  ┌──────────────────────▼────────────────────┐  │
    │  │         ImagePreprocessor                  │  │
    │  │  (图像缩放 + UI元素标注 + 网格坐标参考)    │  │
    │  └──────────────────────┬────────────────────┘  │
    │                         │                        │
    │              _call_api(messages)                  │
    │                         │                        │
    │  ┌──────────────────────▼────────────────────┐  │
    │  │         RobustOutputParser                 │  │
    │  │  (多策略解析 + 坐标验证 + fallback机制)    │  │
    │  └───────────────────────────────────────────┘  │
    └─────────────────────────────────────────────────┘
         │
    输出: AgentOutput(action, parameters)
```

### 创新点一：层级任务规划（HierarchicalPlanning）

**原理**：首步调用模型生成整个任务的「宏观计划」，后续步骤在计划指引下执行

```python
# 第一步：任务规划（仅调用一次，消耗少量Token）
planning_prompt = """
你是一个移动端操作专家。请为以下任务制定操作计划：
任务：{instruction}
当前界面：{screenshot}

请输出：
1. 预计步骤数
2. 关键步骤清单（每步一行）
3. 可能遇到的界面类型
"""

# 后续每步：结合计划执行
action_prompt = """
当前计划进度：第{step}步，共约{total}步
已完成：{completed_steps}
下一步计划：{next_plan_step}
当前界面：...
"""
```

**优势**：减少走错路概率，提升复杂任务（10+步）的成功率

### 创新点二：智能历史压缩（ContextCompression）

**问题**：直接携带所有历史图片会超 Token 限制（每张截图约 4000-8000 Token）

**方案**：
```python
class ContextManager:
    def build_history(self, history_messages, history_actions, max_images=3):
        """
        动态历史管理策略：
        1. 只保留最近 N 张截图（默认3张）
        2. 将早期动作转为文字摘要
        3. 对截图做降分辨率压缩（1080→540宽度）
        """
        text_summary = self._summarize_actions(history_actions[:-3])
        recent_images = history_messages[-6:]  # 最近3轮对话
        return text_summary, recent_images
    
    def _summarize_actions(self, actions):
        """将历史动作压缩为文字描述"""
        lines = []
        for a in actions:
            action_str = self._format_action(a)
            lines.append(f"步骤{a['step']}: {action_str}")
        return "\n".join(lines)
```

### 创新点三：结构化提示工程（StructuredPromptEngine）

**核心设计**：将提示词分为固定部分（可利用 KV Cache）和动态部分

```python
SYSTEM_PROMPT = """
# 角色
你是一个专业的移动端GUI操作Agent，负责控制安卓手机完成用户交代的任务。

# 坐标系统
- 所有坐标使用归一化坐标系，范围 [0, 1000]
- x=0 为屏幕左边，x=1000 为右边
- y=0 为屏幕顶部，y=1000 为底部

# 动作空间
| 动作 | 格式 | 说明 |
|------|------|------|
| CLICK | CLICK:[[x, y]] | 点击指定坐标 |
| TYPE | TYPE:['文本内容'] | 输入文字 |
| SCROLL | SCROLL:[[x1,y1],[x2,y2]] | 从(x1,y1)滑动到(x2,y2) |
| OPEN | OPEN:['应用名称'] | 打开App |
| COMPLETE | COMPLETE:[] | 任务完成 |

# 思考格式
```
思考：[分析当前界面，判断需要执行的操作]
动作：[ACTION:[[params]]]
```

# 重要规则
1. 坐标必须点击在可交互元素的中心或文字区域
2. 如果任务已完成，立即输出 COMPLETE:[]
3. 搜索框通常在屏幕顶部 y=[50,150] 范围内
4. 列表项通常在 y=[200,900] 范围内
"""
```

**少样本示例注入（Few-Shot）**：
```python
FEW_SHOT_EXAMPLES = [
    {
        "scenario": "美团搜索框",
        "instruction": "搜索外卖",
        "thought": "当前界面显示美团首页，需要点击搜索框",
        "action": "CLICK:[[500, 95]]"
    },
    # ... 更多示例
]
```

### 创新点四：视觉增强预处理（ImagePreprocessing）

**图像降分辨率**：
```python
def preprocess_image(image: Image.Image) -> Image.Image:
    """
    智能图像预处理：
    1. 将图片宽度压缩到 720px（保持宽高比）
    2. 仅在高分辨率图(>1440px高)时应用
    3. 使用 LANCZOS 高质量缩放
    """
    max_width = 720
    if image.width > max_width:
        ratio = max_width / image.width
        new_size = (max_width, int(image.height * ratio))
        image = image.resize(new_size, Image.LANCZOS)
    return image
```

**网格坐标参考线（可选）**：在图像上叠加半透明坐标网格，帮助模型更准确地输出坐标

### 创新点五：多策略鲁棒输出解析（RobustParser）

```python
class RobustOutputParser:
    """多策略解析，逐级 fallback"""
    
    PATTERNS = {
        "CLICK": [
            r'CLICK:\[\[(\d+),\s*(\d+)\]\]',          # 标准格式
            r'click\(point=[\'"]<point>(\d+)\s+(\d+)</point>[\'"]',  # 原始格式
            r'点击.*?[坐标：].*?(\d+)[,，](\d+)',        # 中文格式
            r'CLICK.*?(\d{2,3}),\s*(\d{2,3})',           # 宽松匹配
        ],
        # ... 其他动作
    }
    
    def parse(self, raw_output: str) -> tuple[str, dict]:
        for action_type, patterns in self.PATTERNS.items():
            for pattern in patterns:
                match = re.search(pattern, raw_output, re.IGNORECASE)
                if match:
                    return self._build_output(action_type, match)
        
        # 最终 fallback：要求模型重新输出
        return self._fallback_parse(raw_output)
```

### 创新点六：自适应重试机制（AdaptiveRetry）

```python
def act(self, input_data: AgentInput) -> AgentOutput:
    max_retries = 2
    
    for attempt in range(max_retries):
        messages = self.generate_messages(input_data, retry=attempt)
        response = self._call_api(messages, temperature=0.0 if attempt==0 else 0.3)
        output = self.parser.parse(response.choices[0].message.content)
        
        if self._validate_output(output):
            return output
        
        # 重试时加入错误提示
        error_hint = f"上次输出格式错误，请严格按照格式输出动作"
    
    # 兜底返回
    return self._safe_fallback(input_data)
```

---

## 五、提示词精细设计

### 任务描述模板

```python
USER_PROMPT_TEMPLATE = """
## 当前任务
{instruction}

## 历史操作摘要
{history_summary}

## 当前步骤
第 {step_count} 步

## 最近执行的操作
{recent_actions}

## 请根据当前截图分析界面，执行下一步操作
注意：
- 如果你看到任务已经完成（到达目标页面/状态），请输出 COMPLETE:[]
- 搜索按钮/确认按钮通常在输入框右侧或键盘上方
- 点击坐标要精确落在元素内部，不要点到边缘
"""
```

### 关键 Prompt 技巧

1. **界面类型引导**：告知模型当前可能是"搜索结果页"、"商品列表页"、"结算页"等，引导其聚焦关键区域
2. **禁止幻觉提示**：`"如果你不确定坐标，宁可输出一个大致正确的点，也不要输出格式错误的内容"`
3. **任务完成判断**：`"当你看到订单成功、支付成功、或任务明确完成的界面时，输出COMPLETE"`

---

## 六、代码结构设计

```
submission.zip
├── doc/
│   └── 算法设计说明文档.pdf
└── src/
    ├── agent_base.py          # 禁止修改
    ├── agent.py               # 核心：HierarchicalGUIAgent
    ├── requirements.txt
    └── utils/
        ├── __init__.py
        ├── image_utils.py     # 图像预处理工具
        ├── prompt_engine.py   # 提示词管理
        ├── context_manager.py # 历史上下文管理
        ├── output_parser.py   # 鲁棒输出解析
        └── task_planner.py    # 任务规划模块
```

### agent.py 骨架

```python
from agent_base import BaseAgent, AgentInput, AgentOutput
from utils.prompt_engine import PromptEngine
from utils.context_manager import ContextManager
from utils.output_parser import RobustOutputParser
from utils.task_planner import TaskPlanner

class HierarchicalGUIAgent(BaseAgent):
    
    def _initialize(self):
        self.prompt_engine = PromptEngine()
        self.context_manager = ContextManager(max_image_history=3)
        self.parser = RobustOutputParser()
        self.planner = TaskPlanner()
        self._task_plan = None
        self._history_actions = []
    
    def reset(self):
        self._task_plan = None
        self._history_actions = []
        self.context_manager.reset()
    
    def act(self, input_data: AgentInput) -> AgentOutput:
        # 1. 首步生成任务规划
        if input_data.step_count == 1:
            self._task_plan = self.planner.plan(input_data)
        
        # 2. 构建消息
        messages = self.prompt_engine.build_messages(
            input_data=input_data,
            task_plan=self._task_plan,
            context=self.context_manager.get_context()
        )
        
        # 3. 调用API（带重试）
        for attempt in range(2):
            response = self._call_api(messages, temperature=0.0)
            raw = response.choices[0].message.content
            action, params = self.parser.parse(raw)
            
            if self.parser.is_valid(action, params):
                break
        
        # 4. 更新历史
        self.context_manager.add_step(input_data, action, params)
        
        return AgentOutput(
            action=action,
            parameters=params,
            raw_output=raw,
            usage=self.extract_usage_info(response)
        )
```

---

## 七、算法设计文档（用于20%创新评分）

文档应包含以下内容，突出创新性：

1. **技术架构图**：层级Agent架构，清晰展示各模块
2. **创新点说明**：
   - 层级规划（Plan-then-Execute）
   - 动态上下文压缩（节省Token，扩展记忆）
   - 多策略解析（提升鲁棒性）
   - 视觉预处理（降本增效）
3. **实验对比**：基线 vs 改进方案的步骤准确率对比表
4. **消融实验**：各模块的贡献分析
5. **案例分析**：典型成功/失败案例的可视化展示

---

## 八、实施路线图

### 阶段一：基础实现（第1-2天）
- [ ] 完成 `agent.py` 基础版本（继承BaseAgent，能跑通测试）
- [ ] 实现鲁棒输出解析器（RobustOutputParser）
- [ ] 实现图像预处理（降分辨率）
- [ ] 在本地11个测试用例上跑通，记录基线得分

### 阶段二：增强优化（第3-4天）
- [ ] 实现 ContextManager（历史压缩）
- [ ] 设计精细化 Prompt（系统提示 + 少样本示例）
- [ ] 实现 TaskPlanner（首步规划）
- [ ] 自适应重试机制

### 阶段三：精调打磨（第5-6天）
- [ ] 针对各 App 类型（外卖/地图/视频）定制 Prompt
- [ ] Token 使用量优化（确保在限制内）
- [ ] 针对失败案例逐一分析并改进
- [ ] 打榜提交

### 阶段四：文档撰写（第7天）
- [ ] 撰写算法设计说明文档（含架构图、创新点、对比实验）
- [ ] 整理代码注释和可读性
- [ ] 打包 submission.zip（验证大小 ≤ 20MB）

---

## 九、得分预期

| 项目 | 优化前估计 | 优化后预期 |
|------|-----------|-----------|
| 步骤准确率 | ~40% | ~70-80% |
| 任务完成率（打榜60%） | ~20% | ~50-65% |
| Agent代码（20%） | 基础分 | 85+/100 |
| 设计方案（20%） | 基础分 | 88+/100 |
| **综合得分** | ~低分 | **竞争前10%** |

---

## 十、待确认的开放问题

> [!IMPORTANT]
> 请确认以下事项后再开始实施：

1. **本地测试环境**：是否已配置 `VLM_API_KEY`？需要先能在本地跑通测试
2. **任务规划的 Token 成本**：首步多调用一次 API 是否在 Token 预算内？（可以评估后决定是否启用）
3. **提交时间节点**：还有多少天可以打磨？影响各阶段分配
4. **是否要做 App 专项优化**：针对美团/地图等常见 App 写专用 Few-Shot 示例？

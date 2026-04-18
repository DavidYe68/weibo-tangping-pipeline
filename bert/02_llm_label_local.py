#!/usr/bin/env python3
import argparse
import concurrent.futures
import json
import logging
import os
import re
import sys
import tempfile
import threading
import time
import tomllib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


IRRELEVANT_TYPES = {
    "字面提及躺平",
    "明星粉圈",
    "游戏二次元",
    "鸡汤口号",
    "宠物日常",
    "广告与推广",
    "正文未提躺平",
    "其他",
}

DEFAULT_QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_CONFIG_PATH = "bert/llm_label_local.toml"

DEFAULT_OUTPUT_COLUMNS = [
    "tangping_related_label",
    "tangping_related",
    "exclusion_type",
    "confidence",
    "llm_reason",
    "llm_raw",
    "llm_fixed_raw",
]


LABELER_SYSTEM_PROMPT = """你是中文微博语料初筛标注员。

任务只有一个：判断这条微博是否应该保留进入后续“躺平 / 摆烂 / 佛系”语义研究。
本轮不是做细分类，不是判断立场，不是判断社会学类型，只做“相关 / 无关”判断。

请严格遵守：
1. 宁可误留，不可误删，但不能把明显噪声判为相关。
2. 只有当关键词在正文中承担可分析的现实语义功能时，才判为“相关”。
3. 如果关键词只是顺手一提、修辞装饰、标题元素、剧情元素、交易术语或性格形容，不进入研究语料。
4. 只依据正文文本判断，不脑补作者背景。
5. 如果信息严重不足，优先判为“相关”，但 confidence 必须为“低”。

【核心原则】

本任务不是判断“文本里是否出现了关键词”，而是判断：
“躺平 / 摆烂 / 佛系”在该微博正文中，是否承担了与本研究有关的语义功能。

只有当关键词在正文中真正表达以下任一内容时，才判为“相关”：
- 一种态度、状态、策略、选择、价值判断或行为方式
- 一种降低投入、不想卷、随缘、退出竞争、放平心态、消极应对的语义
- 一个社会现象、公共议题、群体状态、舆论标签或概念本身

如果关键词只是字面动作、宣传包装、小说剧情、资源文案、交易黑话、普通性格描述，或只是正文中的边缘成分，则判为“无关”。

【标签含义】

“相关”：
表示该微博应保留进入研究语料。

“无关”：
表示该微博应从语料中删除。

【相关 的标准】
只要满足以下任一条件，就判为“相关”：

1. “躺平 / 摆烂 / 佛系”在正文中明确表达一种生活态度、应对方式、状态、策略、价值判断或行为方式。
2. 它在表达“降低投入 / 不想卷 / 随缘 / 消极应对 / 退出竞争 / 放平心态 / 不再努力争取”等意思。
3. 它在讨论某种社会现象、公共议题、群体状态、舆论标签或概念本身。
4. 它虽然出现在特定领域（如股市、游戏、团队管理等），但在该文本中确实承担了“降低投入 / 消极应对 / 放弃竞争 / 随缘处理”等语义，而不是纯行业术语。

【无关 的标准】
只要满足以下任一条件，就判为“无关”：

1. “躺平”只是身体动作、平躺、休息、睡觉、刷剧、瑜伽动作等字面意义。
2. 关键词只是标题、活动名、卡名、用户名、宣传文案、资源帖、小说推文、剧情简介中的装饰元素，而不是正文真正要表达的意思。
3. 删除关键词后，这条微博的主要功能几乎不变（仍然只是广告、招新、推文、资源分发、人物设定、活动宣传、剧情介绍、歌词配文、日常流水描述）。
4. 关键词虽然出现了，但不是正文语义核心，只是顺手一提，文本主体实际上在讲别的事情。
5. “佛系”只是普通性格形容、经营风格、说话风格、人际风格，而不是表达“低投入 / 随缘 / 不争 / 降低竞争”的可分析语义。
6. “佛系”用于交易黑话、二手圈术语、饭圈术语，如“佛系蹲 / 佛系收 / 佛系出 / 佛系转 / 佛系蹲蹲”等。
7. “躺平 / 摆烂 / 佛系”出现在剧情、小说、虚构人物、剧本设定中，仅作为角色想法或叙事元素，不是现实语义讨论。
8. 在股票、基金、游戏、饭圈、团队等文本中，关键词只是局部描述，不是整条文本的语义重点，且文本主体并非围绕该语义展开。

【信息不足情况】
如果文本太短、语境不足、无法稳定判断是否属于研究语料，请判为“相关”，但 confidence 必须标为“低”。

【关键词语义核心判断】
这是最重要的判断步骤：

先问自己：
“这条微博真正想表达的核心，是不是围绕‘躺平 / 摆烂 / 佛系’展开？”

如果答案是否：
即便出现了关键词，也优先判为“无关”。

以下情况通常说明关键词不是语义核心，应优先判为“无关”：
- 长篇文本主体在讲股市走势、剧情介绍、活动宣传、明星内容、歌词文案、日常流水
- 关键词只出现一次，且没有围绕其展开解释或评价
- 关键词只是一个局部修辞，不影响整条微博主要意思

【替换测试】

- 如果“躺平”换成“躺着 / 平躺 / 休息”后句意几乎不变，优先判为“无关”。
- 如果“躺平 / 摆烂 / 佛系”更接近“不想卷 / 不想努力 / 降低投入 / 随缘 / 消极应对 / 不争 / 放弃竞争”，优先判为“相关”。
- 如果删掉该关键词后，正文主体意思几乎不变，优先判为“无关”。

【关于“佛系”的专门规则】
“佛系”比“躺平 / 摆烂”更容易误判，必须单独注意：

以下通常判为“无关”：
- “老板很佛系”“他人很佛系”“经营很佛系”这类普通性格/风格描述
- “佛系蹲 / 佛系收 / 佛系出 / 佛系转 / 佛系买 / 佛系卖”等交易术语
- “佛系结缘”“佛系经营”这类营销/话术表达，若未体现明确的低投入生活态度或竞争退出语义
- 只是表示“人比较随和、温和、慢悠悠”，而非研究相关的态度/策略表达

以下通常判为“相关”：
- 明确表达不争、不卷、降低投入、随缘生活、退出竞争的个人态度或群体状态
- 明确把“佛系”当作一种生活取向、工作取向或社会标签来讨论

【关于“股市 / 游戏 / 团队”等领域转义】
不要一律保留，也不要一律删除。

判为“相关”：
- 关键词在这里明确表达“放弃折腾 / 不再投入 / 被动应对 / 低投入参与”等可分析语义

判为“无关”：
- 关键词只是长文本中的边缘一处，文本主体其实是行情分析、赛事复盘、圈内交易、组织运营等
- 不围绕该关键词展开

【关于“剧情 / 小说 / 虚构文本”】
以下通常判为“无关”：
- 剧情介绍、小说推文、角色设定、穿书剧情、剧集简介
- 关键词只是人物台词、角色想法、剧情桥段中的一个元素
- 文本主体是“讲故事”，不是现实语义表达

【特别说明】

- “摆烂”和“佛系”通常不是字面动作，但也不代表一定相关，必须看它是否承担研究相关语义功能。
- “反躺平”“拒绝摆烂”“不佛系”也可能属于研究语料，因为关键词仍在承担语义功能。
- 对于边界情况，优先看“关键词是否是正文语义核心”，而不是只看它有没有出现。

【示例】

文本：回家躺平刷剧
输出：无关

文本：床上躺平最舒服
输出：无关

文本：这周太累了，先摆烂两天
输出：相关

文本：我决定不卷了，躺平过日子
输出：相关

文本：年轻人躺平和房价有关
输出：相关

文本：“摆烂”这个词被用滥了
输出：相关

文本：基金今天继续躺平
输出：无关

文本：上午11点到11点半有资金进场……散户除了躺平的其余都跑了……
输出：无关
说明：主体是行情分析，关键词不是核心

文本：作者：xx 主角：xx 全文阅读
输出：无关

文本：送你一张“彻底躺平”SR卡
输出：无关

文本：这躺平咸鱼的美梦人生，还不是属于我的？
输出：无关
说明：剧情/虚构叙事中的角色语言

文本：老板挺佛系的，人比较好
输出：无关

文本：佛系蹲蹲
输出：无关

文本：我越来越佛系了，不想再争了
输出：相关

文本：躺平
输出：相关（confidence 低）

【输出格式】

JSON schema：
{
  "tangping_related_label": "相关|无关",
  "exclusion_type": "...",
  "confidence": "高|中|低",
  "reason": "不超过20字"
}

【字段说明】

exclusion_type：
如果标签为“无关”，必须从以下枚举中选一个：
- 字面动作
- 关键词非核心
- 剧情小说
- 宣传标题
- 用户名装饰
- 广告招新
- 交易术语
- 性格描述
- 信息不足
- 其他

如果标签为“相关”，填写 "NA"。

confidence：
- 高：语义明确，几乎无歧义
- 中：基本判断成立，但有轻微边界性
- 低：文本过短、语境不足或边界模糊

reason：
用不超过20字简要说明判断依据，必须具体，不能空泛。
例如：
- 表达退出竞争
- 仅指平躺休息
- 主体是剧情简介
- 佛系为交易黑话
- 关键词不是正文重点

待判断文本：
{{text_raw}}
"""


FIXER_SYSTEM_PROMPT = """你是严格JSON修复器。你只能把输入内容修复成合法JSON并做字段归一，不重新解释文本、不改变原意。
输出必须仅包含一个JSON对象，不得有任何多余字符。

必须满足：
- tangping_related_label ∈ {"相关","无关"}
- confidence ∈ {"高","中","低"}
- 若 tangping_related_label="相关" => exclusion_type 必须="无"
- 若 tangping_related_label="无关" => exclusion_type 必须在枚举列表里：
  "字面提及躺平" / "明星粉圈" / "游戏二次元" / "鸡汤口号" / "宠物日常" / "广告与推广" / "正文未提躺平" / "其他"
- reason 字符串，<=20字，可为空

若输入缺字段或不可恢复，按最保守方式补齐：
tangping_related_label="无关", exclusion_type="其他", confidence="低", reason=""
"""


@dataclass
class LabelingStats:
    total: int = 0
    success: int = 0
    resumed: int = 0
    fixer_triggered: int = 0
    degraded: int = 0
    parse_fail: int = 0
    schema_fail: int = 0
    labeler_retries: int = 0


def get_first_env(*names: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return ""


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        timeout: int = 120,
        request_retries: int = 0,
        retry_backoff_sec: float = 0.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.request_retries = request_retries
        self.retry_backoff_sec = retry_backoff_sec

    def _post(self, url: str, payload: Dict[str, Any]) -> requests.Response:
        attempts = max(1, self.request_retries + 1)
        last_error: Optional[Exception] = None

        for attempt in range(attempts):
            try:
                response = requests.post(url, json=payload, timeout=self.timeout)
                if response.status_code in {408, 429, 500, 502, 503, 504} and attempt + 1 < attempts:
                    time.sleep(self.retry_backoff_sec * (attempt + 1))
                    continue
                return response
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
                last_error = exc
                if attempt + 1 >= attempts:
                    raise
                time.sleep(self.retry_backoff_sec * (attempt + 1))

        if last_error is not None:
            raise last_error
        raise RuntimeError("request failed without response")

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
    ) -> str:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        r = self._post(url, payload)
        if r.status_code != 200:
            raise RuntimeError(f"chat http {r.status_code}: {r.text[:300]}")
        data = r.json()
        msg = data.get("message") or {}
        content = msg.get("content")
        if not isinstance(content, str):
            raise RuntimeError("chat response missing message.content")
        return content

    def generate(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.0,
    ) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        r = self._post(url, payload)
        if r.status_code != 200:
            raise RuntimeError(f"generate http {r.status_code}: {r.text[:300]}")
        data = r.json()
        content = data.get("response")
        if not isinstance(content, str):
            raise RuntimeError("generate response missing response field")
        return content

    def request_with_fallback(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
    ) -> str:
        chat_err: Optional[Exception] = None
        try:
            return self.chat(model=model, messages=messages, temperature=temperature)
        except Exception as e:
            chat_err = e

        prompt_parts = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            prompt_parts.append(f"[{role}]\n{content}")
        prompt = "\n\n".join(prompt_parts)

        try:
            return self.generate(model=model, prompt=prompt, temperature=temperature)
        except Exception as gen_err:
            raise RuntimeError(
                f"Both chat and generate failed. chat_err={chat_err}; generate_err={gen_err}"
            ) from gen_err


class OpenAICompatibleClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: int = 120,
        json_mode: bool = False,
        request_retries: int = 0,
        retry_backoff_sec: float = 0.0,
    ):
        if not api_key:
            raise ValueError("Missing API key. Use --labeler_api_key/--fixer_api_key or env DASHSCOPE_API_KEY/QWEN_API_KEY.")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.json_mode = json_mode
        self.request_retries = request_retries
        self.retry_backoff_sec = retry_backoff_sec

    def _post(self, url: str, payload: Dict[str, Any], headers: Dict[str, str]) -> requests.Response:
        attempts = max(1, self.request_retries + 1)
        last_error: Optional[Exception] = None

        for attempt in range(attempts):
            try:
                response = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
                if response.status_code in {408, 429, 500, 502, 503, 504} and attempt + 1 < attempts:
                    time.sleep(self.retry_backoff_sec * (attempt + 1))
                    continue
                return response
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
                last_error = exc
                if attempt + 1 >= attempts:
                    raise
                time.sleep(self.retry_backoff_sec * (attempt + 1))

        if last_error is not None:
            raise last_error
        raise RuntimeError("request failed without response")

    @staticmethod
    def _extract_content(message: Any) -> str:
        if isinstance(message, str):
            return message
        if not isinstance(message, dict):
            raise RuntimeError("chat response missing message content")

        content = message.get("content")
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            if parts:
                return "".join(parts)

        reasoning = message.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning.strip():
            return reasoning

        raise RuntimeError("chat response missing message.content")

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
    ) -> str:
        url = f"{self.base_url}/chat/completions"
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        r = self._post(url, payload, headers)
        if r.status_code != 200:
            raise RuntimeError(f"chat http {r.status_code}: {r.text[:300]}")

        data = r.json()
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("chat response missing choices")
        message = choices[0].get("message")
        return self._extract_content(message)

    def request_with_fallback(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
    ) -> str:
        return self.chat(model=model, messages=messages, temperature=temperature)


def build_client(
    provider: str,
    base_url: str,
    api_key: str,
    timeout: int,
    json_mode: bool = False,
    request_retries: int = 0,
    retry_backoff_sec: float = 0.0,
) -> Any:
    if provider == "ollama":
        return OllamaClient(
            base_url=base_url,
            timeout=timeout,
            request_retries=request_retries,
            retry_backoff_sec=retry_backoff_sec,
        )
    if provider in {"qwen_openai", "openai_compatible"}:
        return OpenAICompatibleClient(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            json_mode=json_mode,
            request_retries=request_retries,
            retry_backoff_sec=retry_backoff_sec,
        )
    raise ValueError(f"Unsupported provider: {provider}")


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    s = str(value).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def load_config_file(path: Optional[str], required: bool = False) -> Dict[str, Any]:
    if not path:
        return {}
    if not os.path.exists(path):
        if required:
            raise FileNotFoundError(f"Config file not found: {path}")
        return {}

    with open(path, "rb") as f:
        raw = tomllib.load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Config file must contain a TOML table: {path}")

    config: Dict[str, Any] = {}
    for key, value in raw.items():
        if key in {"labeler", "fixer"}:
            continue
        config[key] = value

    for prefix in ["labeler", "fixer"]:
        section = raw.get(prefix)
        if section is None:
            continue
        if not isinstance(section, dict):
            raise ValueError(f"[{prefix}] section must be a TOML table: {path}")
        for key, value in section.items():
            config[f"{prefix}_{key}"] = value

    return config


def parse_args() -> argparse.Namespace:
    config_bootstrap = argparse.ArgumentParser(add_help=False)
    config_bootstrap.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    bootstrap_args, _ = config_bootstrap.parse_known_args()
    config_required = any(arg == "--config" or arg.startswith("--config=") for arg in sys.argv[1:])
    config_defaults = load_config_file(bootstrap_args.config, required=config_required)

    parser = argparse.ArgumentParser(
        description=(
            "对 sample.csv 做 LLM 预标注，输出 labeled.csv 草稿。"
            "结果只用于人工审核前的初筛，不应直接当作最终训练标签。"
        )
    )
    parser.add_argument("--config", default=bootstrap_args.config, help="可选配置文件路径。")
    parser.add_argument("--input", default="bert/data/sample.csv", help="输入抽样 CSV；默认 bert/data/sample.csv。")
    parser.add_argument("--output", default="bert/data/labeled.csv", help="输出预标注 CSV；默认 bert/data/labeled.csv。")
    parser.add_argument(
        "--report_path",
        default="bert/data/labeling_report.json",
        help="预标注报告 JSON 路径；默认 bert/data/labeling_report.json。",
    )
    parser.add_argument("--text_col", default=None, help="可选，显式指定文本列名。")
    parser.add_argument(
        "--labeler_provider",
        default="qwen_openai",
        choices=["qwen_openai", "openai_compatible", "ollama"],
        help="主标注模型的 provider；默认 qwen_openai。",
    )
    parser.add_argument(
        "--labeler_base_url",
        default=os.getenv("QWEN_BASE_URL", DEFAULT_QWEN_BASE_URL),
        help="主标注模型的 base URL。",
    )
    parser.add_argument(
        "--labeler_api_key",
        default=get_first_env("DASHSCOPE_API_KEY", "QWEN_API_KEY", "OPENAI_API_KEY"),
        help="主标注模型的 API key；未传时尝试从环境变量读取。",
    )
    parser.add_argument(
        "--labeler_model",
        default=os.getenv("QWEN_LABELER_MODEL", "qwen-coder-plus-latest"),
        help="主标注模型名；默认 qwen-coder-plus-latest。",
    )
    parser.add_argument(
        "--fixer_provider",
        default="ollama",
        choices=["ollama", "qwen_openai", "openai_compatible"],
        help="JSON 修复器 provider；默认 ollama。",
    )
    parser.add_argument("--fixer_base_url", default=None, help="JSON 修复器的 base URL。")
    parser.add_argument(
        "--fixer_api_key",
        default=get_first_env("FIXER_API_KEY", "DASHSCOPE_API_KEY", "QWEN_API_KEY", "OPENAI_API_KEY"),
        help="JSON 修复器 API key；未传时尝试从环境变量读取。",
    )
    parser.add_argument("--base_url", default="http://localhost:11434", help="兼容旧参数名，等同于 fixer 的本地 base URL。")
    parser.add_argument("--fixer_model", default="qwen3:8b", help="JSON 修复器模型名；默认 qwen3:8b。")
    parser.add_argument("--max_chars", type=int, default=800, help="单条文本送入模型前的最大字符数。")
    parser.add_argument("--temperature", type=float, default=0.0, help="采样温度；默认 0。")
    parser.add_argument("--timeout", type=int, default=120, help="单次请求超时秒数；默认 120。")
    parser.add_argument("--fix_json", type=str2bool, default=True, help="是否启用 JSON 修复步骤；默认 true。")
    parser.add_argument("--save_raw", type=str2bool, default=True, help="是否保存主标注模型原始输出；默认 true。")
    parser.add_argument(
        "--save_fixed_raw",
        type=str2bool,
        default=True,
        help="是否保存修复器原始输出；默认 true。",
    )
    parser.add_argument("--save_every", type=int, default=100, help="每处理多少条落盘一次中间结果；默认 100。")
    parser.add_argument("--max_workers", type=int, default=2, help="并发 worker 数；默认 2。")
    parser.add_argument("--request_retries", type=int, default=2, help="请求失败后的重试次数；默认 2。")
    parser.add_argument("--retry_backoff_sec", type=float, default=3.0, help="重试退避秒数；默认 3。")
    parser.set_defaults(**config_defaults)
    return parser.parse_args()


def load_dataframe(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")
    lower = path.lower()
    if lower.endswith(".parquet"):
        return pd.read_parquet(path)
    if lower.endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError("Unsupported input format. Use .parquet or .csv")


def save_dataframe(df: pd.DataFrame, path: str) -> None:
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)
    suffix = os.path.splitext(path)[1] or ".tmp"
    with tempfile.NamedTemporaryFile(dir=parent, suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name

    lower = path.lower()
    try:
        if lower.endswith(".parquet"):
            df.to_parquet(tmp_path, index=False)
        elif lower.endswith(".csv"):
            df.to_csv(tmp_path, index=False, encoding="utf-8")
        else:
            raise ValueError("Unsupported output format. Use .parquet or .csv")
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def detect_text_col(df: pd.DataFrame, forced_col: Optional[str]) -> str:
    if forced_col:
        if forced_col not in df.columns:
            raise ValueError(f"--text_col not found: {forced_col}")
        return forced_col

    for c in ["cleaned_text", "cleaned_text_with_emoji", "text_raw", "微博正文", "raw_text"]:
        if c in df.columns:
            return c
    raise ValueError("No text column found. Expected cleaned_text/cleaned_text_with_emoji/text_raw/微博正文 or provide --text_col")


def build_labeler_messages(text: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": LABELER_SYSTEM_PROMPT},
        {"role": "user", "content": f"请判断以下文本是否属于躺平相关表达，并输出严格JSON：\n{text}"},
    ]


def build_fixer_messages(raw_out: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": FIXER_SYSTEM_PROMPT},
        {"role": "user", "content": raw_out},
    ]


def parse_json_strict(raw: str) -> Dict[str, Any]:
    if not isinstance(raw, str):
        raise ValueError("raw output is not string")
    text = raw.strip()
    if not text:
        raise ValueError("empty output")
    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise ValueError("json output is not an object")
    return obj


def try_extract_json(raw: str) -> str:
    if not isinstance(raw, str):
        return ""
    s = raw.strip()
    if not s:
        return ""

    s = re.sub(r"```(?:json)?", "", s, flags=re.IGNORECASE)
    s = s.replace("```", "")
    s = s.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")

    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        s = s[start : end + 1]

    s = re.sub(r",\s*([}\]])", r"\1", s)

    try:
        json.loads(s)
        return s
    except Exception:
        pass

    s2 = re.sub(r"(?<!\\)'", '"', s)
    s2 = re.sub(r",\s*([}\]])", r"\1", s2)
    return s2


def normalize_label(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip().lower().replace(" ", "")
    mapping = {
        "相关": "相关",
        "有关": "相关",
        "有关系": "相关",
        "是": "相关",
        "yes": "相关",
        "true": "相关",
        "1": "相关",
        "无关": "无关",
        "不相关": "无关",
        "没关系": "无关",
        "否": "无关",
        "no": "无关",
        "false": "无关",
        "0": "无关",
    }
    if s in mapping:
        return mapping[s]
    return None


def normalize_confidence(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip().lower().replace(" ", "")
    mapping = {
        "高": "高",
        "高置信": "高",
        "high": "高",
        "h": "高",
        "中": "中",
        "中等": "中",
        "中置信": "中",
        "medium": "中",
        "med": "中",
        "m": "中",
        "低": "低",
        "低置信": "低",
        "low": "低",
        "l": "低",
    }
    return mapping.get(s)


def normalize_exclusion_type(value: Any) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    if not s:
        return ""
    direct_mapping = {
        "无": "无",
        "字面提及躺平": "字面提及躺平",
        "字面提及": "字面提及躺平",
        "躺下休息": "字面提及躺平",
        "动作描写": "字面提及躺平",
        "明星粉圈": "明星粉圈",
        "粉圈明星": "明星粉圈",
        "粉圈": "明星粉圈",
        "饭圈": "明星粉圈",
        "游戏二次元": "游戏二次元",
        "小说游戏二次元": "游戏二次元",
        "游戏": "游戏二次元",
        "二次元": "游戏二次元",
        "鸡汤口号": "鸡汤口号",
        "口号": "鸡汤口号",
        "表情包": "鸡汤口号",
        "宠物日常": "宠物日常",
        "宠物": "宠物日常",
        "广告与推广": "广告与推广",
        "广告推广": "广告与推广",
        "营销推广": "广告与推广",
        "广告": "广告与推广",
        "正文未提躺平": "正文未提躺平",
        "正文未提": "正文未提躺平",
        "仅关键词命中": "正文未提躺平",
        "仅话题命中": "正文未提躺平",
        "其他": "其他",
    }
    if s in direct_mapping:
        return direct_mapping[s]

    compact = s.replace(" ", "")
    fuzzy_pairs = [
        ("字面", "字面提及躺平"),
        ("躺下", "字面提及躺平"),
        ("动作", "字面提及躺平"),
        ("粉圈", "明星粉圈"),
        ("饭圈", "明星粉圈"),
        ("明星", "明星粉圈"),
        ("游戏", "游戏二次元"),
        ("二次元", "游戏二次元"),
        ("小说", "游戏二次元"),
        ("口号", "鸡汤口号"),
        ("鸡汤", "鸡汤口号"),
        ("表情包", "鸡汤口号"),
        ("宠物", "宠物日常"),
        ("广告", "广告与推广"),
        ("推广", "广告与推广"),
        ("营销", "广告与推广"),
        ("正文未提", "正文未提躺平"),
        ("仅话题", "正文未提躺平"),
        ("仅关键词", "正文未提躺平"),
    ]
    for needle, normalized in fuzzy_pairs:
        if needle in compact:
            return normalized
    return s


def is_nonempty_value(value: Any) -> bool:
    if value is None:
        return False
    if pd.isna(value):
        return False
    return str(value).strip() != ""


RESUME_OUTPUT_COLUMNS = {
    "tangping_related_label",
    "tangping_related",
    "exclusion_type",
    "confidence",
    "llm_reason",
    "llm_raw",
    "llm_fixed_raw",
}


def _normalize_resume_identity_value(value: Any) -> str:
    if isinstance(value, (list, tuple, set)):
        return "|".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    if value is None or pd.isna(value):
        return ""
    return str(value)


def _resolve_resume_identity_columns(input_df: pd.DataFrame, resume_df: pd.DataFrame) -> List[str]:
    return [
        column
        for column in input_df.columns
        if column in resume_df.columns and column not in RESUME_OUTPUT_COLUMNS
    ]


def _build_resume_row_signatures(df: pd.DataFrame, columns: List[str]) -> pd.Series:
    normalized = df.loc[:, columns].map(_normalize_resume_identity_value)
    return pd.util.hash_pandas_object(normalized, index=False)


def parse_resume_result(row: pd.Series, save_raw: bool, save_fixed_raw: bool) -> Optional[Dict[str, Any]]:
    label = normalize_label(row.get("tangping_related_label"))
    conf = normalize_confidence(row.get("confidence"))
    if label is None or conf is None:
        return None

    raw = row.get("llm_raw", "")
    fixed_raw = row.get("llm_fixed_raw", "")
    reason = truncate_reason(row.get("llm_reason", ""), 20)
    has_resume_marker = any(
        is_nonempty_value(value)
        for value in (reason, raw, fixed_raw)
    )
    if not has_resume_marker:
        return None

    tangping_related = label_to_binary(label)
    existing_binary = row.get("tangping_related")
    if is_nonempty_value(existing_binary):
        try:
            existing_binary_int = int(existing_binary)
        except (TypeError, ValueError):
            return None
        if existing_binary_int != tangping_related:
            return None

    exclusion = normalize_exclusion_type(row.get("exclusion_type", ""))
    ok, norm, _, _ = validate_and_normalize(
        {
            "tangping_related_label": label,
            "exclusion_type": exclusion,
            "confidence": conf,
            "reason": reason,
        }
    )
    if not ok:
        return None

    return {
        "tangping_related_label": norm["tangping_related_label"],
        "tangping_related": tangping_related,
        "exclusion_type": norm["exclusion_type"],
        "confidence": norm["confidence"],
        "llm_reason": norm["reason"],
        "llm_raw": "" if not save_raw or pd.isna(raw) else str(raw),
        "llm_fixed_raw": "" if not save_fixed_raw or pd.isna(fixed_raw) else str(fixed_raw),
    }


def truncate_reason(reason: Any, max_len: int = 20) -> str:
    if reason is None:
        return ""
    s = str(reason).strip()
    if len(s) <= max_len:
        return s
    return s[:max_len]


def validate_and_normalize(obj: Dict[str, Any]) -> Tuple[bool, Dict[str, str], bool, str]:
    if not isinstance(obj, dict):
        return False, {}, True, "not_object"

    label = normalize_label(obj.get("tangping_related_label"))
    conf = normalize_confidence(obj.get("confidence"))
    exclusion = normalize_exclusion_type(obj.get("exclusion_type", ""))
    reason = truncate_reason(obj.get("reason", ""), 20)

    if label is None:
        return False, {}, True, "invalid_label"
    if conf is None:
        return False, {}, True, "invalid_confidence"

    schema_fail = False

    if label == "相关":
        exclusion = "无"
    else:
        if exclusion not in IRRELEVANT_TYPES:
            exclusion = "其他"
            schema_fail = True

    norm = {
        "tangping_related_label": label,
        "exclusion_type": exclusion,
        "confidence": conf,
        "reason": reason,
    }

    if schema_fail:
        return False, norm, True, "invalid_exclusion_for_irrelevant"
    return True, norm, False, ""


def shorten_text(text: Any, max_chars: int) -> str:
    s = "" if text is None else str(text)
    if max_chars <= 0:
        return s
    if len(s) <= max_chars:
        return s
    head = max_chars // 2
    tail = max_chars - head
    return s[:head] + " … " + s[-tail:]


def row_uid(row: pd.Series) -> str:
    for c in ["id", "mid"]:
        if c in row.index:
            v = row[c]
            if pd.notna(v):
                return f"{c}={v}"
    return "uid=NA"


def default_result(save_raw: bool, save_fixed_raw: bool, raw: str = "", fixed_raw: str = "") -> Dict[str, Any]:
    return {
        "tangping_related_label": "无关",
        "tangping_related": 0,
        "exclusion_type": "其他",
        "confidence": "低",
        "llm_reason": "",
        "llm_raw": raw if save_raw else "",
        "llm_fixed_raw": fixed_raw if save_fixed_raw else "",
    }


def label_to_binary(label: str) -> int:
    return 1 if label == "相关" else 0


def label_one_row(
    row: pd.Series,
    text_col: str,
    labeler_client: Any,
    fixer_client: Any,
    labeler_model: str,
    fixer_model: str,
    max_chars: int,
    temperature: float,
    fix_json: bool,
    save_raw: bool,
    save_fixed_raw: bool,
    logger: logging.Logger,
) -> Dict[str, Any]:
    text = shorten_text(row.get(text_col, ""), max_chars=max_chars)
    last_raw = ""
    last_fixed = ""
    used_fixer = 0

    for attempt in range(2):
        try:
            messages = build_labeler_messages(text)
            raw_out = labeler_client.request_with_fallback(
                model=labeler_model,
                messages=messages,
                temperature=temperature,
            )
            last_raw = raw_out

            parsed_obj = None
            parse_ok = False

            try:
                parsed_obj = parse_json_strict(raw_out)
                parse_ok = True
            except Exception:
                extracted = try_extract_json(raw_out)
                if extracted:
                    try:
                        parsed_obj = parse_json_strict(extracted)
                        parse_ok = True
                    except Exception:
                        parse_ok = False

            if parse_ok and parsed_obj is not None:
                ok, norm, schema_fail, _ = validate_and_normalize(parsed_obj)
                if ok:
                    return {
                        "result": {
                            "tangping_related_label": norm["tangping_related_label"],
                            "tangping_related": label_to_binary(norm["tangping_related_label"]),
                            "exclusion_type": norm["exclusion_type"],
                            "confidence": norm["confidence"],
                            "llm_reason": norm["reason"],
                            "llm_raw": raw_out if save_raw else "",
                            "llm_fixed_raw": "",
                        },
                        "used_fixer": used_fixer,
                        "degraded": 0,
                        "parse_fail": 0,
                        "schema_fail": int(schema_fail),
                        "retry_used": attempt,
                    }

            if fix_json:
                used_fixer += 1
                fixer_messages = build_fixer_messages(raw_out)
                fixed_out = fixer_client.request_with_fallback(
                    model=fixer_model,
                    messages=fixer_messages,
                    temperature=temperature,
                )
                last_fixed = fixed_out

                parsed_fixed = None
                fixed_ok = False
                try:
                    parsed_fixed = parse_json_strict(fixed_out)
                    fixed_ok = True
                except Exception:
                    extracted_fixed = try_extract_json(fixed_out)
                    if extracted_fixed:
                        try:
                            parsed_fixed = parse_json_strict(extracted_fixed)
                            fixed_ok = True
                        except Exception:
                            fixed_ok = False

                if fixed_ok and parsed_fixed is not None:
                    ok2, norm2, _, _ = validate_and_normalize(parsed_fixed)
                    if ok2:
                        return {
                            "result": {
                                "tangping_related_label": norm2["tangping_related_label"],
                                "tangping_related": label_to_binary(norm2["tangping_related_label"]),
                                "exclusion_type": norm2["exclusion_type"],
                                "confidence": norm2["confidence"],
                                "llm_reason": norm2["reason"],
                                "llm_raw": raw_out if save_raw else "",
                                "llm_fixed_raw": fixed_out if save_fixed_raw else "",
                            },
                            "used_fixer": used_fixer,
                            "degraded": 0,
                            "parse_fail": 0,
                            "schema_fail": 0,
                            "retry_used": attempt,
                        }

            if attempt == 0:
                continue

        except Exception as e:
            logger.exception("row_label_exception uid=%s err=%s", row_uid(row), str(e))
            if attempt == 0:
                continue

    return {
        "result": default_result(
            save_raw=save_raw,
            save_fixed_raw=save_fixed_raw,
            raw=last_raw,
            fixed_raw=last_fixed,
        ),
        "used_fixer": used_fixer,
        "degraded": 1,
        "parse_fail": 1,
        "schema_fail": 1,
        "retry_used": 1,
    }


def setup_logger() -> logging.Logger:
    os.makedirs("logs", exist_ok=True)
    logger = logging.getLogger("tangping_label")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    fh = logging.FileHandler("logs/llm_label.log", encoding="utf-8")
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


def write_report(report: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def build_output_dataframe(
    df: pd.DataFrame,
    results: List[Optional[Dict[str, Any]]],
    save_raw: bool,
    save_fixed_raw: bool,
) -> pd.DataFrame:
    out_df = df.copy()
    out_df["tangping_related_label"] = ""
    out_df["tangping_related"] = pd.Series([pd.NA] * len(out_df), index=out_df.index, dtype="Int64")
    out_df["exclusion_type"] = ""
    out_df["confidence"] = ""
    out_df["llm_reason"] = ""
    out_df["llm_raw"] = ""
    out_df["llm_fixed_raw"] = ""

    for i, res in enumerate(results):
        if res is None:
            continue
        row_idx = out_df.index[i]
        out_df.at[row_idx, "tangping_related_label"] = res["tangping_related_label"]
        out_df.at[row_idx, "tangping_related"] = int(res["tangping_related"])
        out_df.at[row_idx, "exclusion_type"] = res["exclusion_type"]
        out_df.at[row_idx, "confidence"] = res["confidence"]
        out_df.at[row_idx, "llm_reason"] = res["llm_reason"]
        if save_raw:
            out_df.at[row_idx, "llm_raw"] = res["llm_raw"]
        if save_fixed_raw:
            out_df.at[row_idx, "llm_fixed_raw"] = res["llm_fixed_raw"]
    return out_df


def load_resume_results(
    df: pd.DataFrame,
    output_path: str,
    save_raw: bool,
    save_fixed_raw: bool,
    logger: logging.Logger,
) -> List[Optional[Dict[str, Any]]]:
    results: List[Optional[Dict[str, Any]]] = [None] * len(df)
    if not output_path or not os.path.exists(output_path):
        logger.info("stage=resume_scan output=%s found=0 usable=0 reason=missing_output", output_path)
        return results

    try:
        resume_df = load_dataframe(output_path)
    except Exception as exc:
        logger.warning("stage=resume_scan output=%s found=1 usable=0 reason=load_failed err=%s", output_path, exc)
        return results

    if len(resume_df) != len(df):
        logger.warning(
            "stage=resume_scan output=%s found=1 usable=0 reason=row_count_mismatch input_rows=%d output_rows=%d",
            output_path,
            len(df),
            len(resume_df),
        )
        return results

    identity_columns = _resolve_resume_identity_columns(df, resume_df)
    if not identity_columns:
        logger.warning("stage=resume_scan output=%s found=1 usable=0 reason=no_identity_columns", output_path)
        return results
    input_signatures = _build_resume_row_signatures(df, identity_columns)
    resume_signatures = _build_resume_row_signatures(resume_df, identity_columns)
    if not input_signatures.equals(resume_signatures):
        logger.warning(
            "stage=resume_scan output=%s found=1 usable=0 reason=row_signature_mismatch identity_cols=%s",
            output_path,
            ",".join(identity_columns),
        )
        return results

    resumed = 0
    for i, row in resume_df.iterrows():
        parsed = parse_resume_result(row, save_raw=save_raw, save_fixed_raw=save_fixed_raw)
        if parsed is None:
            continue
        results[i] = parsed
        resumed += 1

    logger.info("stage=resume_scan output=%s found=1 usable=1 resumed_rows=%d", output_path, resumed)
    return results


def maybe_save_checkpoint(
    df: pd.DataFrame,
    results: List[Optional[Dict[str, Any]]],
    output_path: str,
    completed: int,
    total: int,
    save_every: int,
    save_raw: bool,
    save_fixed_raw: bool,
    logger: logging.Logger,
) -> None:
    if save_every <= 0 or completed % save_every != 0 or completed == total:
        return
    checkpoint_df = build_output_dataframe(
        df=df,
        results=results,
        save_raw=save_raw,
        save_fixed_raw=save_fixed_raw,
    )
    logger.info("stage=checkpoint_save completed=%d total=%d output=%s", completed, total, output_path)
    save_dataframe(checkpoint_df, output_path)


def main() -> None:
    args = parse_args()
    logger = setup_logger()

    start_ts = time.time()
    logger.info("stage=load_input input=%s output=%s report=%s", args.input, args.output, args.report_path)
    logger.info("stage=manual_review note=%s", "LLM output is draft-only and must be manually reviewed before training.")

    df = load_dataframe(args.input)
    total = len(df)
    if total == 0:
        raise ValueError("Input dataframe is empty")

    text_col = detect_text_col(df, args.text_col)
    logger.info("stage=detect_text_col text_col=%s total_rows=%d", text_col, total)
    logger.info("stage=checkpoint_config save_every=%d", args.save_every)

    results = load_resume_results(
        df=df,
        output_path=args.output,
        save_raw=args.save_raw,
        save_fixed_raw=args.save_fixed_raw,
        logger=logger,
    )

    fixer_base_url = args.fixer_base_url or args.base_url
    labeler_client = build_client(
        provider=args.labeler_provider,
        base_url=args.labeler_base_url,
        api_key=args.labeler_api_key,
        timeout=args.timeout,
        json_mode=True,
        request_retries=args.request_retries,
        retry_backoff_sec=args.retry_backoff_sec,
    )
    fixer_client = build_client(
        provider=args.fixer_provider,
        base_url=fixer_base_url,
        api_key=args.fixer_api_key,
        timeout=args.timeout,
        json_mode=True,
        request_retries=args.request_retries,
        retry_backoff_sec=args.retry_backoff_sec,
    )
    stats = LabelingStats(total=total)
    stats.resumed = sum(1 for r in results if r is not None)
    stats_lock = threading.Lock()

    def process_one(i: int, row: pd.Series) -> Tuple[int, Dict[str, Any]]:
        out = label_one_row(
            row=row,
            text_col=text_col,
            labeler_client=labeler_client,
            fixer_client=fixer_client,
            labeler_model=args.labeler_model,
            fixer_model=args.fixer_model,
            max_chars=args.max_chars,
            temperature=args.temperature,
            fix_json=args.fix_json,
            save_raw=args.save_raw,
            save_fixed_raw=args.save_fixed_raw,
            logger=logger,
        )
        return i, out

    max_workers = max(1, min(args.max_workers, (os.cpu_count() or 4)))
    pending_indices = [i for i, res in enumerate(results) if res is None]
    logger.info(
        "stage=labeling_start max_workers=%d labeler_provider=%s labeler_model=%s fixer_provider=%s fixer_model=%s resumed=%d pending=%d",
        max_workers,
        args.labeler_provider,
        args.labeler_model,
        args.fixer_provider,
        args.fixer_model,
        stats.resumed,
        len(pending_indices),
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(process_one, i, df.iloc[i]) for i in pending_indices]

        if tqdm is not None:
            iterator = tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="TangpingLabeling")
        else:
            iterator = concurrent.futures.as_completed(futures)

        completed = stats.resumed
        progress_every = max(1, total // 20)
        for fut in iterator:
            i, out = fut.result()
            results[i] = out["result"]
            with stats_lock:
                stats.success += 1
                stats.fixer_triggered += int(out["used_fixer"] > 0)
                stats.degraded += int(out["degraded"])
                stats.parse_fail += int(out["parse_fail"])
                stats.schema_fail += int(out["schema_fail"])
                stats.labeler_retries += int(out["retry_used"])
            completed += 1
            maybe_save_checkpoint(
                df=df,
                results=results,
                output_path=args.output,
                completed=completed,
                total=total,
                save_every=args.save_every,
                save_raw=args.save_raw,
                save_fixed_raw=args.save_fixed_raw,
                logger=logger,
            )
            if tqdm is None and (completed == total or completed % progress_every == 0):
                logger.info("stage=labeling_progress completed=%d total=%d", completed, total)

    if any(r is None for r in results):
        raise RuntimeError("Some labeling results are missing.")

    out_df = build_output_dataframe(
        df=df,
        results=results,
        save_raw=args.save_raw,
        save_fixed_raw=args.save_fixed_raw,
    )

    logger.info("stage=write_output output=%s", args.output)
    save_dataframe(out_df, args.output)

    elapsed = time.time() - start_ts
    report = {
        "task": "tangping_labeling",
        "input": args.input,
        "output": args.output,
        "report_path": args.report_path,
        "rows": total,
        "text_col": text_col,
        "labeler_provider": args.labeler_provider,
        "labeler_model": args.labeler_model,
        "labeler_base_url": args.labeler_base_url,
        "fixer_provider": args.fixer_provider,
        "fixer_model": args.fixer_model,
        "fixer_base_url": fixer_base_url,
        "manual_review_required": True,
        "save_every": args.save_every,
        "elapsed_sec": round(elapsed, 3),
        "stats": {
            "total": stats.total,
            "resumed": stats.resumed,
            "success": stats.success,
            "fixer_triggered": stats.fixer_triggered,
            "degraded": stats.degraded,
            "parse_fail": stats.parse_fail,
            "schema_fail": stats.schema_fail,
            "labeler_retries": stats.labeler_retries,
        },
        "label_distribution": out_df["tangping_related_label"].astype(str).value_counts(dropna=False).to_dict(),
        "binary_distribution": out_df["tangping_related"].astype(str).value_counts(dropna=False).to_dict(),
        "exclusion_distribution": out_df["exclusion_type"].astype(str).value_counts(dropna=False).to_dict(),
    }
    logger.info("stage=write_report report=%s", args.report_path)
    write_report(report, args.report_path)

    logger.info("stage=done rows=%d elapsed=%.3fs output=%s", total, elapsed, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

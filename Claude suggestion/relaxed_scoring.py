"""
relaxed_scoring.py — FinBridge Relaxed Match 评分模块

用法：
    集成到你现有的评分流程中，对每条预测同时计算 strict 和 relaxed 两套分数。

示例：
    from relaxed_scoring import strict_score, relaxed_score

    text = "Answer: 100 * (5.99 - 3.24) / 3.24 = 86.47%"
    gold = 86.47

    s_em, s_tm, s_val = strict_score(text, gold)   # → 0, 0, None
    r_em, r_tm, r_val = relaxed_score(text, gold)   # → 1, 1, 86.47
"""

import re

# ─────────────────────────────────────────────
# 公共工具
# ─────────────────────────────────────────────

def _find_answer_line(text: str):
    """从模型输出中找最后一行以 'Answer:' 开头的行，返回 Answer: 后面的内容"""
    for line in reversed(text.strip().split('\n')):
        if line.strip().lower().startswith('answer:'):
            return line.strip().split(':', 1)[1].strip()
    return None


def _tolerance_check(pred: float, gold: float, tol: float = 0.005) -> int:
    """容差匹配：相对误差 <= tol 则返回 1"""
    if gold == 0:
        return 1 if abs(pred) <= tol else 0
    return 1 if abs(pred - gold) / abs(gold) <= tol else 0


def _clean_numeric(raw: str):
    """清洗字符串，尝试转为 float。成功返回 float，失败返回 None"""
    cleaned = raw.replace(',', '').replace('$', '').replace('%', '').strip()
    # 处理括号负号: (123.4) → -123.4
    if cleaned.startswith('(') and cleaned.endswith(')'):
        cleaned = '-' + cleaned[1:-1]
    try:
        return float(cleaned)
    except ValueError:
        return None


# ─────────────────────────────────────────────
# Strict Match（你现有的 parser v1 逻辑）
# ─────────────────────────────────────────────

def strict_score(prediction_text: str, gold_numeric: float, tol: float = 0.005):
    """
    Strict match：Answer: 行必须是纯数值才算 parse 成功。
    
    Returns:
        (em, tm, parsed_value)
        em: 1 if exact match, 0 otherwise
        tm: 1 if tolerance match, 0 otherwise
        parsed_value: float or None
    """
    answer_content = _find_answer_line(prediction_text)
    if answer_content is None:
        return 0, 0, None

    pred = _clean_numeric(answer_content)
    if pred is None:
        return 0, 0, None  # Answer 行内容不是纯数值

    em = 1 if pred == gold_numeric else 0
    tm = _tolerance_check(pred, gold_numeric, tol)
    return em, tm, pred


# ─────────────────────────────────────────────
# Relaxed Match（宽松提取）
# ─────────────────────────────────────────────

def relaxed_extract(text: str):
    """
    从模型输出中宽松提取最终数值答案。
    
    提取策略（按优先级）：
    1. 有 Answer 行 → 从 Answer 行提取
    2. Answer 行里有 '=' → 取最后一个 '=' 后面的数值
    3. 否则取目标文本中最后一个数值
    4. 没有 Answer 行 → 从整个输出的最后一个数值提取
    """
    answer_content = _find_answer_line(text)
    target = answer_content if answer_content else text

    # 如果有等号，只看最后一个等号后面
    if '=' in target:
        target = target.split('=')[-1]

    # 清洗常见符号
    target = target.replace(',', '').replace('$', '')

    # 提取所有数值模式（含负号、小数点、百分号）
    numbers = re.findall(r'-?\d+\.?\d*%?', target)
    if not numbers:
        return None

    # 取最后一个
    raw = numbers[-1].replace('%', '').strip()
    try:
        val = float(raw)
    except ValueError:
        return None

    # 检查原始 target 中该数值是否被括号包围 → 负号
    # 例如 "(16.34)" → -16.34
    bracket_pattern = r'\(' + re.escape(numbers[-1].replace('%', '')) + r'%?\)'
    if re.search(bracket_pattern, answer_content or text):
        val = -abs(val)

    return val


def relaxed_score(prediction_text: str, gold_numeric: float, tol: float = 0.005):
    """
    Relaxed match：宽松提取数值，然后做容差匹配。
    
    Returns:
        (em, tm, parsed_value)
    """
    pred = relaxed_extract(prediction_text)
    if pred is None:
        return 0, 0, None

    em = 1 if pred == gold_numeric else 0
    tm = _tolerance_check(pred, gold_numeric, tol)
    return em, tm, pred


# ─────────────────────────────────────────────
# 批量评分（集成到你的 validation 883 流程）
# ─────────────────────────────────────────────

def score_prediction(prediction_text: str, gold_numeric: float, tol: float = 0.005):
    """
    对单条预测同时计算 strict 和 relaxed 两套分数。
    
    Returns:
        dict with keys:
            strict_em, strict_tm, strict_parsed,
            relaxed_em, relaxed_tm, relaxed_parsed,
            format_gap (= relaxed_tm - strict_tm)
    """
    s_em, s_tm, s_val = strict_score(prediction_text, gold_numeric, tol)
    r_em, r_tm, r_val = relaxed_score(prediction_text, gold_numeric, tol)

    return {
        'strict_em': s_em,
        'strict_tm': s_tm,
        'strict_parsed': s_val,
        'relaxed_em': r_em,
        'relaxed_tm': r_tm,
        'relaxed_parsed': r_val,
        'format_gap': r_tm - s_tm,  # 被格式问题吃掉的准确率
    }


# ─────────────────────────────────────────────
# 测试
# ─────────────────────────────────────────────

if __name__ == '__main__':
    test_cases = [
        # (模型输出, gold, 期望 strict_tm, 期望 relaxed_tm, 说明)
        ("Answer: 86.47", 86.47, 1, 1, "Qwen 风格：干净数值"),
        ("Answer: 86.47%", 86.47, 1, 1, "带百分号"),
        ("Answer: 100 * (5.99 - 3.24) / 3.24 = 86.47%", 86.47, 0, 1, "Mistral 风格：带算式"),
        ("Answer: -16.34%", -16.34, 1, 1, "负数百分比"),
        ("Answer: (16.34)", -16.34, 1, 1, "括号负号"),
        ("To calculate the growth rate...\nThe answer is 86.47", 86.47, 0, 1, "没有 Answer: 前缀，relaxed 从全文提取"),
        ("Answer: approximately 86.5", 86.47, 0, 1, "strict 失败(有文字)，relaxed 提取 86.5 并容差匹配"),
        ("I think the answer is\nAnswer: 17.8", 17.8, 1, 1, "多行，Answer 在最后"),
    ]

    print("=" * 80)
    print("Relaxed Scoring 测试")
    print("=" * 80)

    for text, gold, exp_strict, exp_relaxed, desc in test_cases:
        result = score_prediction(text, gold)
        status_s = "✅" if result['strict_tm'] == exp_strict else "❌"
        status_r = "✅" if result['relaxed_tm'] == exp_relaxed else "❌"
        print(f"\n{desc}")
        print(f"  输入: {text!r}")
        print(f"  Gold: {gold}")
        print(f"  Strict  TM={result['strict_tm']} (parsed={result['strict_parsed']}) {status_s}")
        print(f"  Relaxed TM={result['relaxed_tm']} (parsed={result['relaxed_parsed']}) {status_r}")
        print(f"  Format Gap: {result['format_gap']}")

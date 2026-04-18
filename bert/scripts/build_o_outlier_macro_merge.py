from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


DEFAULT_RUN_DIR = Path(
    "bert/artifacts/broad_analysis/topic_model_compare/O_outlier"
)
DEFAULT_OUTPUT_DIR = Path(
    "bert/artifacts/broad_analysis/topic_model_compare/O_outlier_manual_merge_v1"
)
DEFAULT_EXCLUDED_PERIODS = ("2026-01", "2026-02")


@dataclass(frozen=True)
class Rule:
    macro_topic: str
    report_action: str
    markers: tuple[str, ...]


MANUAL_TOPIC_OVERRIDES: dict[int, tuple[str, str, str]] = {
    0: ("直接躺平/摆烂宣言", "keep", "manual-top"),
    3: ("摆烂式旅游/出行休闲", "keep", "manual-top"),
    4: ("婚恋与亲密关系", "keep", "manual-top"),
    5: ("平台模板/噪声混合主题", "omit", "manual-noise"),
    6: ("焦虑、心累、卷不动、压力过载", "keep", "manual-top"),
    8: ("学习厌倦、考试压力、考研上岸", "keep", "manual-top"),
    9: ("直接躺平/摆烂宣言", "keep", "manual-top"),
    10: ("周末、假期、节律性休息", "keep", "manual-top"),
    11: ("平台模板/噪声混合主题", "omit", "manual-noise"),
    12: ("学习厌倦、考试压力、考研上岸", "keep", "manual-top"),
    13: ("焦虑、心累、卷不动、压力过载", "keep", "manual-top"),
    14: ("育儿、母职、妈妈角色", "keep", "manual-top"),
    15: ("游戏/二游/电竞圈摆烂", "weak", "manual-top"),
    16: ("在家躺平/回家/床上腐烂/只想睡觉", "keep", "manual-top"),
    17: ("慢生活、天气、晒太阳、轻日常", "keep", "manual-top"),
    18: ("直接躺平/摆烂宣言", "keep", "manual-top"),
    19: ("周末、假期、节律性休息", "keep", "manual-top"),
    20: ("平台模板/噪声混合主题", "omit", "manual-noise"),
    21: ("慢生活、天气、晒太阳、轻日常", "keep", "manual-top"),
    22: ("周末、假期、节律性休息", "keep", "manual-top"),
    23: ("工作厌倦/不想上班/不想干活", "keep", "manual-top"),
    24: ("文娱作品评论", "weak", "manual-top"),
    25: ("游戏/二游/电竞圈摆烂", "weak", "manual-top"),
    26: ("星座/命理/日运与人格标签", "weak", "manual-top"),
    27: ("直接躺平/摆烂宣言", "keep", "manual-top"),
    28: ("工作厌倦/不想上班/不想干活", "keep", "manual-top"),
    29: ("星座/命理/日运与人格标签", "weak", "manual-top"),
    30: ("创作记录", "weak", "manual-top"),
    31: ("论文、作业、写作卡壳", "keep", "manual-top"),
    32: ("周末、假期、节律性休息", "keep", "manual-top"),
    33: ("财富自由、暴富、提前退休、何时躺平", "keep", "manual-top"),
    34: ("体育赛事与竞技吐槽", "weak", "manual-top"),
    35: ("周末、假期、节律性休息", "keep", "manual-top"),
    36: ("学习厌倦、考试压力、考研上岸", "keep", "manual-top"),
    38: ("运动、自律、减肥", "keep", "manual-top"),
    39: ("学习厌倦、考试压力、考研上岸", "keep", "manual-top"),
    40: ("直接躺平/摆烂宣言", "keep", "manual-top"),
    41: ("接单/陪玩/代招交易", "weak", "manual-top"),
    42: ("周末、假期、节律性休息", "keep", "manual-top"),
    43: ("直接躺平/摆烂宣言", "keep", "manual-top"),
    44: ("工作厌倦/不想上班/不想干活", "keep", "manual-top"),
    45: ("摸鱼、离职、精神离职、职场摆烂", "keep", "manual-top"),
    49: ("育儿、母职、妈妈角色", "keep", "manual-top"),
    51: ("摆烂生活哲学/摆烂人生", "keep", "manual-top"),
    53: ("财富自由、暴富、提前退休、何时躺平", "keep", "manual-top"),
    54: ("松弛、自洽、顺其自然", "keep", "manual-top"),
    55: ("游戏/二游/电竞圈摆烂", "weak", "manual-top"),
    57: ("焦虑、心累、卷不动、压力过载", "keep", "manual-top"),
    61: ("运动、自律、减肥", "keep", "manual-top"),
    63: ("游戏/二游/电竞圈摆烂", "weak", "manual-top"),
    65: ("饭圈、佛系追星、CP文化", "weak", "manual-top"),
    67: ("在家躺平/回家/床上腐烂/只想睡觉", "keep", "manual-top"),
    68: ("直接躺平/摆烂宣言", "keep", "manual-top"),
    69: ("饭圈、佛系追星、CP文化", "weak", "manual"),
    70: ("在家躺平/回家/床上腐烂/只想睡觉", "keep", "manual-top"),
    71: ("年轻人该不该躺平的公共讨论", "weak", "manual-top"),
    74: ("饭圈、佛系追星、CP文化", "weak", "manual-top"),
    76: ("平台模板/噪声混合主题", "omit", "manual-noise"),
    77: ("焦虑、心累、卷不动、压力过载", "keep", "manual-top"),
    81: ("工作厌倦/不想上班/不想干活", "keep", "manual-top"),
    82: ("饭圈、佛系追星、CP文化", "weak", "manual"),
    85: ("饭圈、佛系追星、CP文化", "weak", "manual-low-purity"),
    86: ("票务/周边/收物交易型追星", "weak", "manual-top"),
    89: ("游戏/二游/电竞圈摆烂", "weak", "manual"),
    96: ("工作厌倦/不想上班/不想干活", "keep", "manual-top"),
    102: ("摆烂生活哲学/摆烂人生", "keep", "manual-top"),
    103: ("创作记录", "weak", "manual-top"),
    106: ("游戏/二游/电竞圈摆烂", "weak", "manual"),
    107: ("焦虑、心累、卷不动、压力过载", "keep", "manual-top"),
    108: ("职场规则、领导关系、办公室博弈", "weak", "manual"),
    110: ("学习厌倦、考试压力、考研上岸", "keep", "manual"),
    113: ("星座/命理/日运与人格标签", "weak", "manual"),
    114: ("饭圈、佛系追星、CP文化", "weak", "manual-top"),
    116: ("在家躺平/回家/床上腐烂/只想睡觉", "keep", "manual-top"),
    119: ("平台模板/噪声混合主题", "omit", "manual-noise"),
    120: ("游戏/二游/电竞圈摆烂", "weak", "manual"),
    122: ("学习厌倦、考试压力、考研上岸", "keep", "manual"),
    126: ("松弛、自洽、顺其自然", "keep", "manual"),
    130: ("文娱作品评论", "weak", "manual"),
    132: ("慢生活、天气、晒太阳、轻日常", "keep", "manual-top"),
    133: ("周末、假期、节律性休息", "keep", "manual-top"),
    136: ("年轻人该不该躺平的公共讨论", "weak", "manual"),
    145: ("年轻人该不该躺平的公共讨论", "weak", "manual"),
    152: ("国际热点/社会时评外溢", "weak", "manual"),
    163: ("松弛、自洽、顺其自然", "keep", "manual"),
    167: ("摆烂式旅游/出行休闲", "weak", "manual"),
    172: ("直接躺平/摆烂宣言", "keep", "manual"),
    174: ("房地产、买房、婚育成本压力", "keep", "manual-top"),
    175: ("摆烂生活哲学/摆烂人生", "keep", "manual"),
    178: ("饭圈、佛系追星、CP文化", "weak", "manual"),
    184: ("运动、自律、减肥", "keep", "manual-top"),
    185: ("校园/宿舍/开学场景", "keep", "manual-top"),
    186: ("平台客服/消费维权", "weak", "manual"),
    192: ("育儿、母职、妈妈角色", "keep", "manual"),
    204: ("在家躺平/回家/床上腐烂/只想睡觉", "keep", "manual-top"),
    205: ("摸鱼、离职、精神离职、职场摆烂", "keep", "manual-top"),
    208: ("平台模板/噪声混合主题", "omit", "manual-noise"),
    210: ("慢生活、天气、晒太阳、轻日常", "keep", "manual"),
    211: ("学习厌倦、考试压力、考研上岸", "keep", "manual-top"),
    212: ("直接躺平/摆烂宣言", "keep", "manual-top"),
    216: ("平台模板/噪声混合主题", "omit", "manual-noise"),
    241: ("年轻人该不该躺平的公共讨论", "weak", "manual"),
    251: ("校园/宿舍/开学场景", "weak", "manual"),
    271: ("摆烂生活哲学/摆烂人生", "keep", "manual"),
    280: ("焦虑、心累、卷不动、压力过载", "keep", "manual-top"),
    285: ("摆烂生活哲学/摆烂人生", "keep", "manual-top"),
    286: ("摆烂生活哲学/摆烂人生", "keep", "manual"),
    303: ("松弛、自洽、顺其自然", "keep", "manual"),
    309: ("运动、自律、减肥", "keep", "manual"),
    320: ("计划、任务、拖延、完成度", "weak", "manual"),
    326: ("计划、任务、拖延、完成度", "weak", "manual-top"),
    329: ("年轻人该不该躺平的公共讨论", "weak", "manual"),
    335: ("学习厌倦、考试压力、考研上岸", "keep", "manual"),
    339: ("职场规则、领导关系、办公室博弈", "weak", "manual"),
    343: ("平台客服/消费维权", "weak", "manual"),
    346: ("学习厌倦、考试压力、考研上岸", "keep", "manual"),
    357: ("周末、假期、节律性休息", "keep", "manual"),
    374: ("体育赛事与竞技吐槽", "weak", "manual"),
    377: ("焦虑、心累、卷不动、压力过载", "keep", "manual"),
    390: ("摸鱼、离职、精神离职、职场摆烂", "keep", "manual"),
    392: ("直接躺平/摆烂宣言", "keep", "manual-top"),
    414: ("焦虑、心累、卷不动、压力过载", "keep", "manual"),
    423: ("计划、任务、拖延、完成度", "weak", "manual"),
    425: ("泛情绪碎片与口号化表达", "weak", "manual"),
    432: ("财富自由、暴富、提前退休、何时躺平", "keep", "manual"),
    456: ("在家躺平/回家/床上腐烂/只想睡觉", "keep", "manual"),
}


RULES: tuple[Rule, ...] = (
    Rule(
        "股票、基金、A股市场观察",
        "keep",
        ("a股", "股票", "财经", "今日看盘", "基金", "半导体", "上证指数"),
    ),
    Rule(
        "财富自由、暴富、提前退休、何时躺平",
        "keep",
        ("存款", "利息", "提前退休", "暴富", "赚钱", "财富自由"),
    ),
    Rule(
        "房地产、买房、婚育成本压力",
        "keep",
        ("房地产", "房价", "房产", "买房", "不买房", "不结婚"),
    ),
    Rule(
        "游戏/二游/电竞圈摆烂",
        "weak",
        (
            "王者荣耀",
            "英雄联盟",
            "第五人格",
            "剑网3",
            "炉石传说",
            "原神",
            "奇迹暖暖",
            "闪耀暖暖",
            "恋与深空",
            "恋与制作人",
            "edg",
            "rng",
            "lpl",
        ),
    ),
    Rule(
        "饭圈、佛系追星、CP文化",
        "weak",
        (
            "追星",
            "饭圈",
            "粉圈",
            "cp",
            "cp粉",
            "solo",
            "时代少年团",
            "时代峰峻",
            "r1se",
            "王一博",
            "成毅",
        ),
    ),
    Rule(
        "票务/周边/收物交易型追星",
        "weak",
        ("收票", "走平台", "带价来", "勿扰", "溢价", "佛系收"),
    ),
    Rule(
        "体育赛事与竞技吐槽",
        "weak",
        ("孙颖莎", "王楚钦", "樊振东", "陈梦", "乒乓", "nba", "篮球", "火箭"),
    ),
    Rule(
        "文娱作品评论",
        "weak",
        ("演员", "导演", "编剧", "演技", "书名", "主角", "小说", "评分"),
    ),
    Rule(
        "摆烂式旅游/出行休闲",
        "keep",
        (
            "旅游",
            "旅行",
            "微博旅行家",
            "带着微博去旅行",
            "大理",
            "丽江",
            "海滩",
            "酒店",
            "好吃",
        ),
    ),
    Rule(
        "育儿、母职、妈妈角色",
        "keep",
        ("孩子", "妈妈", "育儿", "家庭教育", "宝宝", "二胎", "孕期", "喂奶"),
    ),
    Rule(
        "婚恋与亲密关系",
        "keep",
        ("婚姻", "情感", "恋爱", "爱情", "异地恋", "失恋", "分手"),
    ),
    Rule(
        "论文、作业、写作卡壳",
        "keep",
        ("论文", "答辩", "作业", "不想写", "写不出来", "写不完"),
    ),
    Rule(
        "校园/宿舍/开学场景",
        "keep",
        ("大学", "大学生", "宿舍", "开学", "室友"),
    ),
    Rule(
        "学习厌倦、考试压力、考研上岸",
        "keep",
        (
            "学习",
            "上学",
            "高考",
            "高三",
            "考研",
            "考试",
            "不想学",
            "不想上学",
            "读书",
            "看书",
            "老师",
            "同学",
        ),
    ),
    Rule(
        "职场规则、领导关系、办公室博弈",
        "weak",
        ("领导", "提拔", "升职", "办公室", "职场生存法则", "员工"),
    ),
    Rule(
        "摸鱼、离职、精神离职、职场摆烂",
        "keep",
        ("摸鱼", "辞职", "离职", "裸辞", "加班", "实习", "不想加班"),
    ),
    Rule(
        "工作厌倦/不想上班/不想干活",
        "keep",
        ("不想上班", "工作", "上班", "打工人", "不想工作", "不想干活", "老板", "同事"),
    ),
    Rule(
        "运动、自律、减肥",
        "keep",
        ("减肥", "减肥打卡", "运动", "运动打卡", "跑步", "马拉松", "游泳", "羽毛球"),
    ),
    Rule(
        "身体不适、疼痛、疲劳、失眠",
        "keep",
        ("咳嗽", "发烧", "感冒", "腰疼", "头疼", "失眠", "熬夜", "睡眠", "嗓子疼", "鼻塞"),
    ),
    Rule(
        "焦虑、心累、卷不动、压力过载",
        "keep",
        (
            "焦虑",
            "心累",
            "好累",
            "太累了",
            "卷不动",
            "躺又躺不平",
            "压力",
            "崩溃",
            "不想努力",
            "天赋",
        ),
    ),
    Rule(
        "在家躺平/回家/床上腐烂/只想睡觉",
        "keep",
        ("回家", "在家", "睡觉", "床上", "躺床", "休息", "刷手机", "玩手机", "好困"),
    ),
    Rule(
        "周末、假期、节律性休息",
        "keep",
        ("周末", "假期", "放假", "周一", "周五", "新的一年", "生日快乐", "生日", "这一年"),
    ),
    Rule(
        "创作记录",
        "weak",
        ("速写", "画画", "绘画", "约稿", "拍照", "修图", "相册"),
    ),
    Rule(
        "星座/命理/日运与人格标签",
        "weak",
        ("mbti", "isfp", "infp", "infj", "enfp", "星座", "星盘", "水瓶座", "摩羯座", "日运", "乙木", "壬水"),
    ),
    Rule(
        "商业交易/售后/平台服务",
        "weak",
        ("退款", "退货", "售后", "赔款", "投诉", "保险", "假一赔十"),
    ),
    Rule(
        "平台客服/消费维权",
        "weak",
        ("客服", "维权", "投诉", "保险", "赔付", "误工费", "护理费"),
    ),
    Rule(
        "接单/陪玩/代招交易",
        "weak",
        ("代招", "陪玩", "人头", "日入", "红包", "接单"),
    ),
    Rule(
        "计划、任务、拖延、完成度",
        "weak",
        ("计划", "任务完成", "主线任务", "完成任务", "拖延", "今日任务"),
    ),
    Rule(
        "慢生活、天气、晒太阳、轻日常",
        "keep",
        ("晒太阳", "天气", "好好吃饭", "轻日常", "日常碎片", "ootd", "慢生活", "适合躺平"),
    ),
    Rule(
        "年轻人该不该躺平的公共讨论",
        "weak",
        (
            "年轻人",
            "躺平一词源于",
            "为什么很多年轻人",
            "更多是调侃",
            "大环境不好",
            "ai写文",
            "该不该躺平",
            "躺平式旅游为什么越来越受年轻人欢迎",
        ),
    ),
    Rule(
        "国际热点/社会时评外溢",
        "weak",
        ("日本", "台独", "美国", "中国", "热点解读", "国际"),
    ),
    Rule(
        "松弛、自洽、顺其自然",
        "keep",
        ("佛系", "顺其自然", "允许一切发生", "放平心态", "情绪稳定", "都行", "没关系", "无所谓"),
    ),
    Rule(
        "摆烂生活哲学/摆烂人生",
        "keep",
        ("摆烂人生", "摆烂生活", "人生感悟", "偶尔摆烂", "间歇摆烂", "快乐", "生活"),
    ),
    Rule(
        "直接躺平/摆烂宣言",
        "keep",
        (
            "想躺平",
            "只想躺平",
            "躺平了",
            "躺平吧",
            "摆烂的一天",
            "想摆烂",
            "只想摆烂",
            "摆烂了",
            "彻底摆烂",
            "主打一个摆烂",
        ),
    ),
    Rule(
        "平台模板/噪声混合主题",
        "omit",
        (
            "emoji",
            "p1",
            "jpg",
            "ps",
            "13 / 14 / 11 / 10",
            "朋友圈文案",
            "微博客服",
            "微博兴趣创作计划",
            "新星v计划",
            "曝光计划",
        ),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build macro-merge tables for O_outlier BERTopic run.")
    parser.add_argument("--run_dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--exclude_periods",
        nargs="*",
        default=list(DEFAULT_EXCLUDED_PERIODS),
        help="Period labels to drop before computing macro-level trends and share-based peaks.",
    )
    return parser.parse_args()


def assign_macro_topic(row: pd.Series) -> pd.Series:
    topic_id = int(row["topic_id"])
    if topic_id in MANUAL_TOPIC_OVERRIDES:
        macro_topic, report_action, source = MANUAL_TOPIC_OVERRIDES[topic_id]
        return pd.Series(
            {
                "macro_topic": macro_topic,
                "report_action": report_action,
                "merge_source": source,
            }
        )

    text = " ".join(
        str(row[column])
        for column in ("topic_name_raw", "topic_label_machine", "top_terms")
        if column in row
    ).lower()
    for rule in RULES:
        if any(marker.lower() in text for marker in rule.markers):
            return pd.Series(
                {
                    "macro_topic": rule.macro_topic,
                    "report_action": rule.report_action,
                    "merge_source": "rule",
                }
            )

    if "躺平" in text or "摆烂" in text:
        return pd.Series(
            {
                "macro_topic": "直接躺平/摆烂宣言",
                "report_action": "keep",
                "merge_source": "fallback-keyword",
            }
        )
    if "佛系" in text:
        return pd.Series(
            {
                "macro_topic": "松弛、自洽、顺其自然",
                "report_action": "keep",
                "merge_source": "fallback-keyword",
            }
        )
    if "崩溃" in text or "梦" in text:
        return pd.Series(
            {
                "macro_topic": "泛情绪碎片与口号化表达",
                "report_action": "weak",
                "merge_source": "fallback-emotion",
            }
        )
    return pd.Series(
        {
            "macro_topic": "其他长尾待复核",
            "report_action": "weak",
            "merge_source": "fallback-other",
        }
    )


def build_macro_overview(
    mapping_df: pd.DataFrame,
    share_df: pd.DataFrame,
    excluded_periods: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    action_rank = {"omit": 0, "weak": 1, "keep": 2}
    macro_action = (
        mapping_df.assign(_rank=mapping_df["report_action"].map(action_rank).fillna(-1))
        .sort_values(["macro_topic", "_rank"], ascending=[True, False])
        .drop_duplicates("macro_topic")
        .rename(columns={"report_action": "macro_report_action"})[["macro_topic", "macro_report_action"]]
    )
    mapping_for_macro = mapping_df.drop(columns=["report_action"]).merge(macro_action, on="macro_topic", how="left")
    filtered_share_df = share_df.copy()
    if excluded_periods:
        filtered_share_df = filtered_share_df[~filtered_share_df["period_label"].isin(excluded_periods)].copy()

    macro_period = (
        filtered_share_df.merge(
            mapping_for_macro[["topic_id", "macro_topic", "macro_report_action"]],
            on="topic_id",
            how="left",
        )
        .groupby(["macro_topic", "macro_report_action", "period_label"], dropna=False)
        .agg(doc_count=("doc_count", "sum"))
        .reset_index()
    )
    period_totals = macro_period.groupby("period_label", dropna=False)["doc_count"].sum().rename("period_total")
    macro_period = macro_period.merge(period_totals, on="period_label", how="left")
    macro_period["doc_share"] = macro_period["doc_count"] / macro_period["period_total"]

    peak = (
        macro_period.sort_values(["macro_topic", "doc_share", "doc_count", "period_label"], ascending=[True, False, False, True])
        .drop_duplicates("macro_topic")
        .rename(
            columns={
                "period_label": "peak_period",
                "doc_count": "peak_doc_count",
                "doc_share": "peak_doc_share",
            }
        )
    )[["macro_topic", "peak_period", "peak_doc_count", "peak_doc_share"]]

    overview = (
        mapping_for_macro.groupby(["macro_topic", "macro_report_action"], dropna=False)
        .agg(
            topic_n=("topic_id", "size"),
            doc_n=("topic_count", "sum"),
            top_topic_ids=("topic_id", lambda values: ", ".join(str(item) for item in list(values)[:10])),
            example_terms=("top_terms", lambda values: " || ".join(list(values)[:3])),
        )
        .reset_index()
        .rename(columns={"macro_report_action": "report_action"})
    )
    total_docs = overview["doc_n"].sum()
    overview["share_of_clustered_docs_pct"] = overview["doc_n"] / total_docs * 100
    overview = overview.merge(peak, on="macro_topic", how="left")
    return overview.sort_values(["report_action", "doc_n"], ascending=[True, False]).reset_index(drop=True), macro_period


def write_summary(
    summary_path: Path,
    mapping_df: pd.DataFrame,
    macro_overview: pd.DataFrame,
    excluded_periods: list[str] | tuple[str, ...],
) -> None:
    action_share = (
        macro_overview.groupby("report_action", dropna=False)["share_of_clustered_docs_pct"].sum().to_dict()
    )
    top_rows = macro_overview.head(15)
    omit_rows = macro_overview[macro_overview["report_action"] == "omit"]
    lines = [
        "# O_outlier Manual Macro Merge v1",
        "",
        "- source_run: `bert/artifacts/broad_analysis/topic_model_compare/O_outlier`",
        f"- macro_topics_used: `{mapping_df['macro_topic'].nunique()}`",
        f"- excluded_periods_for_trend: `{', '.join(excluded_periods) if excluded_periods else 'none'}`",
        "- peak_basis: `doc_share` after dropping excluded periods",
        f"- keep_share_pct: `{action_share.get('keep', 0.0):.2f}`",
        f"- weak_share_pct: `{action_share.get('weak', 0.0):.2f}`",
        f"- omit_share_pct: `{action_share.get('omit', 0.0):.2f}`",
        "",
        "## Read This As",
        "",
        "- `keep`：可以直接进入中期报告的 substantive 中层主题。",
        "- `weak`：可以保留为 broad 使用场景或边缘簇，但不建议当核心结论。",
        "- `omit`：明显的平台模板、虚词混合、数字文案或强污染 topic。",
        "",
        "## Top Macro Topics",
        "",
    ]
    for row in top_rows.itertuples(index=False):
        lines.append(
            f"- `{row.macro_topic}` | topics={row.topic_n} | docs={row.doc_n} | "
            f"share={row.share_of_clustered_docs_pct:.2f}% | peak={row.peak_period}"
        )
    if not omit_rows.empty:
        lines.extend(["", "## Omit Buckets", ""])
        for row in omit_rows.itertuples(index=False):
            lines.append(
                f"- `{row.macro_topic}` | topics={row.topic_n} | docs={row.doc_n} | "
                f"share={row.share_of_clustered_docs_pct:.2f}%"
            )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    overview_path = run_dir / "readouts" / "topic_overview.csv"
    share_path = run_dir / "readouts" / "topic_share_by_period.csv"

    topic_overview = pd.read_csv(overview_path).sort_values("topic_count", ascending=False).reset_index(drop=True)
    mapping = pd.concat([topic_overview, topic_overview.apply(assign_macro_topic, axis=1)], axis=1)
    excluded_periods = [period for period in args.exclude_periods]
    macro_overview, macro_share = build_macro_overview(mapping, pd.read_csv(share_path), excluded_periods)

    mapping.to_csv(output_dir / "topic_macro_mapping.csv", index=False)
    macro_overview.to_csv(output_dir / "macro_topic_overview.csv", index=False)
    macro_share.to_csv(output_dir / "macro_topic_share_by_period.csv", index=False)
    write_summary(output_dir / "merge_summary.md", mapping, macro_overview, excluded_periods)

    print(f"wrote {output_dir}")
    print(f"macro_topics_used={mapping['macro_topic'].nunique()}")
    print(f"excluded_periods={excluded_periods}")
    print(macro_overview.head(12).to_string(index=False))


if __name__ == "__main__":
    main()

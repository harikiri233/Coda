"""评测任务定义：20 个任务，3 个仓库（PaperLens + toolz 1.0.0 + more-itertools 10.8.0）。

- bug：对源码做一处替换（mutations），提示词用 issue 的口吻描述现象。
- feature：把某个函数的实现换成 raise NotImplementedError（stub），提示词描述需求。

隐藏测试不手写：build.py 应用改动后跑一遍完整测试，失败的用例就是隐藏测试，并把它们从工作区的
测试文件里删掉（Agent 看不到）。判定时把原始测试文件放回去，只跑这些用例。
"""

REPOS = {
    "paperlens": {
        "archive": "paperlens.tar.gz",
        "setup": "uv sync --frozen -q",
        "test": "uv run --frozen pytest -q -p no:cacheprovider",
    },
    "toolz": {
        "archive": "toolz.tar.gz",
        # 根目录放一个空 conftest.py，让直接运行 pytest 时也导入工作区里的 toolz，而不是系统里装的
        "setup": "touch conftest.py",
        "test": "python -m pytest -q -p no:cacheprovider",
    },
    "more-itertools": {
        "archive": "more-itertools.tar.gz",
        "setup": "",
        "test": "python -m pytest -q -p no:cacheprovider",
    },
}

TASKS = [
    # ------------------------------------------------------------ PaperLens：bug
    {
        "id": "pl-embed-order",
        "repo": "paperlens",
        "kind": "bug",
        "mutations": [
            (
                "src/paperlens/models/siliconflow.py",
                'items = sorted(data["data"], key=lambda item: item["index"])',
                'items = data["data"]',
            )
        ],
        "prompt": "入库新论文之后，稠密检索的结果明显错乱：问 Self-RAG 的反思标记，返回的段落和问题毫不相关，"
        "但 BM25 单路检索是正常的。怀疑是批量生成向量时哪一步出了问题（硅基流动的接口不保证按输入顺序返回）。请排查并修复。",
    },
    {
        "id": "pl-weekend-price",
        "repo": "paperlens",
        "kind": "bug",
        "mutations": [
            ("src/paperlens/models/usage.py", "if t.weekday() >= 5:", "if t.weekday() > 5:")
        ],
        "prompt": "命令行最后打印的费用估算偏高：周六白天的 DeepSeek 调用按高峰价计费了。"
        "DeepSeek 周六、周日全天都是非高峰价。请修复。",
    },
    {
        "id": "pl-dup-citation",
        "repo": "paperlens",
        "kind": "bug",
        "mutations": [
            (
                "src/paperlens/rag/answer.py",
                "            if n not in numbers:\n                numbers.append(n)",
                "            numbers.append(n)",
            )
        ],
        "prompt": "回答里同一个来源被引用多次时（比如“……[1]。……[1][2]。”），回答下面的参考来源列表里同一条来源会重复出现。"
        "引用编号应该按首次出现的顺序去重。请修复。",
    },
    {
        "id": "pl-history-truncate",
        "repo": "paperlens",
        "kind": "bug",
        "mutations": [
            (
                "src/paperlens/rag/query.py",
                'if m["role"] != "user" and len(content) > 400:',
                'if m["role"] == "user" and len(content) > 400:',
            )
        ],
        "prompt": "多轮对话里，用户发了一段很长的问题之后，下一轮的指代消解经常丢掉用户上一个问题的后半截；"
        "同时助手之前的长回答被完整塞进了查询分析的提示词，token 消耗很大。"
        "预期行为是：拼接对话历史时只截断助手的长回答，用户的原话完整保留。请修复。",
    },
    # ------------------------------------------------------------ PaperLens：feature
    {
        "id": "pl-strip-invalid",
        "repo": "paperlens",
        "kind": "feature",
        "stub": ("src/paperlens/rag/answer.py", "strip_invalid"),
        "prompt": "src/paperlens/rag/answer.py 里的 strip_invalid(answer, invalid) 还没有实现，请实现它。"
        "需求：去掉回答文本里越界的引用编号 invalid（整数列表）。方括号里还剩其他编号时保留剩下的，"
        "用英文逗号加空格连接，例如只有 6 条证据时“[2, 9]”去掉 9 变成“[2]”；一个编号都不剩时把整个方括号删掉；"
        "invalid 为空时原样返回。引用的写法见同一文件里的 _CITATION（编号之间可以用中文或英文逗号分隔）。",
    },
    {
        "id": "pl-build-query",
        "repo": "paperlens",
        "kind": "feature",
        "stub": ("src/paperlens/ingest/arxiv_client.py", "build_query"),
        "prompt": "src/paperlens/ingest/arxiv_client.py 里的 build_query(keywords, op='AND') 还没有实现，请实现它："
        '把关键词列表转换成 arXiv API 的 search_query 字符串。规则：每个关键词先把 ( ) " : 这四种字符替换成空格，'
        '再去掉首尾空白，结果为空就跳过；含空格的短语写成 all:"短语"，单个词写成 all:词；'
        "各项之间用两边带空格的 op 连接。例如 ['RAG', 'dense retrieval'] 得到 'all:RAG AND all:\"dense retrieval\"'。",
    },
    # ------------------------------------------------------------ toolz：bug
    {
        "id": "tz-partition-all",
        "repo": "toolz",
        "kind": "bug",
        "mutations": [
            (
                "toolz/itertoolz.py",
                "            yield prev[:len(seq) % n]",
                "            yield prev[:len(seq) // n]",
            )
        ],
        "prompt": "partition_all 对有长度的序列，最后一组的元素数不对：list(partition_all(3, range(5))) 应该是 "
        "[(0, 1, 2), (3, 4)]，现在最后一组是 (3,)。传入迭代器（iter(range(5))）时结果是对的。请修复。",
    },
    {
        "id": "tz-interpose",
        "repo": "toolz",
        "kind": "bug",
        "mutations": [
            (
                "toolz/itertoolz.py",
                "    inposed = concat(zip(itertools.repeat(el), seq))\n    next(inposed)\n",
                "    inposed = concat(zip(itertools.repeat(el), seq))\n",
            )
        ],
        "prompt": "interpose 的结果开头多了一个分隔符：list(interpose('.', ['a', 'b', 'c'])) 应该是 "
        "['a', '.', 'b', '.', 'c']，现在是 ['.', 'a', '.', 'b', '.', 'c']。请修复。",
    },
    {
        "id": "tz-reduceby",
        "repo": "toolz",
        "kind": "bug",
        "mutations": [
            (
                "toolz/itertoolz.py",
                "                d[k] = item\n                continue\n",
                "                d[k] = item\n",
            )
        ],
        "prompt": "reduceby 不传 init 时，每组的第一个元素被算了两次：reduceby(lambda x: x % 2 == 0, operator.add, "
        "[1, 2, 3, 4]) 应该是 {True: 6, False: 4}，现在得到 {True: 8, False: 5}。传 init 时是对的。请修复。",
    },
    {
        "id": "tz-unique-key",
        "repo": "toolz",
        "kind": "bug",
        "mutations": [
            (
                "toolz/itertoolz.py",
                "            val = key(item)\n            if val not in seen:\n                seen_add(val)\n",
                "            val = key(item)\n            if val not in seen:\n                seen_add(item)\n",
            )
        ],
        "prompt": "unique 传 key 时没有去重：tuple(unique(['cat', 'mouse', 'dog', 'hen'], key=len)) 应该是 "
        "('cat', 'mouse')，现在四个元素全都返回了。不传 key 时正常。请修复。",
    },
    # ------------------------------------------------------------ toolz：feature
    {
        "id": "tz-topk",
        "repo": "toolz",
        "kind": "feature",
        "stub": ("toolz/itertoolz.py", "topk"),
        "prompt": "toolz/itertoolz.py 里的 topk(k, seq, key=None) 还没有实现，请实现：返回 seq 中最大的 k 个元素组成的 "
        "tuple，从大到小排列。key 可以是函数；也可以是不可调用的索引或键，这时用同一文件里的 getter(key) 取值"
        "（如 key='a' 取字典的 'a' 项，key=0 取元组的第一项）。比较值相同时保持元素在原序列中的先后顺序。"
        "seq 可以是迭代器。",
    },
    {
        "id": "tz-merge-with",
        "repo": "toolz",
        "kind": "feature",
        "stub": ("toolz/dicttoolz.py", "merge_with"),
        "prompt": "toolz/dicttoolz.py 里的 merge_with(func, *dicts, **kwargs) 还没有实现，请实现：合并多个字典，"
        "同一个键在各个字典里的值按出现顺序收集成列表 [v1, v2, ...]，结果里这个键的值是 func(列表)；"
        "只出现一次的键也是 func([v])。只传了一个位置参数、而且它不是 Mapping 时，把它当作字典组成的可迭代对象"
        "（列表或迭代器）。关键字参数 factory 指定结果映射的类型（默认 dict），其他关键字参数抛 TypeError"
        "（用同一文件里的 _get_factory）。一个字典都没有时返回空映射。",
    },
    {
        "id": "tz-diff",
        "repo": "toolz",
        "kind": "feature",
        "stub": ("toolz/itertoolz.py", "diff"),
        "prompt": "toolz/itertoolz.py 里的 diff(*seqs, **kwargs) 还没有实现，请实现：并行遍历多个序列，"
        "产出各序列同一位置上的元素不完全相同的那些元组。要求：只传了一个参数而且它是 list 时，把它当作序列组成的列表；"
        "序列少于 2 个时抛 TypeError；默认按最短的序列截断，传了 default= 时按最长的序列补齐，缺的位置用 default；"
        "传了 key= 时用 key(元素) 比较，但产出的仍是原始元素。例如 list(diff([1, 2, 3], [1, 2, 10])) == [(3, 10)]。",
    },
    # ------------------------------------------------------------ more-itertools：bug
    {
        "id": "mi-windowed-step",
        "repo": "more-itertools",
        "kind": "bug",
        "mutations": [
            (
                "more_itertools/more.py",
                "    for _ in islice(filler, step - 1, None, step):",
                "    for _ in islice(filler, step, None, step):",
            )
        ],
        "prompt": "windowed 的窗口位置不对：list(windowed([1, 2, 3, 4, 5], 2)) 应该是 [(1, 2), (2, 3), (3, 4), (4, 5)]，"
        "现在是 [(1, 2), (3, 4), (4, 5)]；list(windowed(range(1, 8), 3, step=2)) 应该是 "
        "[(1, 2, 3), (3, 4, 5), (5, 6, 7)]，现在是 [(1, 2, 3), (4, 5, 6), (6, 7, None)]。请修复。",
    },
    {
        "id": "mi-chunked-even",
        "repo": "more-itertools",
        "kind": "bug",
        "mutations": [
            (
                "more_itertools/more.py",
                "    num_full = length - partial_size * num_lists",
                "    num_full = length - full_size * num_lists",
            )
        ],
        "prompt": "chunked_even 分块不均匀：list(chunked_even('ABCDEFG', 3)) 应该是 "
        "[['A', 'B', 'C'], ['D', 'E'], ['F', 'G']]（各块长度最多差 1），现在是 "
        "[['A', 'B', 'C'], ['D'], ['E'], ['F'], ['G']]。请修复。",
    },
    {
        "id": "mi-split-at-maxsplit",
        "repo": "more-itertools",
        "kind": "bug",
        "mutations": [
            (
                "more_itertools/more.py",
                "            if maxsplit == 1:\n                yield list(it)",
                "            if maxsplit == 0:\n                yield list(it)",
            )
        ],
        "prompt": "split_at 的 maxsplit 多切了一次：list(split_at('a,bb,ccc,dddd', lambda x: x == ',', maxsplit=1)) "
        "应该和 'a,bb,ccc,dddd'.split(',', 1) 一样得到两段 [['a'], ['b', 'b', ',', 'c', 'c', 'c', ',', 'd', 'd', 'd', 'd']]，"
        "现在得到三段。请修复。",
    },
    {
        "id": "mi-consecutive-groups",
        "repo": "more-itertools",
        "kind": "bug",
        "mutations": [
            (
                "more_itertools/more.py",
                "        key = lambda x: x[0] - x[1]\n",
                "        key = lambda x: x[0] + x[1]\n",
            )
        ],
        "prompt": "consecutive_groups 不传 ordering 时分组错误：[list(g) for g in consecutive_groups([1, 2, 4, 5, 10])] "
        "应该是 [[1, 2], [4, 5], [10]]，现在每个数字各自成了一组。传了 ordering 时是对的。请修复。",
    },
    # ------------------------------------------------------------ more-itertools：feature
    {
        "id": "mi-mark-ends",
        "repo": "more-itertools",
        "kind": "feature",
        "stub": ("more_itertools/more.py", "mark_ends"),
        "prompt": "more_itertools/more.py 里的 mark_ends(iterable) 还没有实现，请实现：产出 (is_first, is_last, item) "
        "三元组，标记每个元素是不是第一个、是不是最后一个。空的可迭代对象什么都不产出；只有一个元素时产出 "
        "(True, True, x)。要求惰性求值，能用于迭代器（最多预读一个元素）。",
    },
    {
        "id": "mi-distribute",
        "repo": "more-itertools",
        "kind": "feature",
        "stub": ("more_itertools/more.py", "distribute"),
        "prompt": "more_itertools/more.py 里的 distribute(n, iterable) 还没有实现，请实现：把元素轮流分配到 n 个"
        "较小的可迭代对象里，返回长度为 n 的列表，第 i 个包含原序列的第 i、i+n、i+2n……个元素（从 0 开始数）；"
        "元素少于 n 个时，后面的为空；n 小于 1 时抛 ValueError。例如 [list(c) for c in distribute(3, [1, 2, 3, 4, 5, 6, 7])] "
        "== [[1, 4, 7], [2, 5], [3, 6]]。",
    },
    {
        "id": "mi-split-when",
        "repo": "more-itertools",
        "kind": "feature",
        "stub": ("more_itertools/more.py", "split_when"),
        "prompt": "more_itertools/more.py 里的 split_when(iterable, pred, maxsplit=-1) 还没有实现，请实现："
        "依次检查相邻的两个元素，pred(前一个, 后一个) 返回 True 时在它们之间切开，产出各段组成的列表。"
        "maxsplit 是最多切几次（-1 表示不限制，0 表示整个作为一段）；输入为空时什么都不产出。例如 "
        "list(split_when([1, 2, 3, 3, 2, 5, 2, 4, 2], lambda x, y: x > y)) == [[1, 2, 3, 3], [2, 5], [2, 4], [2]]。",
    },
]

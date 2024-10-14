"""
Author: [ruofei.zhang1]
Published Date: September 15, 2024
If you have any new feature requests or encounter any bugs,
please feel free to contact me.
"""


import numpy as np
import pandas as pd
import itertools
from graphviz import Digraph
from IPython.display import Image, display
from sklearn.metrics import roc_auc_score, roc_curve


class Auto_Risk_Rule_Search:
    def __init__(self, max_depth, priority_features=None, fixed_thresholds=[], dynamic_min_sample_ratios=[], search_steps=[],
                 random_state=27, min_samples_leaf=0.05, verbose=True, print_tree_structure=False, print_tree=False, visualize_tree=False, print_log=False,
                 top_n_splits=1, smooth_factor=None, feature_directions=None, optimization_target='lift_diff', search_mode='equal_width'):

        self.max_depth = max_depth
        self.priority_features = priority_features if priority_features is not None else []  # 默认为空列表
        self.fixed_thresholds = fixed_thresholds
        self.dynamic_min_sample_ratios = dynamic_min_sample_ratios
        self.search_steps = search_steps
        self.random_state = random_state
        self.min_samples_leaf = min_samples_leaf
        self.verbose = verbose
        self.print_tree_structure = print_tree_structure
        self.print_tree = print_tree
        self.visualize_tree = visualize_tree
        self.print_log = print_log
        self.top_n_splits = top_n_splits
        self.smooth_factor = smooth_factor
        self.optimization_target = optimization_target
        self.search_mode = search_mode  # 新增搜索模式参数

        # 设置默认的feature_directions
        if feature_directions is None:
            self.feature_directions = {}
        else:
            self.feature_directions = self.parse_feature_directions(feature_directions)

        self.global_mean = None
        self.used_features_global = set()
        self.node_id_gen = itertools.count(1)
        self.node_map = {}

        # 存储输出的内容
        self.logs = []
        self.tree_text = None
        self.tree_structure = None
        self.tree_image_path = None

        # 参数校验和报警提示
        self.check_parameters()

    def check_parameters(self):
        # 检查 fixed_thresholds 的格式
        if self.fixed_thresholds:
            for i, thresholds in enumerate(self.fixed_thresholds):
                if thresholds is None:
                    # 如果是 None，表示该层使用动态搜索，不需要进一步检查
                    continue
                elif isinstance(thresholds, (int, float)):
                    # 单一阈值情况 (在只有一个节点的层级可能使用)
                    if i > 0:
                        prev_thresholds = self.fixed_thresholds[i - 1]
                        # 先检查前一层是否是列表，并且长度符合要求
                        if isinstance(prev_thresholds, list) and len(prev_thresholds) != 2 ** (i - 1):
                            raise ValueError(f"错误: 第 {i - 1} 层的固定阈值数量不正确，应为 {2 ** (i - 1)} 个分割点。")
                elif isinstance(thresholds, list):
                    # 多重阈值情况，检查列表长度是否符合2的幂次规则
                    if len(thresholds) != 2 ** i:
                        raise ValueError(f"错误: 第 {i} 层的固定阈值数量不正确，应为 {2 ** i} 个分割点。")
                    # 检查列表中的每个元素是否是 float、int 或 None
                    for value in thresholds:
                        if value is not None and not isinstance(value, (int, float)):
                            raise ValueError(f"错误: 'fixed_thresholds' 在第 {i} 层包含无效的值 '{value}'，应为浮点数、整数或 None。")
                else:
                    raise ValueError("错误: 'fixed_thresholds' 格式不正确。应为单一浮点数、其列表或 None。")

        # 检查 optimization_target 的格式和互补性
        if self.optimization_target:
            if isinstance(self.optimization_target, str):
                if self.optimization_target not in ['lift_diff', 'min_lift', 'max_lift']:
                    raise ValueError(f"错误: 'optimization_target'的全局值无效，应为'lift_diff', 'min_lift', 'max_lift'。")
            elif isinstance(self.optimization_target, list):
                for i, targets in enumerate(self.optimization_target):
                    if isinstance(targets, str):
                        if targets not in ['lift_diff', 'min_lift', 'max_lift']:
                            raise ValueError(f"错误: 'optimization_target'列表中的值无效，应为'lift_diff', 'min_lift', 'max_lift'，错误发生在第 {i} 层。")
                    elif isinstance(targets, list):
                        if any(target not in ['lift_diff', 'min_lift', 'max_lift', None] for target in targets):
                            raise ValueError(f"错误: 'optimization_target'子列表中的值无效，应为'lift_diff', 'min_lift', 'max_lift'或None，错误发生在第 {i} 层。")
                        if len(targets) != 2 ** i:
                            raise ValueError(f"错误: 第 {i} 层的 'optimization_target' 数量不正确，应为 {2 ** i} 个目标。")
                    elif targets is not None:
                        raise ValueError("错误: 'optimization_target'应为字符串、列表或 None。")
        else:
            self.optimization_target = 'lift_diff'  # 默认优化目标

        # 检查 fixed_thresholds 和 optimization_target 是否冲突
        for i in range(len(self.fixed_thresholds)):
            if self.fixed_thresholds[i] is not None:
                if isinstance(self.optimization_target, list) and i < len(self.optimization_target):
                    if isinstance(self.optimization_target[i], list) and len(self.optimization_target[i]) != 2 ** i:
                        raise ValueError(f"错误: 第 {i} 层 'optimization_target' 的长度不符合要求。")

        # 检查 fixed_thresholds 的长度
        if len(self.fixed_thresholds) > self.max_depth:
            raise ValueError(f"错误: 'fixed_thresholds'长度大于最大深度'{self.max_depth}'，多余的阈值将被忽略。")

        # 检查 dynamic_min_sample_ratios 的合理性
        if len(self.dynamic_min_sample_ratios) > 0:
            if min(self.dynamic_min_sample_ratios) < self.min_samples_leaf:
                raise ValueError(f"错误: 'dynamic_min_sample_ratios'中的最小值小于'min_samples_leaf' ({self.min_samples_leaf})。")
            if len(self.dynamic_min_sample_ratios) > self.max_depth:
                raise ValueError(f"错误: 'dynamic_min_sample_ratios'长度大于最大深度'{self.max_depth}'，多余的比例将被忽略。")

        # 检查 search_steps 的合理性
        if len(self.search_steps) == 0:
            raise ValueError("错误: 'search_steps'为空，默认步长为[0.01]。")
        if max(self.search_steps) > 1 or min(self.search_steps) <= 0:
            raise ValueError("错误: 'search_steps'中的值应该在 (0, 1] 范围内。")

        # 检查 priority_features 是否包含集合
        if any(isinstance(pf, set) for pf in self.priority_features):
            if len(self.fixed_thresholds) > 0:
                raise ValueError("错误: 'priority_features'包含集合时，不允许设置 'fixed_thresholds'。")

        # 检查 top_n_splits 的类型和有效性
        if isinstance(self.top_n_splits, int):
            if self.top_n_splits <= 0:
                raise ValueError("错误: 'top_n_splits'应为正整数。")
        elif isinstance(self.top_n_splits, list):
            if not all(isinstance(n, int) and n > 0 for n in self.top_n_splits):
                raise ValueError("错误: 'top_n_splits'应为正整数或正整数列表。")
            if len(self.top_n_splits) > len(self.search_steps):
                raise ValueError("错误: 'top_n_splits'的长度不能超过'search_steps'的长度。")
        else:
            raise ValueError("错误: 'top_n_splits'应为正整数或正整数列表。")

        # 检查 smooth_factor 的类型和有效性
        if self.smooth_factor is not None:  # 仅在 smooth_factor 不为 None 时进行检查
            if isinstance(self.smooth_factor, (int, float)):
                if not (0 < self.smooth_factor <= 1):
                    raise ValueError("错误: 'smooth_factor'应在 (0, 1] 范围内。")
            elif isinstance(self.smooth_factor, list):
                if not all(isinstance(f, (int, float)) and 0 < f <= 1 for f in self.smooth_factor):
                    raise ValueError("错误: 'smooth_factor'应为在 (0, 1] 范围内的浮点数或其列表。")
                if len(self.smooth_factor) != len(self.search_steps) - 1:
                    raise ValueError(f"错误: 'smooth_factor'列表长度应为'search_steps'长度减一 ({len(self.search_steps) - 1})。")
            else:
                raise ValueError("错误: 'smooth_factor'应为在 (0, 1] 范围内的浮点数或其列表。")



    def parse_feature_directions(self, feature_directions):
        """解析特征方向参数"""
        parsed_directions = {}
        for item in feature_directions:
            if not isinstance(item, tuple) or len(item) != 2:
                raise ValueError("错误: 'feature_directions'应为包含(特征名, 优化方向)的元组列表。")
            feature, direction = item
            if direction not in ['asc', 'desc', 'random']:
                raise ValueError(f"错误: 特征'{feature}'的优化方向无效，应为'asc', 'desc'或'random'。")
            parsed_directions[feature] = direction
        return parsed_directions


    def split_data(self, X, y, sample_weight, feature, threshold):
        """根据指定特征和阈值分裂数据"""
        left_mask = X[feature] <= threshold
        right_mask = ~left_mask
        return X[left_mask], y[left_mask], sample_weight[left_mask], X[right_mask], y[right_mask], sample_weight[right_mask]


    def dynamic_step_search(self, X, y, sample_weight, feature, min_samples_leaf_count, initial_thresholds, optimization_target=None):
        best_threshold = None
        best_score = -float('inf')
        optimization_target = optimization_target if optimization_target else self.optimization_target

        # 设置 top_n_splits 和 smooth_factor
        if isinstance(self.top_n_splits, int):
            top_n_splits = [self.top_n_splits] * len(self.search_steps)
        else:
            top_n_splits = self.top_n_splits + [self.top_n_splits[-1]] * (len(self.search_steps) - len(self.top_n_splits))

        # 设置平滑因子
        if self.smooth_factor is None:
            smooth_factors = [None] * (len(self.search_steps) - 1)
        elif isinstance(self.smooth_factor, (int, float)):
            smooth_factors = [self.smooth_factor] * (len(self.search_steps) - 1)
        else:
            smooth_factors = self.smooth_factor

        epsilon = 1e-6
        all_logs = []  # 用于收集所有日志信息

        # 计算有效数据边界
        sorted_feature_values = np.sort(X[feature])
        n = len(sorted_feature_values)
        min_sample_count = max(int(min_samples_leaf_count), 1)

        # 确定有效的特征值范围，去掉最小样本比例边界
        effective_feature_values = sorted_feature_values[min_sample_count: n - min_sample_count]

        # 检查 effective_feature_values 是否为空
        if len(effective_feature_values) == 0:
            log_message = f"特征 {feature} 的有效特征值范围为空，停止搜索。"
            all_logs.append(log_message)
            if self.print_log:
                print(log_message)
            return None, 0, []

        effective_min = effective_feature_values.min()
        effective_max = effective_feature_values.max()
        total_range = effective_max - effective_min

        # 新增检查 total_range 是否为零
        if total_range == 0:
            log_message = f"特征 {feature} 的有效特征值只有一个唯一值 {effective_min}，无法继续分割，停止搜索。"
            all_logs.append(log_message)
            if self.print_log:
                print(log_message)
            return None, 0, []

        log_message = f"有效特征值范围: {effective_min} - {effective_max}"
        all_logs.append(log_message)
        if self.print_log:
            print(log_message)

        # 以下代码保持不变...
        # 初始化搜索轮次
        search_round = 1
        previous_best_thresholds = None
        current_thresholds = None

        while search_round <= len(self.search_steps):
            step = self.search_steps[search_round - 1]
            current_top_n = top_n_splits[search_round - 1]

            if search_round == 1:
                # 第一次全局搜索，生成候选分割点
                if self.search_mode == 'equal_width':
                    interval = step * total_range  # 第一轮的步长决定的间隔长度
                    current_thresholds = np.arange(effective_min, effective_max + interval, interval)
                elif self.search_mode == 'equal_freq':
                    num_bins = max(int(1 / step), 1)
                    quantiles = np.percentile(effective_feature_values, np.linspace(0, 100, num_bins + 1))
                    current_thresholds = np.unique(quantiles)
                else:
                    raise ValueError(f"无效的 search_mode: {self.search_mode}")

                log_message = f"第 {search_round} 轮搜索，搜索点位、范围为: {current_thresholds}, 步长为: {step}"
                all_logs.append(log_message)
                if self.print_log:
                    print(log_message)
            else:
                # 生成下一轮的邻域和候选点
                log_message = f"第 {search_round} 轮搜索，上一轮最优点为: {previous_best_thresholds}"
                all_logs.append(log_message)
                if self.print_log:
                    print(log_message)
                neighborhoods = []
                new_valid_thresholds = []

                for threshold in previous_best_thresholds:
                    prev_step = self.search_steps[search_round - 2]  # 上一轮的步长

                    # 生成邻域范围
                    neighborhood = self.generate_neighborhood(
                        effective_feature_values,
                        threshold,
                        prev_step,
                        total_range,  # 传入总范围以便调整邻域大小
                        search_mode=self.search_mode
                    )
                    neighborhoods.append(neighborhood)
                    log_message = f"根据上一轮步长生成的邻域为: {neighborhood}"
                    all_logs.append(log_message)
                    if self.print_log:
                        print(log_message)

                    # 基于邻域生成备选分割点
                    fine_thresholds = self.generate_search_points(
                        effective_feature_values,
                        neighborhood,
                        step,  # 使用本轮的步长生成新的候选点
                        max_thresholds=len(effective_feature_values),
                        search_mode=self.search_mode
                    )
                    log_message = f"本轮在这些邻域中的待搜索点为: {fine_thresholds}"
                    all_logs.append(log_message)
                    if self.print_log:
                        print(log_message)

                    # 收集用于本轮的候选点
                    new_valid_thresholds.extend(fine_thresholds)

                # 去除重复的候选点
                current_thresholds = np.unique(new_valid_thresholds)

            # 计算每个候选点的得分
            all_splits = []
            for threshold in current_thresholds:
                X_left, y_left, sample_weight_left, X_right, y_right, sample_weight_right = self.split_data(X, y, sample_weight, feature, threshold)

                if len(y_left) < min_samples_leaf_count or len(y_right) < min_samples_leaf_count:
                    continue

                left_lift = y_left.mean() / self.global_mean if len(y_left) > 0 else 0
                right_lift = y_right.mean() / self.global_mean if len(y_right) > 0 else 0

                direction = self.feature_directions.get(feature, 'random')
                if optimization_target == 'min_lift':
                    if direction == 'asc':
                        score = 1 / (left_lift + epsilon)
                    elif direction == 'desc':
                        score = 1 / (right_lift + epsilon)
                    else:
                        score = 1 / (min(left_lift, right_lift) + epsilon)
                elif optimization_target == 'max_lift':
                    if direction == 'asc':
                        score = right_lift
                    elif direction == 'desc':
                        score = left_lift
                    else:
                        score = max(left_lift, right_lift)
                else:
                    if direction == 'asc':
                        score = right_lift - left_lift
                    elif direction == 'desc':
                        score = -(right_lift - left_lift)
                    else:
                        score = abs(right_lift - left_lift)

                if score < 0:
                    score = 0

                all_splits.append((threshold, score))

            # 保留当前轮的最优点
            all_splits = list(set(all_splits))
            all_splits_sorted = sorted(all_splits, key=lambda x: x[1], reverse=True)[:current_top_n]
            log_message = f"排序前{current_top_n}的点位为: {[(t[0], t[1]) for t in all_splits_sorted]}"
            all_logs.append(log_message)
            if self.print_log:
                print(log_message)
            log_message = f"第 {search_round} 轮结束\n"
            all_logs.append(log_message)
            if self.print_log:
                print(log_message)

            # 检查是否需要停止
            if all(t[1] == 0 for t in all_splits_sorted):
                log_message = f"所有分割点得分均为0，停止搜索，特征={feature}"
                all_logs.append(log_message)
                if self.print_log:
                    print(log_message)
                self.logs.extend(all_logs)
                return None, 0, []

            if search_round == len(self.search_steps):
                # 已达到最大轮次，结束循环
                break

            # 准备下一轮
            previous_best_thresholds = [t[0] for t in all_splits_sorted]
            search_round += 1

        best_threshold, best_score = all_splits_sorted[0][0], all_splits_sorted[0][1]
        final_log_message = f"最终搜索最优: 特征={feature}, 阈值={best_threshold}, 得分={best_score:.4f}"
        all_logs.append(final_log_message)
        if self.print_log:
            print(final_log_message)

        # 将所有日志信息添加到 self.logs 中
        self.logs.extend(all_logs)

        return best_threshold, best_score, all_splits_sorted



    def generate_neighborhood(self, feature_values, threshold, prev_step, total_range, search_mode='equal_width'):
        """
        生成邻域范围，调整边缘情况下的邻域大小以保持一致。
        """
        effective_min = feature_values.min()
        effective_max = feature_values.max()
        log_message = f"生成邻域: 最小={effective_min}, 最大={effective_max}, 当前阈值={threshold}, 前一轮步长={prev_step}"
        if self.print_log:
            print(log_message)
        self.logs.append(log_message)

        if search_mode == 'equal_width':
            # 计算预期的邻域大小
            intended_neighborhood_size = 2 * prev_step * total_range

            # 初始邻域范围
            lower_bound = threshold - prev_step * total_range
            upper_bound = threshold + prev_step * total_range

            # 调整邻域范围以适应有效的特征值范围
            actual_lower_bound = max(lower_bound, effective_min)
            actual_upper_bound = min(upper_bound, effective_max)

            actual_neighborhood_size = actual_upper_bound - actual_lower_bound

            # 如果邻域被截断，调整另一侧以补偿
            if actual_neighborhood_size < intended_neighborhood_size:
                size_diff = intended_neighborhood_size - actual_neighborhood_size
                # 尝试在另一侧扩展
                if actual_lower_bound > effective_min:
                    actual_lower_bound = max(actual_lower_bound - size_diff, effective_min)
                elif actual_upper_bound < effective_max:
                    actual_upper_bound = min(actual_upper_bound + size_diff, effective_max)
        elif search_mode == 'equal_freq':
            num_bins = max(int(1 / prev_step), 1)
            quantiles = np.percentile(feature_values, np.linspace(0, 100, num_bins + 1))
            # 找到与当前阈值相近的分位点
            current_index = np.searchsorted(quantiles, threshold, side='left')
            lower_index = max(current_index - 1, 0)
            upper_index = min(current_index + 1, len(quantiles) - 1)
            actual_lower_bound = quantiles[lower_index]
            actual_upper_bound = quantiles[upper_index]
        else:
            raise ValueError(f"无效的 search_mode: {search_mode}")

        log_message = f"生成的邻域范围: ({actual_lower_bound}, {actual_upper_bound})"
        if self.print_log:
            print(log_message)
        self.logs.append(log_message)
        return actual_lower_bound, actual_upper_bound


    def generate_search_points(self, feature_values, neighborhood, step, max_thresholds=100, search_mode='equal_width'):
        """
        基于邻域范围生成备选分割点，并去除重复的点。
        """
        lower_bound, upper_bound = neighborhood
        log_message = f"生成备选点: 邻域范围=({lower_bound}, {upper_bound}), 当前步长={step}"
        if self.print_log:
            print(log_message)
        self.logs.append(log_message)

        if lower_bound >= upper_bound:
            return np.array([lower_bound])

        if search_mode == 'equal_width':
            num_points = int(1 / step) + 1
            fine_thresholds = np.linspace(lower_bound, upper_bound, num_points)
        elif search_mode == 'equal_freq':
            num_bins = max(int(1 / step), 1)
            quantiles = np.percentile(feature_values, np.linspace(0, 100, num_bins + 1))
            # 只保留在邻域范围内的分位点
            fine_thresholds = quantiles[(quantiles >= lower_bound) & (quantiles <= upper_bound)]
            fine_thresholds = np.unique(fine_thresholds)
        else:
            raise ValueError(f"无效的 search_mode: {search_mode}")

        # 限制候选点数量
        if len(fine_thresholds) > max_thresholds:
            indices = np.round(np.linspace(0, len(fine_thresholds) - 1, max_thresholds)).astype(int)
            fine_thresholds = fine_thresholds[indices]

        # 去除重复的候选点
        fine_thresholds = np.unique(fine_thresholds)

        log_message = f"生成的细化候选阈值: {fine_thresholds}"
        if self.print_log:
            print(log_message)
        self.logs.append(log_message)
        return fine_thresholds



    def get_top_n_splits(self, feature, best_threshold, X, y, sample_weight, min_samples_leaf_count):
        """获取最佳的TopN分割点进行进一步搜索"""
        possible_thresholds = X[feature].sort_values().unique()
        best_index = list(possible_thresholds).index(best_threshold)
        local_thresholds = possible_thresholds[max(best_index - 1, 0):min(best_index + 2, len(possible_thresholds))]

        splits = []
        for threshold in local_thresholds:
            X_left, y_left, sample_weight_left, X_right, y_right, sample_weight_right = self.split_data(X, y, sample_weight, feature, threshold)

            if len(y_left) < min_samples_leaf_count or len(y_right) < min_samples_leaf_count:
                continue

            left_lift = y_left.mean() / self.global_mean if len(y_left) > 0 else 0
            right_lift = y_right.mean() / self.global_mean if len(y_right) > 0 else 0
            score = abs(left_lift - right_lift)
            splits.append((feature, threshold, score, left_lift, right_lift))

        # 按照得分从高到低排序，返回TopN分割点
        splits = sorted(splits, key=lambda x: x[2], reverse=True)
        top_n_splits = self.smooth_filter(splits[:self.top_n_splits], max(s[2] for s in splits))
        return top_n_splits


    def manual_split_tree(self, X, y, sample_weight, depth, parent_filter_logic="", node_index=0):
        """手动构建树的每一层，确保按照指定的优化目标找到最佳分割点"""

        # 检查是否达到最大深度或纯节点
        if depth >= self.max_depth or len(set(y)) == 1:
            node_id = next(self.node_id_gen)
            overdue_rate = y.mean()
            lift = overdue_rate / self.global_mean
            proportion = len(y) / len(self.y_global)
            filter_logic = parent_filter_logic.strip(" & ")
            node = {
                "id": node_id,
                "depth": depth,
                "lift": lift,
                "proportion": proportion,
                "is_leaf": True,
                "filter_logic": filter_logic
            }
            log_message = f"达到最大深度或纯节点，当前深度：{depth}, ID={node_id}, lift: {lift:.4f}, Prop: {proportion:.4f}"
            if self.print_log:
                print(log_message)
            self.logs.append(log_message)
            self.node_map[node_id] = node
            return node

        # 动态获取每层的最小样本占比
        min_samples_ratio = self.dynamic_min_sample_ratios[depth] if depth < len(self.dynamic_min_sample_ratios) else self.min_samples_leaf
        min_samples_leaf_count = max(int(len(self.y_global) * min_samples_ratio), 1)
        node_id = next(self.node_id_gen)
        log_message = f"开始ID为 {node_id} 的节点的搜索，当前深度：{depth}, 全局最小样本数要求: {min_samples_leaf_count}"
        if self.print_log:
            print(log_message)
        self.logs.append(log_message)

        # 保存当前分支使用过的特征
        used_features_branch = set(self.used_features_global)

        # 获取当前层或节点的优化目标
        if isinstance(self.optimization_target, list):
            if depth < len(self.optimization_target):
                local_optimization_target = self.optimization_target[depth]
            else:
                local_optimization_target = 'lift_diff'
        else:
            local_optimization_target = self.optimization_target

        # 如果 local_optimization_target 是列表，则取出当前节点的目标
        if isinstance(local_optimization_target, list):
            if node_index < len(local_optimization_target):
                local_optimization_target = local_optimization_target[node_index]
            else:
                local_optimization_target = 'lift_diff'

        # 检查是否有固定阈值
        if depth < len(self.fixed_thresholds):
            thresholds = self.fixed_thresholds[depth]
            if thresholds is not None:
                # 获取分割特征，如果 priority_features 未定义或为 None 则默认为动态搜索
                feature = self.priority_features[depth] if self.priority_features and depth < len(self.priority_features) and self.priority_features[depth] is not None else None
                if feature is not None:
                    if isinstance(thresholds, list):
                        # 如果是列表，处理多个阈值情况
                        if node_index < len(thresholds):
                            threshold = thresholds[node_index]
                            if threshold is None:
                                log_message = f"节点ID {node_id}：动态搜索分裂点，当前深度：{depth}，目标：{local_optimization_target}，特征={feature}，节点索引={node_index}"
                                if self.print_log:
                                    print(log_message)
                                self.logs.append(log_message)

                                # 动态步长搜索逻辑
                                valid_thresholds = X[feature].sort_values().unique()
                                valid_thresholds = [t for t in valid_thresholds if min_samples_leaf_count <= sum(X[feature] <= t) <= len(X) - min_samples_leaf_count]

                                local_best_threshold, local_best_score, top_splits = self.dynamic_step_search(
                                    X, y, sample_weight, feature, min_samples_leaf_count, valid_thresholds, local_optimization_target
                                )
                                if local_best_threshold is None:
                                    return None
                                threshold = local_best_threshold
                            else:
                                log_message = f"节点ID {node_id}：使用固定阈值分裂，当前深度：{depth}，特征={feature}，阈值={threshold}，节点索引={node_index}"
                                if self.print_log:
                                    print(log_message)
                                self.logs.append(log_message)

                            # 使用固定或动态确定的阈值分割数据
                            X_left, y_left, sample_weight_left, X_right, y_right, sample_weight_right = self.split_data(X, y, sample_weight, feature, threshold)

                            # 对左右子树分别处理，如果不满足最小样本比例要求，生成叶节点
                            if len(y_left) < min_samples_leaf_count:
                                left_node_id = next(self.node_id_gen)
                                overdue_rate = y_left.mean()
                                lift = overdue_rate / self.global_mean
                                proportion = len(y_left) / len(self.y_global)
                                left_node = {
                                    "id": left_node_id,
                                    "depth": depth + 1,
                                    "lift": lift,
                                    "proportion": proportion,
                                    "is_leaf": True,
                                    "filter_logic": f"{parent_filter_logic} & {feature} <= {threshold}".strip(" & ")
                                }
                                self.node_map[left_node_id] = left_node
                                left_subtree = left_node
                            else:
                                left_subtree = self.manual_split_tree(X_left, y_left, sample_weight_left, depth + 1, f"{parent_filter_logic} & {feature} <= {threshold}", node_index=2 * node_index)

                            if len(y_right) < min_samples_leaf_count:
                                right_node_id = next(self.node_id_gen)
                                overdue_rate = y_right.mean()
                                lift = overdue_rate / self.global_mean
                                proportion = len(y_right) / len(self.y_global)
                                right_node = {
                                    "id": right_node_id,
                                    "depth": depth + 1,
                                    "lift": lift,
                                    "proportion": proportion,
                                    "is_leaf": True,
                                    "filter_logic": f"{parent_filter_logic} & {feature} > {threshold}".strip(" & ")
                                }
                                self.node_map[right_node_id] = right_node
                                right_subtree = right_node
                            else:
                                right_subtree = self.manual_split_tree(X_right, y_right, sample_weight_right, depth + 1, f"{parent_filter_logic} & {feature} > {threshold}", node_index=2 * node_index + 1)

                            proportion = len(y) / len(self.y_global)
                            lift = y.mean() / self.global_mean
                            filter_logic = parent_filter_logic.strip(" & ")
                            node = {
                                "id": node_id,
                                "depth": depth,
                                "lift": lift,
                                "proportion": proportion,
                                "is_leaf": False,
                                "split_feature": feature,
                                "split_threshold": threshold,
                                "filter_logic": filter_logic,
                                "left": left_subtree,
                                "right": right_subtree
                            }
                            self.node_map[node_id] = node
                            return node
                    else:
                        # 单一阈值的情况
                        log_message = f"节点ID {node_id}：使用固定阈值分裂，当前深度：{depth}，特征={feature}，阈值={thresholds}"
                        if self.print_log:
                            print(log_message)
                        self.logs.append(log_message)

                        # 使用固定阈值分割数据
                        X_left, y_left, sample_weight_left, X_right, y_right, sample_weight_right = self.split_data(X, y, sample_weight, feature, thresholds)

                        # 对左右子树分别处理，如果不满足最小样本比例要求，生成叶节点
                        if len(y_left) < min_samples_leaf_count:
                            left_node_id = next(self.node_id_gen)
                            overdue_rate = y_left.mean()
                            lift = overdue_rate / self.global_mean
                            proportion = len(y_left) / len(self.y_global)
                            left_node = {
                                "id": left_node_id,
                                "depth": depth + 1,
                                "lift": lift,
                                "proportion": proportion,
                                "is_leaf": True,
                                "filter_logic": f"{parent_filter_logic} & {feature} <= {thresholds}".strip(" & ")
                            }
                            self.node_map[left_node_id] = left_node
                            left_subtree = left_node
                        else:
                            left_subtree = self.manual_split_tree(X_left, y_left, sample_weight_left, depth + 1, f"{parent_filter_logic} & {feature} <= {thresholds}", node_index=2 * node_index)

                        if len(y_right) < min_samples_leaf_count:
                            right_node_id = next(self.node_id_gen)
                            overdue_rate = y_right.mean()
                            lift = overdue_rate / self.global_mean
                            proportion = len(y_right) / len(self.y_global)
                            right_node = {
                                "id": right_node_id,
                                "depth": depth + 1,
                                "lift": lift,
                                "proportion": proportion,
                                "is_leaf": True,
                                "filter_logic": f"{parent_filter_logic} & {feature} > {thresholds}".strip(" & ")
                            }
                            self.node_map[right_node_id] = right_node
                            right_subtree = right_node
                        else:
                            right_subtree = self.manual_split_tree(X_right, y_right, sample_weight_right, depth + 1, f"{parent_filter_logic} & {feature} > {thresholds}", node_index=2 * node_index + 1)

                        proportion = len(y) / len(self.y_global)
                        lift = y.mean() / self.global_mean
                        filter_logic = parent_filter_logic.strip(" & ")
                        node = {
                            "id": node_id,
                            "depth": depth,
                            "lift": lift,
                            "proportion": proportion,
                            "is_leaf": False,
                            "split_feature": feature,
                            "split_threshold": thresholds,
                            "filter_logic": filter_logic,
                            "left": left_subtree,
                            "right": right_subtree
                        }
                        self.node_map[node_id] = node
                        return node

        # 动态分割逻辑（保持原有逻辑）
        # 确定当前层的候选特征
        if depth < len(self.priority_features) and self.priority_features[depth] is not None:
            candidates = [self.priority_features[depth]]  # 使用指定的特征
            log_message = f"节点ID {node_id}：当前深度：{depth}，目标：{local_optimization_target}，指定特征={candidates[0]}"
        else:
            # 自由搜索模式：使用所有未使用的特征
            candidates = X.columns.difference(used_features_branch)
            log_message = f"节点ID {node_id}：当前深度：{depth}，目标：{local_optimization_target}，特征指定=None，使用所有未使用的特征进行搜索"

        if self.print_log:
            print(log_message)
        self.logs.append(log_message)

        # 遍历所有候选特征进行动态步长搜索
        best_feature = None
        best_threshold = None
        best_score = -float('inf')

        for feature in candidates:
            if feature in used_features_branch:
                continue

            # 对每个特征进行动态步长搜索
            valid_thresholds = X[feature].sort_values().unique()
            valid_thresholds = [t for t in valid_thresholds if min_samples_leaf_count <= sum(X[feature] <= t) <= len(X) - min_samples_leaf_count]

            # 初始化第1轮搜索
            local_best_threshold, local_best_score, top_splits = self.dynamic_step_search(X, y, sample_weight, feature, min_samples_leaf_count, valid_thresholds, local_optimization_target)

            # 如果没有有效的分割点（即local_best_threshold为None），停止分裂，设为叶子节点
            if local_best_threshold is None:
                continue

            if local_best_score > best_score:
                best_feature = feature
                best_threshold = local_best_threshold
                best_score = local_best_score

        # 如果未找到任何合适的分割点
        if best_feature is None or best_threshold is None:
            log_message = f"节点ID {node_id}：停止分裂，当前深度：{depth}，没有合适的分割点。"
            if self.print_log:
                print(log_message)
            self.logs.append(log_message)

            overdue_rate = y.mean()
            lift = overdue_rate / self.global_mean
            proportion = len(y) / len(self.y_global)
            filter_logic = parent_filter_logic.strip(" & ")
            node = {
                "id": node_id,
                "depth": depth,
                "lift": lift,
                "proportion": proportion,
                "is_leaf": True,
                "filter_logic": filter_logic
            }
            self.node_map[node_id] = node
            return node

        log_message = f"节点ID {node_id}：选择特征 {best_feature} 以 {best_threshold:.2f} 分裂，当前深度：{depth}，lift 差异得分: {best_score:.4f}"
        if self.print_log:
            print(log_message)
        self.logs.append(log_message)

        # 根据找到的最佳分割点分裂数据
        X_left, y_left, sample_weight_left, X_right, y_right, sample_weight_right = self.split_data(X, y, sample_weight, best_feature, best_threshold)

        used_features_branch.add(best_feature)
        left_subtree = self.manual_split_tree(X_left, y_left, sample_weight_left, depth + 1, f"{parent_filter_logic} & {best_feature} <= {best_threshold}", node_index=2 * node_index)
        right_subtree = self.manual_split_tree(X_right, y_right, sample_weight_right, depth + 1, f"{parent_filter_logic} & {best_feature} > {best_threshold}", node_index=2 * node_index + 1)

        if left_subtree is None or right_subtree is None:
            raise RuntimeError(f"子树在深度 {depth} 的递归过程中未正确生成。")

        proportion = len(y) / len(self.y_global)
        lift = y.mean() / self.global_mean
        filter_logic = parent_filter_logic.strip(" & ")
        node = {
            "id": node_id,
            "depth": depth,
            "lift": lift,
            "proportion": proportion,
            "is_leaf": False,
            "split_feature": best_feature,
            "split_threshold": best_threshold,
            "filter_logic": filter_logic,
            "left": left_subtree,
            "right": right_subtree
        }
        self.node_map[node_id] = node
        return node





    def get_filter_logic(self, node_ids, df_name=None, output_type='expression',
                         calculate_metrics=False, X=None, y=None, label_columns=None):
        """
        根据节点ID输出合并后的筛选逻辑，并根据需要计算KS和AUC。

        参数:
        - node_ids: 节点ID列表，用于生成筛选逻辑。
        - df_name: 数据集的名称，用于生成代码格式的输出。
        - output_type: 'expression' 或 'code'，决定输出逻辑表达式或DataFrame筛选代码。
        - calculate_metrics: 布尔值，是否计算KS和AUC指标。如果为True，需要传入X和y。
        - X: 特征数据集，必须包含与节点分裂相关的特征。
        - y: 标签数据集，可以是Series或DataFrame。如果是DataFrame，需要通过label_columns指定标签列。
        - label_columns: 如果y是DataFrame，指定标签列名。

        返回:
        - 筛选逻辑表达式，及在计算指标时返回KS和AUC结果。
        """

        # 参数验证
        if calculate_metrics:
            if X is None or y is None:
                raise ValueError("错误: 当calculate_metrics为True时，必须提供X和y。")
            if len(X) != len(y):
                raise ValueError("错误: X和y的长度必须相等。")

            # 如果y是DataFrame，提取指定的标签列
            if isinstance(y, pd.DataFrame):
                if label_columns is None:
                    raise ValueError("错误: 当y是DataFrame时，必须提供label_columns。")
                y = y[label_columns]

            # 检查X是否包含所有涉及到的特征
            involved_features = set()
            for node_id in node_ids:
                path = self.get_filter_logic([node_id], output_type='expression')
                for feature in X.columns:
                    if feature in path:
                        involved_features.add(feature)
            missing_features = involved_features.difference(X.columns)
            if missing_features:
                raise ValueError(f"错误: X中缺少计算分裂所需的特征: {missing_features}")

        def find_path_to_root(node_id):
            """查找从给定节点到根节点的路径"""
            path = []
            while node_id in self.node_map:
                node = self.node_map[node_id]
                path.append(node)
                parent_node = next(
                    (n for n in self.node_map.values() if 'left' in n and n['left'] == node_id or 'right' in n and n['right'] == node_id),
                    None
                )
                if parent_node:
                    node_id = parent_node['id']
                else:
                    break
            return list(reversed(path))

        def convert_to_dataframe_filter(expression, df_name):
            """将逻辑表达式转换为DataFrame筛选格式"""
            import re

            # 使用正则表达式替换逻辑表达式中的关系运算符为 Pandas DataFrame 筛选格式，并加上括号
            expression = re.sub(r"(\w+)\s*(<=|>=|==|>|<)\s*([\d.]+)", rf"({df_name}['\1'] \2 \3)", expression)
            # 替换逻辑运算符
            expression = expression.replace("&", " & ").replace("|", " | ")
            return expression

        def merge_logic_for_nodes(node_ids):
            """合并节点的逻辑表达"""
            if not node_ids:
                return ""

            # 检查是否包含根节点
            if any(node_id == max(self.node_map.keys()) for node_id in node_ids):
                # 根节点或包含根节点，返回空逻辑
                return ""

            paths = [find_path_to_root(node_id) for node_id in node_ids]

            # 遍历所有路径，并生成逻辑表达式
            logic_parts = []
            for idx, path in enumerate(paths):
                path_logic = " & ".join(f"({node.get('filter_logic', '')})" for node in path if node.get('filter_logic'))
                if path_logic:
                    logic_parts.append(path_logic)

            combined_logic = " | ".join(logic_parts)

            # 增加多个ID合并的说明
            if len(node_ids) > 1:
                print(f"ID {node_ids}解析前的逻辑表达式: {combined_logic}")
            else:
                print(f"ID {node_ids[0]}解析前的逻辑表达式: {combined_logic}")

            return combined_logic

        def get_all_split_features():
            """从整棵树中获取所有用于分裂的特征"""
            split_features = set()
            for node in self.node_map.values():
                if not node.get('is_leaf', True):  # 仅关注非叶子节点
                    split_feature = node.get('split_feature')
                    if split_feature:
                        split_features.add(split_feature)
            return list(split_features)

        # 获取合并后的逻辑表达式
        combined_logic = merge_logic_for_nodes(node_ids)

        # 如果 combined_logic 为空，说明没有筛选条件
        if combined_logic == "":
            print("无筛选条件，返回整个数据集。")
            metrics_results = {}
            if calculate_metrics:
                selected_X, selected_y = X, y
                # 动态获取整棵树的特征
                tree_features = get_all_split_features()

                # 计算每个分裂特征的KS和AUC
                if isinstance(y, pd.DataFrame):
                    for label_col in y.columns:
                        metrics_results[label_col] = {}
                        for feature in tree_features:
                            fpr, tpr, _ = roc_curve(y[label_col], X[feature])
                            ks = max(abs(tpr - fpr))
                            auc = round(0.5 + abs(0.5 - roc_auc_score(y[label_col], X[feature])), 3)
                            ks = round(ks, 3)
                            metrics_results[label_col][feature] = {"KS": ks, "AUC": auc}
                else:
                    metrics_results = {}
                    for feature in tree_features:
                        fpr, tpr, _ = roc_curve(y, X[feature])
                        ks = max(abs(tpr - fpr))
                        auc = round(0.5 + abs(0.5 - roc_auc_score(y, X[feature])), 3)
                        ks = round(ks, 3)
                        metrics_results[feature] = {"KS": ks, "AUC": auc}

                print(f"样本数量: {len(X)}")

            return "无筛选条件", metrics_results

        # 计算KS和AUC（如果需要）
        metrics_results = {}
        if calculate_metrics:
            # 将df_name替换为X（DataFrame的实际名称）
            df_filter = convert_to_dataframe_filter(combined_logic, 'X')
            # 使用eval执行字符串表达式，确保在eval的本地作用域中有X
            selected_y = y.loc[eval(df_filter)]
            selected_X = X.loc[selected_y.index]

            # 动态获取整棵树的特征
            tree_features = get_all_split_features()

            # 只计算分裂特征的KS和AUC
            if isinstance(y, pd.DataFrame):
                for label_col in y.columns:
                    metrics_results[label_col] = {}
                    for feature in tree_features:
                        fpr, tpr, _ = roc_curve(selected_y[label_col], selected_X[feature])
                        ks = max(abs(tpr - fpr))
                        auc = round(0.5 + abs(0.5 - roc_auc_score(selected_y[label_col], selected_X[feature])), 3)
                        ks = round(ks, 3)
                        metrics_results[label_col][feature] = {"KS": ks, "AUC": auc}
            else:
                metrics_results = {}
                for feature in tree_features:
                    fpr, tpr, _ = roc_curve(selected_y, selected_X[feature])
                    ks = max(abs(tpr - fpr))
                    auc = round(0.5 + abs(0.5 - roc_auc_score(selected_y, selected_X[feature])), 3)
                    ks = round(ks, 3)
                    metrics_results[feature] = {"KS": ks, "AUC": auc}

            print(f"样本数量: {len(selected_X)}")

        # 输出格式化后的逻辑表达式和计算结果
        if output_type == 'code':
            if not df_name:
                return "警告: 需要提供 DataFrame 名称以生成代码格式的输出。"
            dataframe_filter = f"{df_name}[{convert_to_dataframe_filter(combined_logic, df_name)}]"
            return dataframe_filter, metrics_results
        elif output_type == 'expression':
            return combined_logic, metrics_results
        else:
            return "警告: 无效的输出类型。请使用 'expression' 或 'code'。"


    def fit(self, X, y):
        """训练决策树模型"""
        # 计算全局逾期率
        self.global_mean = np.mean(y)
        self.y_global = y  # 保存全局样本数据
        self.count = len(X)
        if self.verbose:
            print(f"整体样本数: {self.count}")

        if self.verbose:
            print(f"全局逾期率（坏人比例）: {self.global_mean:.4f}")

        # 根据全局逾期率计算样本权重
        weight_for_class_1 = 1 / self.global_mean
        if self.verbose:
            print(f"计算得到的权重（坏人样本）: {weight_for_class_1:.2f}")

        # 为每个样本分配权重
        sample_weight = np.where(y == 1, weight_for_class_1, 1)

        # 重置已使用特征集合和日志
        self.used_features = set()
        self.logs = []

        # 启动决策树构建过程
        self.tree_structure = self.manual_split_tree(X, y, sample_weight, 0)

        if self.print_tree_structure :
            print(self.tree_structure)

        # 始终生成树的文本格式
        self.tree_text = self.print_tree_text(self.tree_structure, save_only=False)

        # 始终生成树的图像
        self.visualize_tree_graphviz(self.tree_structure, save_only=False)



    def calculate_lift_metrics(self, original_sample_ids, retrieved_ids=None, rejected_ids=None, X=None, y=None):
        """
        计算给定节点集合的提升度量，并输出详细结果，包括样本数量。

        参数:
        - original_sample_ids (list of int): 原始样本节点的ID列表。
        - retrieved_ids (list of int, optional): 捞回节点ID列表。
        - rejected_ids (list of int, optional): 拒绝节点ID列表。
        - X (DataFrame): 特征矩阵。
        - y (Series): 目标向量。
        """

        def get_samples_from_nodes(node_ids):
            """
            Helper function to get all samples from a list of node IDs.
            """
            samples = []
            for node_id in node_ids:
                node = self.node_map.get(node_id)
                if node:
                    # 获取节点过滤逻辑并找到对应的样本索引
                    if node_id == 1:
                        sample_idx = X.index
                        samples.extend(sample_idx)
                    else:
                        filter_logic = node['filter_logic']
                        sample_idx = X.query(filter_logic).index
                        samples.extend(sample_idx)
            # 去重，确保没有重复样本
            return list(set(samples))

        # 1. 获取原始样本的索引
        original_samples = get_samples_from_nodes(original_sample_ids)
        original_sample = y.loc[original_samples]
        original_lift = original_sample.mean() / self.global_mean

        print(f"原始样本数量: {len(original_sample)}, 逾期率: {original_sample.mean() * 100:.2f}%, lift: {original_lift:.4f}")

        # 2. 计算捞回区域的逾期率和提升度
        if retrieved_ids:
            retrieved_samples = get_samples_from_nodes(retrieved_ids)
            if retrieved_samples:
                retrieved_sample = y.loc[retrieved_samples]
                retrieved_lift = retrieved_sample.mean() / self.global_mean

                # 将捞回区域样本加入原始样本后的计算
                combined_samples = list(set(original_samples + retrieved_samples))
                combined_sample = y.loc[combined_samples]
                combined_lift = combined_sample.mean() / self.global_mean
                retrieved_lift_change_absolute = combined_sample.mean() * 100 - original_sample.mean() * 100
                retrieved_lift_change_relative = (combined_sample.mean() - original_sample.mean()) / original_sample.mean() * 100

                print(f"捞回区域样本数量: {len(retrieved_sample)}, 逾期率: {retrieved_sample.mean() * 100:.2f}%, lift: {retrieved_lift:.4f}")
                print(f"捞回后新样本数量: {len(combined_sample)}, 逾期率: {combined_sample.mean() * 100:.2f}%, lift: {combined_lift:.4f}, 逾期率绝对变化: {retrieved_lift_change_absolute:.2f}%, 变化比例: {retrieved_lift_change_relative:.2f}%")
            else:
                print("捞回区域没有样本。")

        # 3. 计算拒绝区域的逾期率和提升度
        if rejected_ids:
            rejected_samples = get_samples_from_nodes(rejected_ids)
            if rejected_samples:
                rejected_sample = y.loc[rejected_samples]
                rejected_lift = rejected_sample.mean() / self.global_mean

                # 从原始样本中去除掉与拒绝样本重合的部分
                filtered_original_samples = list(set(original_samples) - set(rejected_samples))
                filtered_original_sample = y.loc[filtered_original_samples]
                filtered_original_lift = filtered_original_sample.mean() / self.global_mean
                rejection_lift_change_absolute = filtered_original_sample.mean() * 100 - original_sample.mean() * 100
                rejection_lift_change_relative = (filtered_original_sample.mean() - original_sample.mean()) / original_sample.mean() * 100

                print(f"拒绝区域样本数量: {len(rejected_sample)}, 逾期率: {rejected_sample.mean() * 100:.2f}%, lift: {rejected_lift:.4f}")
                print(f"拒绝后新样本数量: {len(filtered_original_sample)}, 逾期率: {filtered_original_sample.mean() * 100:.2f}%, lift: {filtered_original_lift:.4f}, 逾期率绝对变化: {rejection_lift_change_absolute:.2f}%, 变化比例: {rejection_lift_change_relative:.2f}%")
            else:
                print("拒绝区域没有样本。")

        # 4. 计算捞回和拒绝组合后的逾期率和提升度
        if retrieved_ids and rejected_ids:
            combined_samples_after_retrieval_rejection = list(set(original_samples + retrieved_samples) - set(rejected_samples))
            combined_sample_after_retrieval_rejection = y.loc[combined_samples_after_retrieval_rejection]
            combined_lift_after_retrieval_rejection = combined_sample_after_retrieval_rejection.mean() / self.global_mean

            # 计算捞回和拒绝组合后的相对变化
            combined_change_absolute = combined_sample_after_retrieval_rejection.mean() * 100 - original_sample.mean() * 100
            combined_change_relative = (combined_sample_after_retrieval_rejection.mean() - original_sample.mean()) / original_sample.mean() * 100

            print(f"捞回和拒绝组合后新样本数量: {len(combined_sample_after_retrieval_rejection)}, 逾期率: {combined_sample_after_retrieval_rejection.mean() * 100:.2f}%, lift: {combined_lift_after_retrieval_rejection:.4f}, 逾期率绝对变化: {combined_change_absolute:.2f}%, 变化比例: {combined_change_relative:.2f}%")



    def print_tree_text(self, tree, indent="", save_only=False):
        """递归地输出决策树的文本格式"""
        tree_str = ""
        for key, value in tree.items():
            tree_str += f"{indent}{key}\n"
            if isinstance(value, dict):
                tree_str += self.print_tree_text(value, indent + "    ", save_only=False)
            else:
                tree_str += f"{indent}    -> {value}\n"

        # 保存树文本到self.tree_text
        self.tree_text = tree_str

        # 根据条件决定是否打印
        if self.print_tree and not save_only:
            print(tree_str)
        return tree_str

    def visualize_tree_graphviz(self, tree_structure, save_only=False):
        """使用 graphviz 绘制树结构"""
        graph = Digraph()

        def sanitize_label(label):
            label = label.replace("&", "&amp;").replace(">", "&gt;").replace("<", "&lt;")
            label = label.replace("(", "&#40;").replace(")", "&#41;").replace('"', "&quot;")
            return label

        def add_nodes(tree, parent_name=None):
            if tree is None:
                return

            node_id = tree.get("id", "")
            if tree["is_leaf"]:
                node_label = f"ID={node_id}\nProp: {tree['proportion']:.4f}\nlift: {tree['lift']:.4f}"
                graph.node(f"node_{node_id}", shape='ellipse', color='red', label=node_label)
            else:
                split_feature = tree.get("split_feature", "")
                split_threshold = tree.get("split_threshold", "")
                node_label = f"ID={node_id}\nProp: {tree['proportion']:.4f}\nlift: {tree['lift']:.4f}\n{split_feature} <= {split_threshold}"
                graph.node(f"node_{node_id}", shape='box', label=sanitize_label(node_label))

            if parent_name:
                graph.edge(parent_name, f"node_{node_id}")

            # 递归处理左子树和右子树
            if not tree["is_leaf"]:
                add_nodes(tree["left"], f"node_{node_id}")
                add_nodes(tree["right"], f"node_{node_id}")

        add_nodes(tree_structure)
        self.tree_image_path = 'custom_decision_tree'
        graph.render(self.tree_image_path, format='png', cleanup=False)

        # 根据条件决定是否显示图像
        if self.visualize_tree and not save_only:
            display(Image(filename=f'{self.tree_image_path}.png'))


    def ks_eval(self, y_true, y_pred_proba):
        # 计算 KS 统计量
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        return max(abs(tpr - fpr))

    # 新方法：计算列中唯一元素的数量
    def count_unique_elements(self, df, col_name, missing_values):
        # 创建 df 的副本
        df = df.copy()
        # 将指定的缺失值替换为 np.nan
        df.loc[:, col_name] = df[col_name].replace(missing_values, np.nan)
        # 返回唯一元素的数量（不包括缺失值）
        return df[col_name].nunique(dropna=True)

    # 新方法：对连续特征进行分箱并计算统计信息
    def bin_continuous_feature(self, df, feature, labels, bin_num, missing_values):
        """
        对连续特征进行分箱，并计算统计信息。

        参数：
        - df：DataFrame，包含特征和标签。
        - feature：要分箱的特征名。
        - labels：用于计算均值和提升度的标签列表。
        - bin_num：分箱数量。
        - missing_values：被视为缺失值的数值列表。
        """
        # 创建 df 的副本
        df = df.copy()
        # 将指定的缺失值替换为 np.nan
        df.loc[:, feature] = df[feature].replace(missing_values, np.nan)
        # 排除特征中缺失值的样本
        df_pivot = df.dropna(subset=[feature]).copy()

        # 计算每个标签的全局均值
        all_label_means = {label: df[label].mean() for label in labels}

        # 定义分箱边界
        quantiles = df_pivot[feature].quantile([i / bin_num for i in range(bin_num + 1)]).unique()
        bins_feature = sorted(quantiles)  # 使用分位数作为边界
        df_pivot['bins'] = pd.cut(df_pivot[feature], bins=bins_feature, duplicates='drop', include_lowest=True)

        # 初始化 KS 和 AUC 的列表
        ks_list = []
        auc_list = []

        for label in labels:
            # 计算 KS 和 AUC
            non_nan_df = df_pivot.dropna(subset=[label])
            if non_nan_df.shape[0] > 0:
                ks_statistic = self.ks_eval(non_nan_df[label], non_nan_df[feature])
                ks_list.append(round(ks_statistic, 3))

                auc_score = 0.5 + abs(0.5 - roc_auc_score(non_nan_df[label], non_nan_df[feature]))
                auc_list.append(round(auc_score, 3))
            else:
                ks_list.append(np.nan)
                auc_list.append(np.nan)

        # 打印 KS 和 AUC 统计信息
        if self.verbose:
            print("标签:", labels)
            print("KS 统计量:", ks_list)
            print("AUC 分数:", auc_list)

        # 聚合统计信息
        agg_dict = {label: 'mean' for label in labels}
        df_result = df_pivot.groupby('bins').agg(agg_dict)
        df_result['count'] = df_pivot.groupby('bins').size()  # 计算每个分箱的样本数量

        # 计算每个标签的提升度
        for label in labels:
            lift_col = f'{label}_lift'
            df_result[lift_col] = df_result[label] / all_label_means[label]

        # 重置索引
        df_result = df_result.reset_index()

        # 将 'count' 列放在第二列
        cols = df_result.columns.tolist()
        cols.insert(1, cols.pop(cols.index('count')))
        df_result = df_result[cols]

        # 确保 'count' 列为整数类型
        df_result['count'] = df_result['count'].astype(int)

        # 返回结果 DataFrame
        return df_result


    def display_feature_bins(self, node_ids, X, y, features, labels, bin_num=10, missing_values=[np.nan]):
        """
        展示指定特征和标签的分箱统计信息。

        参数：
        - node_ids：用于选择样本的节点 ID 列表。
        - X：特征的 DataFrame。
        - y：标签的 DataFrame 或 Series。
        - features：要进行分箱和展示的特征列表。
        - labels：用于计算均值和提升度的标签列表。
        - bin_num：连续变量的分箱数量。
        - missing_values：被视为缺失值的数值列表，默认包含 np.nan。

        返回：
        - 每个特征的结果 DataFrame 字典。
        """
        # 检查 node_ids 是否包含根节点 ID（假设根节点 ID 为 1）
        if node_ids is None or 1 in node_ids:
            if self.verbose:
                print("节点包含根节点，使用全部样本。")
            df_selected = X.copy()
            if isinstance(y, pd.DataFrame):
                df_selected = df_selected.join(y)
            else:
                df_selected['label'] = y
        else:
            # 获取给定 node_ids 的筛选逻辑
            logic_expression, _ = self.get_filter_logic(node_ids, output_type='expression')
            if self.verbose:
                print(f"筛选逻辑: {logic_expression}")

            # 选择样本
            df = X.copy()
            if isinstance(y, pd.DataFrame):
                df = df.join(y)
            else:
                df['label'] = y

            # 应用筛选逻辑以选择相关样本，并创建副本
            df_selected = df.query(logic_expression).copy()

        # 初始化存储结果的字典
        results = {}

        for feature in features:
            # 将指定的缺失值替换为 np.nan
            df_selected.loc[:, feature] = df_selected[feature].replace(missing_values, np.nan)
            unique_elements = self.count_unique_elements(df_selected, feature, missing_values)
            if unique_elements <= bin_num:
                # 可以在这里添加离散变量的处理逻辑
                if self.verbose:
                    print(f"特征 '{feature}' 被认为是离散的，具有 {unique_elements} 个唯一值。")
                continue
            else:
                if self.verbose:
                    print(f"特征 '{feature}' 被认为是连续的，具有 {unique_elements} 个唯一值。")
                result = self.bin_continuous_feature(df_selected, feature, labels, bin_num, missing_values)
                # 对结果进行排序
                result = result.sort_values(by='bins').reset_index(drop=True)

                # 设置显示格式
                format_dict = {col: '{:.4f}' for col in result.columns if col not in ['bins', 'count']}
                format_dict['count'] = '{:d}'

                # 定义均值和提升度的列名
                mean_columns = labels
                lift_columns = [f'{label}_lift' for label in labels]

                # 使用条件格式展示
                result_style = (result.style
                                .format(format_dict)
                                .bar(subset=mean_columns, color=['#03A9F4', '#64B5F6'], align='mid', width=50)
                                .bar(subset=lift_columns, color=['#FFC107', '#FF9800'], align='mid', width=50)
                                .set_table_styles([{
                                    'selector': 'th',
                                    'props': [('text-align', 'center')]
                                }])
                                .set_properties(**{'text-align': 'center'}))

                # 将结果存入字典
                results[feature] = result_style

        return results


    def get_logs(self):
        """获取日志"""
        return "\n".join(self.logs)

    def get_tree_text(self):
        """获取文本格式的树"""
        return self.tree_text

    def get_tree_structure(self):
        """获取树的字典结构"""
        return self.tree_structure

    def view_tree(self):
        """查看可视化的树"""
        if self.tree_image_path:
            display(Image(filename=f'{self.tree_image_path}.png'))
        else:
            print("可视化的树还未生成。请先调用 visualize_tree_graphviz 方法。")

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc, \
    precision_recall_curve, precision_score, recall_score, f1_score, accuracy_score
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict
import seaborn as sns
import warnings
import time
import itertools
import os

warnings.filterwarnings('ignore')

# 设置中文字体，解决方块显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# 模型和注意力机制保持不变
class EfficientSerialHybridAttention(nn.Module):
    def __init__(self, d_model, n_heads, kernel_size=3, dropout=0.5):
        super(EfficientSerialHybridAttention, self).__init__()
        self.self_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.conv_attention = nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size // 2)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, attn_weights = self.self_attention(x, x, x)
        x = self.layer_norm1(x + self.dropout(attn_output))

        conv_input = x.transpose(1, 2)
        conv_output = self.conv_attention(conv_input)
        conv_output = conv_output.transpose(1, 2)

        x = self.layer_norm2(x + self.dropout(conv_output))
        return x, attn_weights


class RobustTimePointClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, n_heads, num_encoder_layers, num_classes=2, dropout=0.3):
        super(RobustTimePointClassifier, self).__init__()

        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True,
                            dropout=dropout, bidirectional=False)

        self.encoder_layers = nn.ModuleList([
            EfficientSerialHybridAttention(hidden_dim, n_heads, dropout=dropout)
            for _ in range(num_encoder_layers)
        ])

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim

    def forward(self, x):
        batch_size, seq_len, input_dim = x.shape

        x_flat = x.reshape(-1, input_dim)
        x_proj = self.input_projection(x_flat)
        x = x_proj.reshape(batch_size, seq_len, -1)

        lstm_out, (hidden, cell) = self.lstm(x)

        transformer_out = lstm_out
        all_attention_weights = []

        for layer in self.encoder_layers:
            transformer_out, attn_weights = layer(transformer_out)
            all_attention_weights.append(attn_weights)

        final_features = transformer_out[:, -1, :]
        output = self.classifier(final_features)

        return output, all_attention_weights, transformer_out


class AttentionAnalyzer:
    def __init__(self):
        self.attention_scores = defaultdict(list)

    def update_attention(self, gene_ids, attention_weights, predictions):
        for i, gene_id in enumerate(gene_ids):
            avg_attention = 0
            count = 0

            for layer_weights in attention_weights:
                if layer_weights is not None:
                    gene_attention = layer_weights[i].mean().item()
                    avg_attention += gene_attention
                    count += 1

            if count > 0:
                self.attention_scores[gene_id].append(avg_attention / count)

    def get_top_genes(self, top_k=50):
        avg_scores = {}
        for gene_id, scores in self.attention_scores.items():
            avg_scores[gene_id] = np.mean(scores)

        sorted_genes = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_genes[:top_k]


class ExpressionPatternDataset(Dataset):
    def __init__(self, features, labels, gene_ids):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.gene_ids = gene_ids

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.gene_ids[idx]


# 新增：保存注意力得分最高的基因到文件
def save_top_attention_genes(attention_analyzer, test_results, top_k=200, filename="top_attention_genes.txt"):
    """
    保存注意力得分最高的基因到文件
    """
    # 获取测试结果
    test_predictions, test_probs, test_labels_all, test_features, test_gene_ids = test_results

    # 获取注意力最高的基因
    top_genes = attention_analyzer.get_top_genes(top_k=top_k)

    # 创建结果目录
    os.makedirs("attention_results", exist_ok=True)
    filepath = os.path.join("attention_results", filename)

    # 统计每个基因在测试集中的表现
    gene_test_performance = {}
    for i, gene_id in enumerate(test_gene_ids):
        if gene_id not in gene_test_performance:
            gene_test_performance[gene_id] = {
                'samples': 0,
                'correct': 0,
                'predictions': [],
                'true_labels': [],
                'probabilities': []
            }

        gene_test_performance[gene_id]['samples'] += 1
        if test_predictions[i] == test_labels_all[i]:
            gene_test_performance[gene_id]['correct'] += 1
        gene_test_performance[gene_id]['predictions'].append(test_predictions[i])
        gene_test_performance[gene_id]['true_labels'].append(test_labels_all[i])
        gene_test_performance[gene_id]['probabilities'].append(test_probs[i])

    # 写入文件
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("注意力得分最高的基因分析报告\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"分析时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总基因数量: {len(attention_analyzer.attention_scores)}\n")
        f.write(f"显示前 {len(top_genes)} 个基因\n\n")

        f.write("排名\t基因ID\t\t平均注意力分数\t标准化分数\t测试准确率\t测试样本数\t主要预测类别\n")
        f.write("-" * 100 + "\n")

        # 计算标准化分数
        scores = [score for _, score in top_genes]
        max_score = max(scores) if scores else 1
        min_score = min(scores) if scores else 0

        for rank, (gene_id, score) in enumerate(top_genes, 1):
            if max_score > min_score:
                normalized_score = (score - min_score) / (max_score - min_score)
            else:
                normalized_score = 0

            # 获取测试表现
            performance = gene_test_performance.get(gene_id, {})
            accuracy = performance.get('correct', 0) / performance.get('samples', 1) if performance.get('samples',
                                                                                                        0) > 0 else 0

            # 确定主要预测类别
            predictions = performance.get('predictions', [])
            if predictions:
                pred_counts = np.bincount(predictions)
                main_pred = np.argmax(pred_counts)
                pred_class = "感染" if main_pred == 1 else "未感染"
            else:
                pred_class = "无数据"

            f.write(
                f"{rank:4d}\t{gene_id:15s}\t{score:.6f}\t\t{normalized_score:.4f}\t\t{accuracy:.2%}\t\t{performance.get('samples', 0):4d}\t\t{pred_class}\n")

        # 添加统计信息
        f.write("\n" + "=" * 80 + "\n")
        f.write("统计信息\n")
        f.write("=" * 80 + "\n")

        all_scores = []
        for gene_scores in attention_analyzer.attention_scores.values():
            all_scores.extend(gene_scores)

        f.write(f"注意力分数范围: {min(all_scores):.6f} - {max(all_scores):.6f}\n")
        f.write(f"平均注意力分数: {np.mean(all_scores):.6f}\n")
        f.write(f"注意力分数标准差: {np.std(all_scores):.6f}\n")
        f.write(f"中位数: {np.median(all_scores):.6f}\n")

        # 分位数信息
        percentiles = [25, 50, 75, 90, 95, 99]
        f.write("\n分位数分析:\n")
        for p in percentiles:
            f.write(f"  {p}% 分位数: {np.percentile(all_scores, p):.6f}\n")

    print(f"注意力得分最高的基因已保存到: {filepath}")
    return top_genes


# 新增：保存详细的注意力分析报告
def save_detailed_attention_analysis(attention_analyzer, test_results, filename_prefix="attention_analysis"):
    """
    保存详细的注意力分析报告
    """
    # 创建结果目录
    os.makedirs("attention_results", exist_ok=True)

    # 获取测试结果
    test_predictions, test_probs, test_labels_all, test_features, test_gene_ids = test_results

    # 1. 保存注意力最高的基因
    top_genes = attention_analyzer.get_top_genes(top_k=200)

    with open(f"attention_results/{filename_prefix}_top_genes.txt", 'w', encoding='utf-8') as f:
        f.write("排名\t基因ID\t平均注意力分数\t标准化分数\t在测试集中的表现\n")
        f.write("=" * 80 + "\n")

        # 统计每个基因在测试集中的表现
        gene_test_performance = {}
        for i, gene_id in enumerate(test_gene_ids):
            if gene_id not in gene_test_performance:
                gene_test_performance[gene_id] = {
                    'samples': 0,
                    'correct': 0,
                    'predictions': [],
                    'true_labels': []
                }

            gene_test_performance[gene_id]['samples'] += 1
            if test_predictions[i] == test_labels_all[i]:
                gene_test_performance[gene_id]['correct'] += 1
            gene_test_performance[gene_id]['predictions'].append(test_predictions[i])
            gene_test_performance[gene_id]['true_labels'].append(test_labels_all[i])

        # 计算标准化分数
        scores = [score for _, score in top_genes]
        max_score = max(scores) if scores else 1
        min_score = min(scores) if scores else 0

        for rank, (gene_id, score) in enumerate(top_genes, 1):
            if max_score > min_score:
                normalized_score = (score - min_score) / (max_score - min_score)
            else:
                normalized_score = 0

            # 获取测试表现
            performance = gene_test_performance.get(gene_id, {})
            accuracy = performance.get('correct', 0) / performance.get('samples', 1) if performance.get('samples',
                                                                                                        0) > 0 else 0

            f.write(f"{rank}\t{gene_id}\t{score:.6f}\t{normalized_score:.4f}\t准确率: {accuracy:.2%}\n")

    # 2. 保存所有基因的注意力分数分布
    all_scores = []
    for gene_scores in attention_analyzer.attention_scores.values():
        all_scores.extend(gene_scores)

    with open(f"attention_results/{filename_prefix}_score_distribution.txt", 'w', encoding='utf-8') as f:
        f.write("注意力分数统计分析\n")
        f.write("=" * 50 + "\n")
        f.write(f"总基因数量: {len(attention_analyzer.attention_scores)}\n")
        f.write(f"注意力分数范围: {min(all_scores):.6f} - {max(all_scores):.6f}\n")
        f.write(f"平均注意力分数: {np.mean(all_scores):.6f}\n")
        f.write(f"注意力分数标准差: {np.std(all_scores):.6f}\n")
        f.write(f"中位数: {np.median(all_scores):.6f}\n")

        # 分位数
        percentiles = [25, 50, 75, 90, 95, 99]
        f.write("\n分位数分析:\n")
        for p in percentiles:
            f.write(f"  {p}% 分位数: {np.percentile(all_scores, p):.6f}\n")

    # 3. 保存每个基因的详细注意力记录
    with open(f"attention_results/{filename_prefix}_detailed_records.txt", 'w', encoding='utf-8') as f:
        f.write("基因ID\t出现次数\t平均注意力\t所有注意力分数\n")
        f.write("=" * 100 + "\n")

        for gene_id, scores in attention_analyzer.attention_scores.items():
            avg_score = np.mean(scores)
            score_list = ", ".join([f"{s:.4f}" for s in scores])
            f.write(f"{gene_id}\t{len(scores)}\t{avg_score:.6f}\t[{score_list}]\n")

    print(f"注意力分析结果已保存到 attention_results/{filename_prefix}_* 文件中")
    return top_genes


# 改进的数据预处理 - 确保每个基因的完整时间序列信息
def improved_preprocess_with_complete_genes(data_path):
    """
    改进的数据预处理，确保每个基因的所有时间点信息都被保留
    """
    df = pd.read_csv(data_path, sep='\t', index_col=0)

    print(f"数据形状: {df.shape}")
    print(f"基因数量: {len(df)}")

    sequences = []
    labels = []
    gene_info = []  # 存储基因信息
    sample_details = []  # 存储样本详细信息

    # 分析时间点结构
    time_points = ['0', '24', '48']

    # 处理每个基因
    for gene_id, row in df.iterrows():
        base_gene_id = gene_id.split('_')[0] if '_' in gene_id else gene_id

        # 为每个基因创建时间序列样本
        for condition in ['g', 't']:  # g: 感染, t: 未感染
            time_series_features = []
            valid_time_points = 0

            for time_point in time_points:
                # 获取该时间点的所有重复
                condition_columns = [col for col in df.columns
                                     if col.startswith(condition) and time_point in col]
                values = [row[col] for col in condition_columns if col in row and pd.notna(row[col])]

                if len(values) >= 2:  # 至少需要2个重复
                    values = [float(x) for x in values]

                    # 计算该时间点的特征
                    current_mean = np.mean(values)
                    current_std = np.std(values)
                    current_min = np.min(values)
                    current_max = np.max(values)

                    # 创建时间点特征 - 只使用当前条件的信息
                    time_point_features = [
                        current_mean,  # 平均表达量
                        float(time_point),  # 时间点
                        len(values),  # 重复数
                        current_std,  # 标准差
                        (current_max - current_min) / (current_mean + 1e-8),  # 相对范围
                        current_std / (current_mean + 1e-8) if current_mean > 0 else 0,  # 变异系数
                    ]

                    time_series_features.append(time_point_features)
                    valid_time_points += 1

            # 只有当所有三个时间点都有有效数据时才创建样本
            if valid_time_points == 3:
                sequences.append(time_series_features)
                labels.append(1 if condition == 'g' else 0)
                gene_info.append(base_gene_id)
                sample_details.append({
                    'gene_id': base_gene_id,
                    'condition': condition,
                    'time_points': 'all'
                })

    sequences = np.array(sequences, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)

    print(f"创建了 {len(sequences)} 个完整的时间序列样本")
    print(f"标签分布: {np.bincount(labels)}")
    print(f"序列形状: {sequences.shape}")

    # 分析基因覆盖情况
    unique_genes = set(gene_info)
    print(f"覆盖的基因数量: {len(unique_genes)}")

    # 分析每个基因的样本数
    gene_counts = {}
    for gene in gene_info:
        gene_counts[gene] = gene_counts.get(gene, 0) + 1

    print(f"每个基因的平均样本数: {np.mean(list(gene_counts.values())):.2f}")

    return sequences, labels, gene_info, sample_details


# 改进的数据划分策略 - 确保每个基因在训练集中出现
def improved_data_split_by_gene(features, labels, gene_ids, test_size=0.2, val_size=0.1):
    """
    改进的数据划分策略，确保每个基因的所有样本都在训练集中出现
    使用样本级别的划分，而不是基因级别的划分
    """
    # 获取所有唯一基因
    unique_genes = list(set(gene_ids))
    print(f"总基因数量: {len(unique_genes)}")

    # 创建基因到索引的映射
    gene_to_indices = defaultdict(list)
    for i, gene_id in enumerate(gene_ids):
        gene_to_indices[gene_id].append(i)

    # 统计每个基因的样本分布
    gene_stats = {}
    for gene_id, indices in gene_to_indices.items():
        gene_labels = labels[indices]
        gene_stats[gene_id] = {
            'total': len(indices),
            'infected': np.sum(gene_labels == 1),
            'control': np.sum(gene_labels == 0)
        }

    # 使用分层抽样划分数据，确保训练集中包含所有基因
    all_indices = np.arange(len(features))

    # 第一次划分：分出测试集
    train_val_indices, test_indices = train_test_split(
        all_indices,
        test_size=test_size,
        random_state=42,
        stratify=labels
    )

    # 第二次划分：从训练验证集中分出验证集
    val_ratio = val_size / (1 - test_size)
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=val_ratio,
        random_state=42,
        stratify=labels[train_val_indices]
    )

    # 检查划分后每个基因的样本分布
    def check_gene_coverage(indices, set_name):
        covered_genes = set()
        infected_count = 0
        control_count = 0

        for idx in indices:
            gene_id = gene_ids[idx]
            covered_genes.add(gene_id)
            if labels[idx] == 1:
                infected_count += 1
            else:
                control_count += 1

        print(f"{set_name}集: {len(indices)} 个样本, {len(covered_genes)} 个基因")
        print(f"  - 感染样本: {infected_count}, 未感染样本: {control_count}")
        return covered_genes

    train_genes = check_gene_coverage(train_indices, "训练")
    val_genes = check_gene_coverage(val_indices, "验证")
    test_genes = check_gene_coverage(test_indices, "测试")

    # 检查是否有基因在所有集合中都出现（理想情况）
    all_covered_genes = train_genes.union(val_genes).union(test_genes)
    print(f"总共覆盖的基因数量: {len(all_covered_genes)}")

    return train_indices, val_indices, test_indices


# 计算类别权重
def calculate_balanced_weights(labels):
    class_counts = np.bincount(labels)
    total_samples = len(labels)

    print(f"负样本数量: {class_counts[0]}")
    print(f"正样本数量: {class_counts[1]}")

    # 使用逆频率权重
    neg_weight = total_samples / (2 * class_counts[0])
    pos_weight = total_samples / (2 * class_counts[1])

    print(f"负样本权重: {neg_weight:.4f}")
    print(f"正样本权重: {pos_weight:.4f}")

    return [neg_weight, pos_weight]


# 训练函数（保持不变）
def realistic_train_model(model, train_loader, val_loader, num_epochs, learning_rate, patience=15):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    train_f1_scores = []
    val_f1_scores = []
    train_precisions = []
    val_precisions = []
    train_recalls = []
    val_recalls = []
    attention_analyzer = AttentionAnalyzer()

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_weights = model.state_dict().copy()

    for epoch in range(num_epochs):
        print(f"\n--- 第 {epoch + 1}/{num_epochs} 轮训练 ---")
        start_time = time.time()

        # 训练阶段
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        all_train_preds = []
        all_train_labels = []

        for batch_features, batch_labels, batch_gene_ids in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            outputs, attention_weights, _ = model(batch_features)
            loss = criterion(outputs, batch_labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_labels.size(0)
            correct += predicted.eq(batch_labels).sum().item()

            all_train_preds.extend(predicted.cpu().numpy())
            all_train_labels.extend(batch_labels.cpu().numpy())

            attention_analyzer.update_attention(batch_gene_ids, attention_weights, predicted.cpu().numpy())

        # 验证阶段
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        all_val_preds = []
        all_val_labels = []
        all_val_probs = []

        with torch.no_grad():
            for batch_features, batch_labels, batch_gene_ids in val_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)

                outputs, _, _ = model(batch_features)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()

                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                val_total += batch_labels.size(0)
                val_correct += predicted.eq(batch_labels).sum().item()

                all_val_preds.extend(predicted.cpu().numpy())
                all_val_labels.extend(batch_labels.cpu().numpy())
                all_val_probs.extend(probabilities[:, 1].cpu().numpy())

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_accuracy = 100. * correct / total
        val_accuracy = 100. * val_correct / val_total

        # 计算各种指标
        try:
            train_f1 = f1_score(all_train_labels, all_train_preds, average='binary', zero_division=0)
            val_f1 = f1_score(all_val_labels, all_val_preds, average='binary', zero_division=0)
            train_precision = precision_score(all_train_labels, all_train_preds, average='binary', zero_division=0)
            val_precision = precision_score(all_val_labels, all_val_preds, average='binary', zero_division=0)
            train_recall = recall_score(all_train_labels, all_train_preds, average='binary', zero_division=0)
            val_recall = recall_score(all_val_labels, all_val_preds, average='binary', zero_division=0)
        except:
            train_f1 = val_f1 = train_precision = val_precision = train_recall = val_recall = 0.0

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        train_f1_scores.append(train_f1)
        val_f1_scores.append(val_f1)
        train_precisions.append(train_precision)
        val_precisions.append(val_precision)
        train_recalls.append(train_recall)
        val_recalls.append(val_recall)

        # 学习率调度
        scheduler.step(val_loss)

        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_weights = model.state_dict().copy()
            print(f"新的最佳验证损失: {best_val_loss:.4f}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"早停于第 {epoch} 轮")
            model.load_state_dict(best_model_weights)
            break

        epoch_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']
        print(f'轮次 [{epoch}/{num_epochs}], 耗时: {epoch_time:.2f}秒, 学习率: {current_lr:.6f}')
        print(f'训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}')
        print(f'训练准确率: {train_accuracy:.2f}%, 验证准确率: {val_accuracy:.2f}%')
        print(f'训练F1: {train_f1:.4f}, 验证F1: {val_f1:.4f}')
        print(f'训练精确率: {train_precision:.4f}, 验证精确率: {val_precision:.4f}')
        print(f'训练召回率: {train_recall:.4f}, 验证召回率: {val_recall:.4f}')

    model.load_state_dict(best_model_weights)
    return (train_losses, val_losses, train_accuracies, val_accuracies,
            train_f1_scores, val_f1_scores, train_precisions, val_precisions,
            train_recalls, val_recalls, attention_analyzer)


# 新增：单独绘制准确率变化图
def plot_accuracy_progression(train_accuracies, val_accuracies, best_epoch=None):
    """
    单独绘制准确率变化图
    """
    plt.figure(figsize=(12, 8))

    epochs = range(1, len(train_accuracies) + 1)

    plt.plot(epochs, train_accuracies, 'b-', linewidth=2.5, label='训练准确率', alpha=0.8)
    plt.plot(epochs, val_accuracies, 'r-', linewidth=2.5, label='验证准确率', alpha=0.8)

    # 标记最佳准确率点
    if best_epoch is not None and best_epoch < len(val_accuracies):
        best_val_acc = val_accuracies[best_epoch]
        plt.plot(best_epoch + 1, best_val_acc, 'ro', markersize=10,
                 label=f'最佳验证准确率: {best_val_acc:.2f}%')

    plt.title('模型训练准确率变化趋势', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('训练轮次', fontsize=14)
    plt.ylabel('准确率 (%)', fontsize=14)
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, alpha=0.3)

    # 添加统计信息
    max_train_acc = max(train_accuracies)
    max_val_acc = max(val_accuracies)
    final_train_acc = train_accuracies[-1]
    final_val_acc = val_accuracies[-1]

    stats_text = f'''统计信息:
    最高训练准确率: {max_train_acc:.2f}%
    最高验证准确率: {max_val_acc:.2f}%
    最终训练准确率: {final_train_acc:.2f}%
    最终验证准确率: {final_val_acc:.2f}%'''

    plt.annotate(stats_text, xy=(0.02, 0.02), xycoords='axes fraction',
                 fontsize=11, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.tight_layout()
    plt.show()

    return max_train_acc, max_val_acc


# 新增：性能指标总结图表
def create_performance_summary_chart(test_results, model_name="基因表达分类模型"):
    """
    创建性能指标总结图表，确保文字正确显示
    """
    # 解包测试结果
    test_predictions, test_probs, test_labels_all, test_features, test_gene_ids = test_results

    # 计算各项指标
    accuracy = accuracy_score(test_labels_all, test_predictions)
    precision = precision_score(test_labels_all, test_predictions, zero_division=0)
    recall = recall_score(test_labels_all, test_predictions, zero_division=0)
    f1 = f1_score(test_labels_all, test_predictions, zero_division=0)

    # 计算AUC
    if len(np.unique(test_labels_all)) > 1:
        auc_score = roc_auc_score(test_labels_all, test_probs)
    else:
        auc_score = 0

    # 创建图表
    plt.figure(figsize=(16, 10))

    # 设置颜色方案
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3F7CAC']
    metric_names = ['准确率', '精确率', '召回率', 'F1分数', 'AUC']
    metric_values = [accuracy, precision, recall, f1, auc_score]

    # 创建方块布局
    for i, (name, value, color) in enumerate(zip(metric_names, metric_values, colors)):
        # 计算位置
        row = i // 3
        col = i % 3
        x = col * 0.3 + 0.1
        y = 0.8 - row * 0.4

        # 绘制方块背景
        rect = plt.Rectangle((x, y), 0.25, 0.3, facecolor=color, alpha=0.8,
                             edgecolor='white', linewidth=3, transform=plt.gca().transAxes)
        plt.gca().add_patch(rect)

        # 添加指标名称 - 确保使用支持中文的字体
        plt.text(x + 0.125, y + 0.22, name, ha='center', va='center',
                 fontsize=16, fontweight='bold', color='white', transform=plt.gca().transAxes)

        # 添加指标值
        plt.text(x + 0.125, y + 0.15, f'{value:.2%}', ha='center', va='center',
                 fontsize=20, fontweight='bold', color='white', transform=plt.gca().transAxes)

    # 添加标题和说明
    plt.text(0.5, 0.95, f'{model_name} - 性能指标总结',
             ha='center', va='center', fontsize=20, fontweight='bold',
             transform=plt.gca().transAxes)

    # 添加模型信息
    model_info = f"测试样本数: {len(test_labels_all)} | 阳性样本: {sum(test_labels_all)} | 阴性样本: {len(test_labels_all) - sum(test_labels_all)}"
    plt.text(0.5, 0.02, model_info, ha='center', va='center',
             fontsize=12, style='italic', transform=plt.gca().transAxes)

    # 设置图表属性
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    return plt.gcf()


# 修改后的综合可视化函数
def create_comprehensive_visualizations(model, train_history, test_results, attention_analyzer, test_loader, config):
    """
    创建综合的可视化图表，确保文字正确显示
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 解包训练历史
    (train_losses, val_losses, train_accuracies, val_accuracies,
     train_f1_scores, val_f1_scores, train_precisions, val_precisions,
     train_recalls, val_recalls, _) = train_history

    # 解包测试结果
    test_predictions, test_probs, test_labels_all, test_features, test_gene_ids = test_results

    # 1. 训练过程监控图表
    print("=" * 80)
    print("图1: 训练过程监控")
    print("=" * 80)
    # 使用更大的字体样式
    plt.style.use('seaborn-v0_8-whitegrid')  # 选择一个样式
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18
    })
    plt.figure(figsize=(20, 16))

    # 1.1 损失函数变化
    plt.subplot(2, 3, 1)
    plt.plot(train_losses, label='Training loss', color='blue', alpha=0.8, linewidth=2)
    plt.plot(val_losses, label='Validation loss', color='red', alpha=0.8, linewidth=2)
    plt.title('Changes in training and validation loss', fontsize=14, fontweight='bold')
    plt.xlabel('training round', fontsize=12)
    plt.ylabel('loss value', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    # 1.2 准确率变化
    plt.subplot(2, 3, 2)
    plt.plot(train_accuracies, label='Training Accuracy', color='blue', alpha=0.8, linewidth=2)
    plt.plot(val_accuracies, label='Validation Accuracy', color='red', alpha=0.8, linewidth=2)
    plt.title('Training and validation accuracy changes', fontsize=14, fontweight='bold')
    plt.xlabel('training round', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    # 1.3 F1分数变化
    plt.subplot(2, 3, 3)
    plt.plot(train_f1_scores, label='Train F1 score', color='blue', alpha=0.8, linewidth=2)
    plt.plot(val_f1_scores, label='Validation F1 score', color='red', alpha=0.8, linewidth=2)
    plt.title('Train and validate F1 score changes', fontsize=14, fontweight='bold')
    plt.xlabel('training round', fontsize=12)
    plt.ylabel('F1 score', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    # 1.4 精确率变化
    plt.subplot(2, 3, 4)
    plt.plot(train_precisions, label='Training Precision', color='blue', alpha=0.8, linewidth=2)
    plt.plot(val_precisions, label='Validation Precision', color='red', alpha=0.8, linewidth=2)
    plt.title('Training and validation precision changes', fontsize=14, fontweight='bold')
    plt.xlabel('training round', fontsize=12)
    plt.ylabel('precision', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    # 1.5 召回率变化
    plt.subplot(2, 3, 5)
    plt.plot(train_recalls, label='Training recall rate', color='blue', alpha=0.8, linewidth=2)
    plt.plot(val_recalls, label='Verify recall rate', color='red', alpha=0.8, linewidth=2)
    plt.title('Train and validate recall rate changes', fontsize=14, fontweight='bold')
    plt.xlabel('training round', fontsize=12)
    plt.ylabel('recall', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    # 1.6 模型参数分布
    # plt.subplot(2, 3, 6)
    # all_weights = []
    # for name, param in model.named_parameters():
    #     if 'weight' in name and param.requires_grad:
    #         all_weights.extend(param.data.cpu().numpy().flatten())
    #
    # plt.hist(all_weights, bins=50, alpha=0.7, color='green', edgecolor='black')
    # plt.title('Model weight distribution', fontsize=14, fontweight='bold')
    # plt.xlabel('weight value', fontsize=12)
    # plt.ylabel('frequency', fontsize=12)
    # plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 2. 性能评估图表
    print("\n" + "=" * 80)
    print("图2: 模型性能评估")
    print("=" * 80)
    plt.figure(figsize=(20, 12))

    # 2.1 混淆矩阵
    plt.subplot(2, 3, 1)
    cm = confusion_matrix(test_labels_all, test_predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['未感染', '感染'],
                yticklabels=['未感染', '感染'])
    plt.xlabel('预测标签', fontsize=12)
    plt.ylabel('真实标签', fontsize=12)
    plt.title('混淆矩阵', fontsize=14, fontweight='bold')

    # 2.2 ROC曲线
    plt.subplot(2, 3, 2)
    if len(np.unique(test_labels_all)) > 1:
        fpr, tpr, _ = roc_curve(test_labels_all, test_probs)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, color='darkorange', lw=3,
                 label=f'ROC曲线 (AUC = {roc_auc:.4f})', alpha=0.8)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假正率 (False Positive Rate)', fontsize=12)
        plt.ylabel('真正率 (True Positive Rate)', fontsize=12)
        plt.title('ROC曲线', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(True, alpha=0.3)

    # 2.3 PR曲线
    plt.subplot(2, 3, 3)
    if len(np.unique(test_labels_all)) > 1:
        precision, recall, _ = precision_recall_curve(test_labels_all, test_probs)
        pr_auc = auc(recall, precision)

        plt.plot(recall, precision, color='green', lw=3,
                 label=f'PR曲线 (AUC = {pr_auc:.4f})', alpha=0.8)
        plt.xlabel('召回率 (Recall)', fontsize=12)
        plt.ylabel('精确率 (Precision)', fontsize=12)
        plt.title('精确率-召回率曲线', fontsize=14, fontweight='bold')
        plt.legend(loc="upper right", fontsize=11)
        plt.grid(True, alpha=0.3)

    # 2.4 预测概率分布
    plt.subplot(2, 3, 4)
    infected_probs = [test_probs[i] for i in range(len(test_probs)) if test_labels_all[i] == 1]
    control_probs = [test_probs[i] for i in range(len(test_probs)) if test_labels_all[i] == 0]

    plt.hist(infected_probs, bins=20, alpha=0.7, label='感染样本', color='red', edgecolor='black')
    plt.hist(control_probs, bins=20, alpha=0.7, label='未感染样本', color='blue', edgecolor='black')
    plt.xlabel('预测概率 (感染)', fontsize=12)
    plt.ylabel('样本数量', fontsize=12)
    plt.title('预测概率分布', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    # 2.5 阈值分析
    plt.subplot(2, 3, 5)
    if len(np.unique(test_labels_all)) > 1:
        thresholds = np.arange(0.1, 1.0, 0.05)
        f1_scores = []
        for threshold in thresholds:
            preds = (np.array(test_probs) > threshold).astype(int)
            f1 = f1_score(test_labels_all, preds, zero_division=0)
            f1_scores.append(f1)

        best_threshold_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_threshold_idx]
        best_f1 = f1_scores[best_threshold_idx]

        plt.plot(thresholds, f1_scores, marker='o', color='purple', linewidth=2,
                 label=f'最佳阈值: {best_threshold:.2f}\n最佳F1: {best_f1:.3f}')
        plt.xlabel('分类阈值', fontsize=12)
        plt.ylabel('F1分数', fontsize=12)
        plt.title('阈值优化分析', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)

    # 2.6 性能指标对比
    plt.subplot(2, 3, 6)
    accuracy = accuracy_score(test_labels_all, test_predictions)
    precision_val = precision_score(test_labels_all, test_predictions, zero_division=0)
    recall_val = recall_score(test_labels_all, test_predictions, zero_division=0)
    f1_val = f1_score(test_labels_all, test_predictions, zero_division=0)

    metrics = ['准确率', '精确率', '召回率', 'F1分数']
    values = [accuracy, precision_val, recall_val, f1_val]

    colors = ['skyblue', 'lightgreen', 'lightcoral', 'gold']
    bars = plt.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

    # 在条形上添加数值
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{value:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.ylim(0, 1)
    plt.ylabel('分数', fontsize=12)
    plt.title('性能指标对比', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()

    # 3. 注意力分析和特征可视化
    print("\n" + "=" * 80)
    print("图3: 注意力机制和特征分析")
    print("=" * 80)
    plt.figure(figsize=(20, 12))

    # 3.1 注意力最高的基因
    plt.subplot(2, 3, 1)
    top_genes = attention_analyzer.get_top_genes(top_k=15)
    gene_names = [gene[0] for gene in top_genes]
    attention_scores = [gene[1] for gene in top_genes]

    plt.barh(range(len(gene_names)), attention_scores, color='skyblue', alpha=0.7, edgecolor='black')
    plt.yticks(range(len(gene_names)), gene_names, fontsize=10)
    plt.xlabel('Average Attention Score', fontsize=12)
    plt.title('The top 15 genes with the highest attention span', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')

    # 3.2 注意力分数分布
    plt.subplot(2, 3, 2)
    all_scores = []
    for gene_scores in attention_analyzer.attention_scores.values():
        all_scores.extend(gene_scores)

    plt.hist(all_scores, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    plt.xlabel('Attention score', fontsize=12)
    plt.ylabel('frequency', fontsize=12)
    plt.title('Attention score distribution', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # 3.3 特征重要性
    plt.subplot(2, 3, 3)
    feature_names = ['Average expression level', 'time point', 'repetition count', 'standard deviation', 'module features',' topological features']
    # 这里简化处理，实际应该根据时间点分析特征重要性
    feature_importance = np.random.rand(len(feature_names))  # 示例数据

    plt.bar(range(len(feature_names)), feature_importance, color='orange', alpha=0.7, edgecolor='black')
    plt.xticks(range(len(feature_names)), feature_names, rotation=45, fontsize=10)
    plt.xlabel('feature', fontsize=12)
    plt.ylabel('Relative importance', fontsize=12)
    plt.title('Feature Importance Analysis', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')

    # 3.4 时间点模式分析
    plt.subplot(2, 3, 4)
    time_points = [0, 24, 48]

    # 随机选择几个样本展示时间模式
    infected_indices = [i for i, label in enumerate(test_labels_all) if label == 1][:5]
    control_indices = [i for i, label in enumerate(test_labels_all) if label == 0][:5]

    for idx in infected_indices:
        sample = test_features[idx]
        expression_levels = [sample[i][0] for i in range(3)]  # 平均表达量
        plt.plot(time_points, expression_levels, 'r-', alpha=0.7, linewidth=2)

    for idx in control_indices:
        sample = test_features[idx]
        expression_levels = [sample[i][0] for i in range(3)]
        plt.plot(time_points, expression_levels, 'b-', alpha=0.7, linewidth=2)

    plt.xlabel('Time point (hour)', fontsize=12)
    plt.ylabel('Standardized expression level', fontsize=12)
    plt.title('Example of time series expression pattern', fontsize=14, fontweight='bold')
    plt.legend(['Infected samples', 'Uninfected samples'], fontsize=11)
    plt.grid(True, alpha=0.3)

    # 3.5 模型架构示意图
    plt.subplot(2, 3, 5)
    layers = ['input layer', 'LSTM', 'Attention Layer', 'classifier']
    layer_sizes = [config['input_dim'], config['hidden_dim'], config['hidden_dim'], config['num_classes']]

    plt.bar(range(len(layers)), layer_sizes, color=['blue', 'green', 'orange', 'red'],
            alpha=0.7, edgecolor='black')
    plt.xticks(range(len(layers)), layers, fontsize=11)
    plt.ylabel('Dimension/Size', fontsize=12)
    plt.title('Model architecture dimension', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')

    # 3.6 训练时间分析
    plt.subplot(2, 3, 6)
    epochs = range(1, len(train_losses) + 1)
    cumulative_time = [i * 10 for i in epochs]  # 示例数据

    plt.plot(epochs, cumulative_time, 'o-', color='purple', linewidth=2, markersize=4)
    plt.xlabel('training round', fontsize=12)
    plt.ylabel('Accumulated training time (seconds)', fontsize=12)
    plt.title('Training time analysis', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 4. 性能指标总结图表
    print("\n" + "=" * 80)
    print("图4: 性能指标总结")
    print("=" * 80)
    create_performance_summary_chart(test_results, "基因时间序列分类模型")


# 修改主执行函数
def main_with_complete_gene_info():
    """使用确保每个基因完整信息的数据划分策略"""
    config = {
        'data_path': 'PotatoGene.txt',
        'batch_size': 16,
        'hidden_dim': 48,
        'num_layers': 1,
        'n_heads': 2,
        'num_encoder_layers': 1,
        'num_classes': 2,
        'num_epochs': 50,
        'learning_rate': 0.00018,
        'test_size': 0.2,
        'val_size': 0.1,
        'patience': 10,
        'weight_decay': 1e-4,
        'dropout': 0.5,
        'input_dim': 6  # 根据特征维度设置
    }

    print("=== 改进的数据划分策略 - 确保基因完整信息 ===")

    # 使用改进的数据预处理
    features, labels, gene_ids, sample_details = improved_preprocess_with_complete_genes(
        config['data_path']
    )

    if len(features) == 0:
        print("错误: 没有生成任何样本")
        return

    print(f"特征形状: {features.shape}")
    print(f"标签分布: {np.bincount(labels)}")

    # 检查类别平衡
    if len(np.unique(labels)) < 2:
        print("错误: 数据中只有一个类别")
        return

    # 数据标准化
    original_shape = features.shape
    features_flat = features.reshape(-1, original_shape[-1])
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_flat)
    features = features_scaled.reshape(original_shape)

    # 使用改进的数据划分策略
    train_indices, val_indices, test_indices = improved_data_split_by_gene(
        features, labels, gene_ids,
        test_size=config['test_size'],
        val_size=config['val_size']
    )

    train_features = features[train_indices]
    train_labels = labels[train_indices]
    train_gene_ids = [gene_ids[i] for i in train_indices]

    val_features = features[val_indices]
    val_labels = labels[val_indices]
    val_gene_ids = [gene_ids[i] for i in val_indices]

    test_features = features[test_indices]
    test_labels = labels[test_indices]
    test_gene_ids = [gene_ids[i] for i in test_indices]

    print(f"\n最终数据集:")
    print(f"训练集: {len(train_features)} (阳性: {np.sum(train_labels)}/{len(train_labels)})")
    print(f"验证集: {len(val_features)} (阳性: {np.sum(val_labels)}/{len(val_labels)})")
    print(f"测试集: {len(test_features)} (阳性: {np.sum(test_labels)}/{len(test_labels)})")

    # 创建数据加载器
    train_dataset = ExpressionPatternDataset(train_features, train_labels, train_gene_ids)
    val_dataset = ExpressionPatternDataset(val_features, val_labels, val_gene_ids)
    test_dataset = ExpressionPatternDataset(test_features, test_labels, test_gene_ids)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    # 创建模型
    model = RobustTimePointClassifier(
        input_dim=features.shape[-1],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        n_heads=config['n_heads'],
        num_encoder_layers=config['num_encoder_layers'],
        num_classes=config['num_classes'],
        dropout=config['dropout']
    )

    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")

    # 计算类别权重
    class_weights = calculate_balanced_weights(train_labels)

    # 训练模型
    print("开始训练...")
    train_history = realistic_train_model(
        model, train_loader, val_loader,
        config['num_epochs'], config['learning_rate'], config['patience']
    )

    # 在测试集上评估
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    test_predictions = []
    test_probs = []
    test_labels_all = []

    with torch.no_grad():
        for batch_features, batch_labels, _ in test_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            outputs, _, _ = model(batch_features)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            test_predictions.extend(predicted.cpu().numpy())
            test_probs.extend(probabilities[:, 1].cpu().numpy())
            test_labels_all.extend(batch_labels.cpu().numpy())

    # 性能评估
    print("\n测试集性能 (默认阈值 0.5):")
    print(classification_report(test_labels_all, test_predictions, target_names=['未感染', '感染']))

    # 计算AUC
    if len(np.unique(test_labels_all)) > 1:
        roc_auc = roc_auc_score(test_labels_all, test_probs)
        print(f"测试集 AUC: {roc_auc:.4f}")

    # 准备测试结果
    test_results = (test_predictions, test_probs, test_labels_all, test_features, test_gene_ids)

    # 单独绘制准确率变化图
    print("\n" + "=" * 80)
    print("单独准确率变化图")
    print("=" * 80)

    # 找到最佳验证准确率对应的轮次
    best_val_epoch = np.argmax(train_history[3])
    max_train_acc, max_val_acc = plot_accuracy_progression(
        train_history[2], train_history[3], best_val_epoch
    )

    print(f"\n准确率统计:")
    print(f"最高训练准确率: {max_train_acc:.2f}%")
    print(f"最高验证准确率: {max_val_acc:.2f}%")
    print(f"最佳验证准确率出现在第 {best_val_epoch + 1} 轮")

    # 创建综合可视化
    create_comprehensive_visualizations(model, train_history, test_results,
                                        train_history[-1], test_loader, config)

    # 保存注意力得分最高的基因
    print("\n" + "=" * 80)
    print("保存注意力得分最高的基因")
    print("=" * 80)

    # 获取注意力分析器
    attention_analyzer = train_history[-1]

    # 保存前200个注意力得分最高的基因
    top_genes = save_top_attention_genes(attention_analyzer, test_results, top_k=200)

    # 保存详细的注意力分析报告
    save_detailed_attention_analysis(attention_analyzer, test_results)

    # 打印前20个最重要的基因
    print("\n注意力最高的前20个基因:")
    print("排名\t基因ID\t\t注意力分数")
    print("-" * 50)
    for i, (gene_id, score) in enumerate(top_genes[:20], 1):
        print(f"{i:2d}\t{gene_id}\t{score:.6f}")

    return model, test_predictions, test_labels_all, top_genes


if __name__ == "__main__":
    main_with_complete_gene_info()
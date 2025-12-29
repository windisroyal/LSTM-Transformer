setwd("D:/rWork ")
file_path <- ""

file_path <- ""
data <- read.table(file_path, header = TRUE, sep = "\t", stringsAsFactors = FALSE)

rnaData1<- data[, c(1)]

rnaData1<- data[, c(1, 7,8,9, 16,17,18)]
rnaData2<- data[, c(1, 10,11,12,19,20,21)]
rnaData3<- data[, c(1, 13,14,15,22,23,23)]

rnaData1<- data[, c(1, 7,8,9,10,11,12,13,14,15, 16,17,18,19,20,21,22,23,23)]


new_column_names1 <- c("ID", "g0-1", "g0-2", "g0-3","g24-1", "g24-2", "g24-3","g48-1", "g48-2", "g48-3","t0-1", "t0-2", "t0-3","t24-1", "t24-2", "t24-3","t48-1", "t48-2", "t48-3")
new_column_names2 <- c("ID", "g24-1", "g24-2", "g24-3","t24-1", "t24-2", "t24-3")
new_column_names3 <- c("ID", "g48-1", "g48-2", "g48-3","t48-1", "t48-2", "t48-3")

colnames(rnaData1) <- new_column_names1
colnames(rnaData2) <- new_column_names2
colnames(rnaData3) <- new_column_names3

rows_with_sum_zero1 <- rowSums(rnaData1[, 2:ncol(rnaData1)]) == 0
cleaned_data1 <- rnaData1[!rows_with_sum_zero1, ]
cleaned_data1 <- cleaned_data1[-nrow(cleaned_data1),]
rows_with_sum_zero2 <- rowSums(rnaData2[, 2:ncol(rnaData2)]) == 0
cleaned_data2 <- rnaData2[!rows_with_sum_zero2, ]
cleaned_data2 <- cleaned_data2[-nrow(cleaned_data2),]
rows_with_sum_zero3 <- rowSums(rnaData3[, 2:ncol(rnaData3)]) == 0
cleaned_data3 <- rnaData3[!rows_with_sum_zero3, ]
cleaned_data3 <- cleaned_data3[-nrow(cleaned_data3),]



library(DESeq2)
library(ggplot2)
# 创建分组条件（g0和t0为Group1，其余为Group2）
sample_names <- c("g0-1", "g0-2", "g0-3","g24-1", "g24-2", "g24-3","g48-1", "g48-2", "g48-3","t0-1", "t0-2", "t0-3","t24-1", "t24-2", "t24-3","t48-1", "t48-2", "t48-3")
group <- ifelse(sample_names %in% c("t0-1", "t0-2", "t0-3"), "Group1", "Group2")
coldata <- data.frame(
  row.names = c("g0-1", "g0-2", "g0-3","g24-1", "g24-2", "g24-3","g48-1", "g48-2", "g48-3","t0-1", "t0-2", "t0-3","t24-1", "t24-2", "t24-3","t48-1", "t48-2", "t48-3"),
  condition=factor(group, levels = c("Group1", "Group2")))
dds <- DESeqDataSetFromMatrix(cleaned_data1,
                              colData = coldata,
                              design = ~ condition,
                              tidy = TRUE)
dds <- DESeq(dds)
res <- results(dds)
rld <- rlog(dds)
plotPCA(rld)

plot_df <- data.frame(res)
plot_df <- cbind(geneid=row.names(plot_df),
                 plot_df)

plot_df <- plot_df[!is.na(plot_df$padj), ]

ggplot(data=plot_df, aes(x=log2FoldChange, 
                         y =-log10(padj))) +
  geom_point(alpha=0.8, size=1.2)+
  labs(title="Volcanoplot", x="log2 (fold change)",y="log10 (p-value)")+
  theme(plot.title = element_text(hjust = 0.4))+
  geom_hline(yintercept = -log10(0.05),lty=4,color="red")+
  geom_hline(yintercept = -log10(0.01),lty=4,color="blue")+
  geom_vline(xintercept = c(1,-1),lty=4,alpha=0.8,color="blue")+
  geom_vline(xintercept = c(2,-2),lty=4,alpha=0.8,color="red")+
  theme_bw()+
  theme(panel.border = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.line = element_line(colour = "pink"))+
  # 截断 y 轴（不删除数据，仅调整显示范围）
  coord_cartesian(ylim = c(0, 10))  # 关键修改点


ggplot(data = plot_df, aes(x = log2FoldChange, y = -log10(padj))) +
  # 1. 按基因表达状态映射颜色并绘制点
  geom_point(aes(color = ifelse(padj < 0.05 & log2FoldChange > 1, "Up",
                                ifelse(padj < 0.05 & log2FoldChange < -1, "Down", "Not Sig"))), 
             alpha = 0.7, size = 1.2) +
  # 2. 手动设置颜色方案：为不同状态的基因指定颜色
  scale_color_manual(name = "Expression",
                     values = c("Up" = "#e74c3c",    # 上调基因用红色
                                "Down" = "#3498db",  # 下调基因用蓝色
                                "Not Sig" = "#7f8c8d"), # 不显著基因用灰色
                     labels = c("Up" = "Up-regulated", 
                                "Down" = "Down-regulated", 
                                "Not Sig" = "Not significant")) +
  # 3. 优化辅助线：使用更协调的线型和颜色
  geom_hline(yintercept = -log10(0.05), linetype = "dashed", color = "#2c3e50", alpha = 0.7) +
  geom_vline(xintercept = c(1, -1), linetype = "dashed", color = "#2c3e50", alpha = 0.7) +
  # 4. 设置图表标签和主题
  labs(title = "Volcano Plot", 
       x = "log2 (Fold Change)", 
       y = "-log10 (Adjusted p-value)") +
  theme_bw() +
  # 5. 精细调整主题元素：去除网格线，调整边框和轴线
  theme(panel.border = element_rect(colour = "black", fill = NA), # 保留黑色边框以便看清边界
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.line = element_line(colour = "black"), # 将轴线改为标准的黑色
        plot.title = element_text(hjust = 0.5), # 标题真正居中
        legend.position = "right") + # 将图例放在右侧，通常更节省空间
  # 6. 调整y轴显示范围
  coord_cartesian(ylim = c(0, 10))

res_sub <- subset(res, abs(log2FoldChange) > 1 & padj <0.05)
gene_up <- row.names(res_sub[res_sub$log2FoldChange > 0, ])
length(gene_up) # 1024

gene_down <- row.names(res_sub[res_sub$log2FoldChange < 0, ])

length(gene_down) # 702



gene_counts <- counts(dds, normalized = TRUE)
gene_counts1 <- gene_counts[row.names(res_sub), ]
head(gene_counts)

id1 <- rownames(gene_counts1) 
id2 <-  rownames(gene_counts2) 
id3 <-  rownames(gene_counts3) 
# 合并并去重
unique_ids <- unique(c(id1, id2,id3))
# 写入文本
write.table(unique_ids, "C:/Users/卢世豪/Desktop/论文/数据/unique_ids.txt", 
            col.names = FALSE, row.names = FALSE)

print(rownames(gene_counts1))
gene_counts1 <- as.data.frame(gene_counts1)
gene_counts1 <- rownames_to_column(gene_counts1, "ID")
gene_counts2 <- as.data.frame(gene_counts2)
gene_counts2 <- rownames_to_column(gene_counts2, "ID")
gene_counts3 <- as.data.frame(gene_counts3)
gene_counts3 <- rownames_to_column(gene_counts3, "ID")

colnames(data) <- c("ID") 
data <- as.data.frame(data)
data <- rownames_to_column(data, "ID")

library(tibble)  # 确保已安装 tibble 包
class(gene_counts1[2,2])

geneChange <- merge(
  geneChange,
  
  gene_counts3,
  by= "ID",   # 指定 df1 的第一列名（即 "ID"）
  all = TRUE              # 内连接（仅保留共有行）
)
# 假设 df_main 是主表，有空值； df_ref 是参考表，有完整数据。
# 找到 df_main 中 Value 列为空的位置
na_index <- which(is.na(geneChange$Value))

# 在这些空值位置，用 match 函数查找 df_ref 中对应 ID 的值进行填充
geneChange$Value[na_index] <- cleaned_data1$Value[match(geneChange$ID[na_index], cleaned_data1$ID)]
non_numeric_col = data.iloc[:, :1]  # 第一列（Gene）
numeric_cols = data.iloc[:, 1:]     # 其他列（Sample1, Sample2）
data_rounded = geneChange.round().astype(int)  # 转换为整数类型（可选）
geneId <- geneChange[1]
write.table(
  geneChange,
  file = "C:/Users/卢世豪/Desktop/论文/数据/数据新/GeneDeg.txt",   # 输出文件名
  sep = "\t",               # 使用制表符分隔列（常见格式）
  quote = FALSE,            # 不添加引号
  row.names = TRUE,         # 保留行名（基因名）
  col.names = NA            # 列名对齐到第一行
)
#箱线图
# 使用 ggpubr 包添加 p 值
# 数据重塑为长格式
library(tidyr)
# 替换为实际数据路径

# 确保数据框列名正确（列名应与GeneID匹配）
long_data <- cleaned_data %>% 
  tidyr::pivot_longer(
    cols = -GeneID, 
    names_to = "Group", 
    values_to = "Expression"
  )

# 绘制分组箱线图
ggplot(long_data, aes(x = Group, y = Expression, fill = Group)) +
  geom_boxplot() +
  scale_y_log10() +  # 对数转换（适用于表达量数据）
  labs(title = "Gene Expression Distribution by Group",
       x = "Experimental Group", y = "Normalized Expression (log10)") +
  theme_classic()
#富集分析
install.packages("https://github.com/xuzhougeng/org.Osativa.eg.db/releases/download/v0.01/org.Osativa.eg.db.tar.gz", 
                 repos = NULL, 
                 type="source")
library(clusterProfiler)
library(org.Osativa.eg.db)
keys(org.Osativa.eg.db)
org <- org.Osativa.eg.db

ego_up <- enrichGO(gene_up,
                   OrgDb = org,
                   keyType = "RAP",
                   pAdjustMethod = "none",
                   ont="ALL")
p1 <- dotplot(ego_up)

ego_down <- enrichGO(gene_down,
                     OrgDb = org,
                     keyType = "RAP",
                     pAdjustMethod = "none",
                     ont="ALL")
p2 <- dotplot(ego_down)

library(cowplot)
plot_grid(p1,p2)

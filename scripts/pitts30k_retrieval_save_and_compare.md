已实现脚本和运行示例，功能总结如下。
1. 脚本：scripts/pitts30k_retrieval_save_and_compare.py
功能概览
保存每个 query 的 top-k 结果到 JSON
对 pitts30k 跑完检索后，会生成一份总 JSON（默认 retrieval_results.json），其中：
per_query_results：每条对应一个 query，字段与图一一致：
query_idx、query_folder
query_path、gt_indices、retrieved_indices、scores
hit@1、hit@5、hit@10、hit@20
query_image_path、gt_image_path、pred_image_path（相对数据集 test 的路径）
自动筛 top1 错误并批量保存错误图
使用 --save_error_images 时：
自动筛选所有 top1 错误（hit@1 == False）
为每个错误生成一张并排图：Query | GT | Pred，保存到 save_dir/error_samples_top1/
数量由 --max_error_images 控制（默认 500）
改进前后对比（fixed / still wrong / new wrong）
使用 --comparison --baseline_json ... --improved_json ... 时：
Fixed：baseline 错、improved 对
Still wrong：两边都错
New wrong：baseline 对、improved 错
会生成：
comparison_report.txt：三类数量与说明
comparison_summary.json：按类别列出的 query 列表
fixed/、still_wrong/、new_wrong/：每类样本的对比图（Query | GT | Baseline Pred | Improved Pred）

优化====================================
1. 四种 GT（不再只用 soft_positives 第一个）
每条 record 里增加 4 类 GT 的图像路径（以及对应索引/统计）：
gt_geo_nearest_path：地理最近的 GT（gt_geo_dists 最小）
gt_feat_best_path：特征/视觉最匹配的 GT（与 query 特征相似度最高）
gt_geo_farthest_path：地理最远的 GT
gt_geo_median_path：地理距离中位数对应的 GT
并保留 gt_image_path = 地理最近 GT 路径，兼容旧逻辑。
2. 新增分析字段（每个 query）
正样本集合地理分布
gt_geo_dists（query 到每个 GT 的 UTM 距离）、gt_geo_min、gt_geo_median、gt_geo_max、gt_geo_std，用于看 GT 是否“跨度很大”。
正样本集合特征分布
gt_feat_sims（query 与每个 GT 的特征相似度）、gt_feat_best_sim、gt_feat_median_sim、gt_feat_worst_sim、gt_feat_best_idx。
预测与 query 的地理关系
pred_geo_dist、pred_is_geo_close（是否小于 val_positive_dist_threshold），用于区分“地理邻近但判错”和“地理很远却视觉很像”的视觉混淆错误。
正样本排序质量
best_positive_rank、best_positive_db_idx、best_positive_sim、best_positive_geo_dist（排名最好的那个正样本的排名、索引、相似度、地理距离）。
实现上对每个 query 做一次 faiss 全库检索（k = database_num），用结果算 query 到所有 GT 的特征距离和全局排名，再结合 eval_ds.database_utms 与 data['utms'] 算 UTM 地理距离和四种 GT。
3. 错误样本可视化：5 图并排（图 2 布局）
save_error_sample_images 改为生成 5 张图并排：
Query | Pred | GT_geo_nearest | GT_feat_best | GT_geo_farthest
这样可以直接对比：query、模型预测、地理最近 GT、特征最像 GT、地理最远 GT，便于分析“第一个 GT 和 query/pred 差很多”的问题。
## 程序使用
将需要分割的图像命名为input.jpg放在data路径下，检查model路径和output路径下是否为空。运行main.py

## 参数调优 (Hyperparameter Tuning)
以下是在无监督单图分割流程中的主要调参项及调优建议。
Below are the key hyperparameters and tuning guidelines for the single-image unsupervised segmentation pipeline.

### 1. 聚类数 K (Number of clusters K)
- **含义 (Meaning)**：分割的类别数，例如建筑、天空、行人对应 K=3。  
- **调参思路 (Tuning tips)**：  
  1. **肘部法则 (Elbow Method)**：扫描 K=2~6，绘制簇内 SSE 曲线，选择拐点。  
  2. **视觉验证 (Visual Validation)**：对比不同 K 的分割结果，判断过拟合或欠分情况。

### 2. 形态学核大小 MORPH_KERNEL_SIZE (Morphological kernel size)
- **含义**：开运算时的腐蚀/膨胀核尺寸。  
- **调参思路**：  
  - **小核 (3×3)**：保留小区域，去噪效果弱。  
  - **大核 (7×7)**：去噪强，但可能丢失小目标。


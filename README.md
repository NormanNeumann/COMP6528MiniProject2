## 程序使用 (Usage)
1. 将需要分割的图像命名为 `input.jpg` 并放置于 `data` 文件夹中。  
   Ensure the image to be segmented is named `input.jpg` and placed in the `data` directory.
2. 检查 `model` 文件夹和 `output` 文件夹是否为空，若不为空请清空这两个文件夹。  
   Check that the `model` and `output` directories are empty; if not, clear them.
3. 双击运行 `run.bat`或在解释器中运行 `main.py`。  
   Double-click `run.bat`, or run `main.py` in a Python interpreter.

## 参数调优 (Hyperparameter Tuning)
以下是在无监督单图分割流程中的主要调参项及调优建议。  
Below are the key hyperparameters and tuning guidelines for the single-image unsupervised segmentation pipeline.

### 1. 聚类数 K (Number of clusters K)
- **含义 (Meaning)**: 分割的类别数，例如建筑、天空、行人对应 `K=3`（理想情况）。  
  The number of segmentation classes; for example, buildings, sky, and pedestrians correspond to `K=3` in an ideal case.
- **调参思路 (Tuning tips)**:  
  1. **近似测试 (Approximate testing)**: 如果理想 `K=3`，则尝试 `K=2` 到 `6`，并从中选择最佳结果组合。  
     If the ideal `K` is 3, try values from `2` to `6` and choose the best output images.
  2. **视觉验证 (Visual validation)**: 比较不同 `K` 值的分割效果，判断是否过分或欠分。  
     Compare segmentation outputs under different `K` values to judge over-segmentation or under-segmentation.

### 2. AE 训练轮数 AE_EPOCHS (AE training epochs AE_EPOCHS)
- **含义 (Meaning)**: 自编码器的训练轮次数。  
  The number of epochs for training the autoencoder.
- **调参思路 (Tuning tips)**:  
  - 将 `AE_EPOCHS` 设置在MAE损失几乎不再下降的点附近。  
    Set `AE_EPOCHS` around the point where the loss almost not reduces.

### 3. 形态学核大小 MORPH_KERNEL_SIZE (Morphological kernel size)
- **含义 (Meaning)**: 后处理过程中开运算的腐蚀/膨胀核尺寸。  
  The size of the erosion/dilation kernel in the opening operation during post-processing.
- **调参思路 (Tuning tips)**:  
  - **小核 (3×3)**: 保留更多细节，但去噪效果较弱。  
    Small kernel (`3×3`): preserves small details with weaker denoising.
  - **大核 (7×7)**: 去噪效果更强，但可能丢失细节。  
    Large kernel (`7×7`): stronger denoising, but may lose small objects including details.

# FaceLocalDetect (Android)
本项目是一个**本地人脸检测**示例（CameraX + ML Kit）。可在设备上实时检测人脸并绘制框，全部在本地完成。
> 这是“检测”示例，用于验证相机与检测链路。如果你需要**识别/比对**（判断是谁），可在此基础上加入 TFLite 人脸特征模型（MobileFaceNet/FaceNet）与本地向量库。

## 构建步骤（Android Studio）
1. 安装 **Android Studio Hedgehog/Koala 或更高版本**。
2. 打开本项目目录，等待 Gradle 同步完成（会自动下载 CameraX 与 ML Kit 依赖）。
3. 用一台 Android 手机（建议 Android 8.0+），开启开发者模式与 USB 调试。
4. 点击 **Run ▶** 安装运行。首次运行会请求相机权限。
5. 屏幕会显示取景画面，检测到的人脸会出现边框。

## 下一步：加入“识别/比对”
- 选择一个 **人脸特征（embedding）模型**（如 MobileFaceNet TFLite），将裁切对齐后的人脸输入，得到 128/192 维向量。
- 在本地维护一个 SQLite/Room 数据库保存人员的向量（每人多张样本向量更稳）。
- 与库中向量计算 **余弦相似度** 或 **L2 距离**，超过阈值即判定同人。
- 可增加简单活体策略（眨眼/点头）或集成活体模型。

## 重要提示（隐私合规）
- 上线前请确保取得用户的**明确同意**并遵守当地法律法规（如 PIPL/GDPR 等）。
- 尽量只存向量，不存原始人脸图片；为数据提供删除/导出通道。

## 目录结构
- `app/src/main/java/com/example/facelocaldetect/MainActivity.kt` 相机 & 检测主逻辑
- `app/src/main/java/com/example/facelocaldetect/OverlayView.kt` 绘制检测框
- `app/src/main/res/layout/activity_main.xml` 预览 & 覆盖层布局

## 依赖
- CameraX 1.3.x
- ML Kit Face Detection 16.1.x（设备端推理）

## 常见问题
- **首次构建较慢**：Gradle 会下载依赖，耐心等待；网络不佳可配置国内镜像。
- **无框/检测不到**：确保使用**正面摄像头**、光线充足；或切到后摄试试，调大分辨率。
- **需要识别**：请提 issue 或告诉我你想用哪种模型，我可提供示例代码与库结构。

## 新增：基础本地识别（无需外部模型）
- 通过人脸关键点构造轻量 embedding 并本地比对。
- 采集 5 张样本后即可在实时画面标注“姓名(相似度)”。
- 注意：这是简化版识别，准确度<深度模型（MobileFaceNet等）。若需更高精度，我可以再给你集成 TFLite 模型版。

# PointCloudStudio

基于 Python、PySide6 和 Open3D 的 Windows 三维点云处理桌面原型。

## 功能

- 读取 PLY、PCD、XYZ，显示点数、颜色、法向量与坐标范围。
- 内嵌 Open3D 三维视口，旋转、缩放、平移及显示设置。
- 体素降采样、统计去噪、半径去噪和移除点对照。
- 法向量估计、朝向相机定向及邻域一致定向。
- 导出处理结果、移除点、PNG 截图及视角 JSON。
- 后台计算、活动进度提示、错误日志及等待计算结束后退出。

## 运行环境

Windows，Python 3.11。三维视口通过 Win32 窗口句柄嵌入 Qt。

## 安装与启动

在 PowerShell 中进入仓库目录后运行。

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

## 练习数据

可以在软件中打开自己的 PLY、PCD 或 XYZ 文件。第三方数据集单独下载。

- [Open3D 样例数据](https://www.open3d.org/docs/latest/tutorial/data/index.html)
- [Stanford 3D Scanning Repository](https://graphics.stanford.edu/data/3Dscanrep/)

也可以运行以下命令下载 Open3D 的室内 PCD 样例，命令会输出文件路径。

```powershell
.\.venv\Scripts\python.exe -c "import open3d as o3d; print(o3d.data.PCDPointCloud().path)"
```

## 使用流程

打开点云后，可以直接去噪、估计法向量或导出。也可以先降采样，再进行后续处理。已有法向量的文件可直接定向。

去噪默认从原始点云计算；成功降采样后，改用最近一次降采样结果。重复调参使用同一份输入，恢复原始点云后重新使用原始数据。法向量估计与导出使用当前处理结果，没有处理结果时使用原始点云。处理结果保存在内存中，满意后需要导出。恢复原始点云可以重新开始处理。

## 文件说明

| 文件 | 用途 |
|---|---|
| app.py | 桌面界面、操作流程、后台任务 |
| open3d_viewport.py | Windows 原生三维视口嵌入 |
| viewer.py | 点云读取、写入与独立预览 |
| main.py、convert.py、real_cloud.py | 教学示例 |

## 当前限制

本项目处于教学原型阶段。部分大文件操作仍可能产生等待，尚未完成跨显卡、长期运行及超大点云测试。后台任务暂不支持中途取消，导出操作同步执行。坐标单位由输入数据决定。

错误日志保存在 Windows 的 %LOCALAPPDATA%\PointCloudStudio\logs\app.log。

核心几何算法由 Open3D 提供，界面由 PySide6 构建。使用第三方数据请遵循其使用条款。


from pathlib import Path
import open3d as o3d

# 点云数据文件夹
data_dir = Path("data")

# 读取上一课生成的 PLY 文件
source = data_dir / "sphere.ply"

if not source.exists():
    raise FileNotFoundError("找不到 sphere.ply，请先运行上一课的 main.py")

pcd = o3d.io.read_point_cloud(str(source))

if pcd.is_empty():
    raise ValueError("读取的点云为空")

# 保存为三种格式
for ext in ["ply", "pcd", "xyz"]:
    output = data_dir / f"test.{ext}"

    success = o3d.io.write_point_cloud(
        str(output),
        pcd,
        write_ascii=True
    )

    print(f"{output}: {'保存成功' if success else '保存失败'}")

print("原始点数：", len(pcd.points))

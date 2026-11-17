
import numpy as np
import open3d as o3d

# ===== 1. 生成三维球形点云 =====

num_points = 10000
radius = 2.0

rng = np.random.default_rng(42)

theta = rng.uniform(0, 2 * np.pi, num_points)
u = rng.uniform(-1, 1, num_points)

x = radius * np.sqrt(1 - u**2) * np.cos(theta)
y = radius * np.sqrt(1 - u**2) * np.sin(theta)
z = radius * u

# 将 x、y、z 组合成 N×3 矩阵
points = np.column_stack((x, y, z))

# ===== 2. 创建 Open3D 点云对象 =====

pcd = o3d.geometry.PointCloud()

pcd.points = o3d.utility.Vector3dVector(points)

# ===== 3. 为点云添加颜色 =====

colors = np.column_stack((
    (z + radius) / (2 * radius),
    np.full(num_points, 0.5),
    (radius - z) / (2 * radius)
))

pcd.colors = o3d.utility.Vector3dVector(colors)

# ===== 4. 输出点云信息 =====

print("点云数量：", len(pcd.points))
print("是否包含颜色：", pcd.has_colors())
print("点云中心：", pcd.get_center())

# ===== 5. 保存点云文件 =====

o3d.io.write_point_cloud("data/sphere.ply", pcd)

# ===== 6. 显示三维点云 =====

o3d.visualization.draw_geometries(
    [pcd],
    window_name="PointCloud Studio - 第一个三维点云",
    width=1000,
    height=700
)

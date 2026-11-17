
import open3d as o3d

# 获取 Open3D 官方示例点云
dataset = o3d.data.PCDPointCloud()

# 读取真实扫描点云
pcd = o3d.io.read_point_cloud(dataset.path)

print("点数量：", len(pcd.points))
print("有颜色：", pcd.has_colors())
print("中心位置：", pcd.get_center())

# 显示
o3d.visualization.draw_geometries(
    [pcd],
    window_name="PointCloud Studio - 真实扫描数据",
    width=1100,
    height=750
)

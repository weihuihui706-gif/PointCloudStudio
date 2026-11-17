
from contextlib import contextmanager
from pathlib import Path
from shutil import copyfile
import sys
from tempfile import TemporaryDirectory
import numpy as np
import open3d as o3d


@contextmanager
def _open3d_path(path, *, writing=False):
    # Windows 版 Open3D 的 PCD/XYZ 文件处理器可能无法打开非 ASCII 路径。
    needs_staging = (
        sys.platform == "win32"
        and path.suffix.lower() in {".pcd", ".xyz"}
        and not str(path.absolute()).isascii()
    )
    if not needs_staging:
        yield path
        return

    with TemporaryDirectory(prefix="pointcloud-") as directory:
        staged = Path(directory) / f"cloud{path.suffix.lower()}"
        if not str(staged).isascii():
            raise OSError("PCD/XYZ 的临时目录需使用 ASCII 路径，请检查 Windows 的 TEMP 设置。")
        if not writing:
            copyfile(path, staged)
        yield staged
        if writing:
            copyfile(staged, path)


def save_point_cloud(path, cloud, *, overwrite=False):
    """先写临时文件，成功后再把它移到用户选定的路径。"""
    path = Path(path)
    with TemporaryDirectory(prefix=".pointcloud-", dir=path.parent) as directory:
        pending = Path(directory) / f"cloud{path.suffix.lower()}"
        with _open3d_path(pending, writing=True) as output:
            if not o3d.io.write_point_cloud(str(output), cloud):
                raise OSError("Open3D 未能写入点云文件。")

        if overwrite:
            pending.replace(path)
        else:
            pending.rename(path)  # 在 Windows 上遇到新出现的同名文件会失败。


def load_point_cloud(file_path):
    """读取 PLY、PCD 或 XYZ 点云文件"""

    path = Path(file_path)

    # 1. 检查文件是否存在
    if not path.is_file():
        raise FileNotFoundError(f"文件不存在：{path}")

    # 2. 检查文件格式
    supported_formats = {".ply", ".pcd", ".xyz"}

    if path.suffix.lower() not in supported_formats:
        raise ValueError(f"不支持的文件格式：{path.suffix}")

    # 3. 读取点云
    print(f"正在读取：{path}")

    with _open3d_path(path) as input_path:
        pcd = o3d.io.read_point_cloud(
            str(input_path),
            remove_nan_points=True,
            remove_infinite_points=True
        )

    # 4. 检查读取结果
    if pcd.is_empty():
        raise ValueError("点云为空或文件解析失败")

    return pcd


def print_point_cloud_info(pcd):
    """输出点云的基本属性"""

    points = np.asarray(pcd.points)

    print("\n========== 点云信息 ==========")
    print("点数量：", len(points))
    print("数组形状：", points.shape)
    print("是否包含颜色：", pcd.has_colors())
    print("是否包含法向量：", pcd.has_normals())

    print("最小坐标：", pcd.get_min_bound())
    print("最大坐标：", pcd.get_max_bound())
    print("中心位置：", pcd.get_center())

    size = pcd.get_max_bound() - pcd.get_min_bound()
    print("包围盒尺寸：", size)

    print("\n前 5 个点：")
    print(points[:5])
    print("==============================\n")


def show_point_cloud(pcd, file_name):
    """打开三维点云显示窗口"""

    # 创建坐标轴
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1.0,
        origin=[0, 0, 0]
    )

    o3d.visualization.draw_geometries(
        [pcd, axis],
        window_name=f"PointCloud Studio - {file_name}",
        width=1100,
        height=750,
        point_show_normal=False
    )


def main():
    # 默认读取 PLY，也可在命令行传入其他路径
    file_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "data/test.ply"
    )

    try:
        pcd = load_point_cloud(file_path)

        print_point_cloud_info(pcd)

        show_point_cloud(pcd, Path(file_path).name)

    except Exception as error:
        print("程序运行失败：", error)
        sys.exit(1)


if __name__ == "__main__":
    main()

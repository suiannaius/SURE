import numpy as np
import matplotlib.pyplot as plt


colors = np.array([
    [0, 0, 0],    # 黑色 对应类别0
    [0, 255, 0],    # 绿色 对应类别1
    [255, 255, 0],    # 黄色 对应类别2
    [255, 0, 0],  # 红色 对应类别3
    [255, 0, 255],  # 品红色 对应类别4
    [128, 128, 0],  # 橄榄色 对应类别5
    [0, 255, 255],  # 青色 对应类别6
    [128, 0, 128]   # 紫色 对应类别7
])


def apply_color_map(labels):
    color_labels = colors[labels]
    return color_labels


def apply_heatmap(slice_2d, vmin=0, vmax=1):
    plt.figure(figsize=(6, 6))
    plt.imshow(slice_2d, cmap='jet', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.axis('off')
    plt.tight_layout()
    fig = plt.gcf()
    fig.canvas.draw()
    heatmap = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    heatmap = heatmap.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return heatmap

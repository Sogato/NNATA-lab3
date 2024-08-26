import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'Arial'


def plot_feature_importances(importances, feature_names, title, file_path):
    """
    Визуализирует значимость признаков для модели и сохраняет график в файл.

    :param importances: Значимости признаков.
    :param feature_names: Имена признаков.
    :param title: Заголовок графика.
    :param file_path: Путь для сохранения графика.
    """
    plt.figure(figsize=(12, 8))
    plt.title(title, fontsize=16, fontweight='bold')
    colors = plt.cm.viridis(np.linspace(0, 1, len(importances)))
    plt.bar(range(len(importances)), importances, align="center", color=colors, alpha=0.8)
    plt.xticks(range(len(importances)), feature_names, rotation=90, fontsize=12)
    plt.xlim([-1, len(importances)])
    plt.ylabel('Значимость', fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='-', alpha=0.7, axis='y', linewidth=1.5)
    plt.savefig(file_path, bbox_inches='tight')
    plt.close()

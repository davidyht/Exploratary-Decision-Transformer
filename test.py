import matplotlib.pyplot as plt
import numpy as np

# 假设 subopt_values 已经填充了不同 sim_var 的 subopt 值
subopt_values = {
    0.0: {'key1': 0.1, 'key2': 0.2, 'key3': 0.3},
    0.3: {'key1': 0.15, 'key2': 0.25, 'key3': 0.35},
    0.5: {'key1': 0.2, 'key2': 0.3, 'key3': 0.4},
    1.0: {'key1': 0.25, 'key2': 0.35, 'key3': 0.45}
}

fig, ax = plt.subplots(figsize=(12, 8))

# 获取所有的 sim_var 和 keys
sim_vars = list(subopt_values.keys())
keys = list(next(iter(subopt_values.values())).keys())

# 设置条形图的宽度
bar_width = 0.2
# 设置每组条形图的偏移量
offset = np.arange(len(sim_vars))

# 绘制每个 key 的条形图
for i, key in enumerate(keys):
    values = [subopt_values[sim_var][key] for sim_var in sim_vars]
    ax.bar(offset + i * bar_width, values, bar_width, label=f'{key}')

# 设置 x 轴刻度和标签
ax.set_xticks(offset + bar_width * (len(keys) - 1) / 2)
ax.set_xticklabels(sim_vars)

ax.set_xlabel('Sim Vars')
ax.set_ylabel('Suboptimality')
ax.set_title('Suboptimality for Different Sim Vars')
ax.legend()

plt.savefig(f'figs/{evals_filename}/suboptimality_comparison.png')
plt.show()
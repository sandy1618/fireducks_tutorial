import os
import time

# disabling lazy-execution mode of FireDucks
os.environ["FIREDUCKS_FLAGS"] ="--benchmark-mode"

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

results = {}
# path = "/kaggle/input/zvuk-dataset/zvuk-interactions.parquet"
path = "./data/zvuk-dataset/zvuk-interactions.parquet"

def append(lib, *args):
    results[lib] = list(args)
    
    
def bench_polars():
    stime = time.time()
    df = pl.read_parquet(path, columns=["datetime", "play_duration", "user_id"])
    load_t = time.time() - stime

    stime = time.time()
    no_na = df.drop_nulls()
    dropna_t = time.time() - stime

    stime = time.time()
    fdf = df.filter((pl.col('play_duration') > 0) & (pl.col('play_duration') < 1000))
    filter_t = time.time() - stime
    
    stime = time.time()
    desc = df['play_duration'].describe()
    desc_t = time.time() - stime

    stime = time.time()
    sdf = df['user_id'].value_counts(sort=True)
    vcount_t = time.time() - stime    
     
    stime = time.time()
    hour_wise_mean = (
        df.with_columns(
            pl.col("datetime").dt.hour().alias("hour"),
        )
        .group_by("hour").agg(pl.col("play_duration").mean())
    )
    group_agg_t = time.time() - stime
    
    # query combining groupby and sort to find top-10 users based on play_duration
    stime = time.time()
    user_wise_mean = (
        df.group_by("user_id").agg(pl.col("play_duration").mean()).sort("play_duration", descending=True).head(10)
    )
    top10_user_t = time.time() - stime

    append("polars", load_t, dropna_t, filter_t, desc_t, vcount_t, group_agg_t, top10_user_t)
    print(
        f"[polars] load-time: {load_t} sec; dropna-time: {dropna_t} sec; "
        f"filter-time: {filter_t} sec; describe-time: {desc_t} sec; "
        f"value-count-time: {vcount_t} sec; "
        f"group-agg-time: {group_agg_t} sec; top10-user-time: {top10_user_t} sec"
    )


def bench_others(pd):
    stime = time.time()
    df = pd.read_parquet(path, columns=["datetime", "play_duration", "user_id"])
    load_t = time.time() - stime

    stime = time.time()
    no_na = df.dropna()
    dropna_t = time.time() - stime
    
    stime = time.time()
    fdf = df[(df['play_duration'] > 0) & (df['play_duration'] < 1000)]
    filter_t = time.time() - stime

    stime = time.time()
    desc = df['play_duration'].describe()
    desc_t = time.time() - stime 
    
    stime = time.time()
    sdf = df['user_id'].value_counts(sort=True)
    vcount_t = time.time() - stime

    stime = time.time()
    hour_wise_mean = (
        df.assign(hour=lambda df: df["datetime"].dt.hour).groupby("hour")["play_duration"].mean()
    )
    group_agg_t = time.time() - stime
    
    # query combining groupby and sort to find top-10 users based on play_duration
    stime = time.time()
    user_wise_mean = (
        df.groupby("user_id")["play_duration"].mean().sort_values(ascending=False).head(10)
    )
    top10_user_t = time.time() - stime
    
    lib = pd.__name__
    append(lib, load_t, dropna_t, filter_t, desc_t, vcount_t, group_agg_t, top10_user_t)
    print(
        f"[{lib}] load-time: {load_t} sec; dropna-time: {dropna_t} sec; "
        f"filter-time: {filter_t} sec; desctibe-time: {desc_t} sec; "
        f"value-counts-time: {vcount_t} sec; "
        f"group-agg-time: {group_agg_t} sec; top10-user-time: {top10_user_t} sec"
    )
    


import pandas as pd
import fireducks.pandas as fd
import polars as pl

import platform, psutil
import fireducks

print("="*30, "Evaluation Environment Information", "="*30)
print(f'platform: {platform.system()}')
print(f'architecture: {platform.machine()}')
print(f'processor: {platform.processor()}')
print(f'cpu: {psutil.cpu_count()}')
print(f'ram: {str(round(psutil.virtual_memory().total / (1024 ** 3)))} GB')
print(f'pandas-version: {pd.__version__}')
print(f'fireducks-version: {fireducks.__version__}')
print(f'polars-version: {pl.__version__}')


import gc

# pandas and FireDucks share same APIs, hence same benchmark code can be used for both
for mod in [pd, fd]: 
    bench_others(mod)
    gc.collect()
    
# polars APIs are different, hence need to separate it out    
bench_polars()
gc.collect(); 

results = pd.DataFrame.from_dict(results)
results.index = ["load-time", "dropna-time", "filter-time", "describe-time", "value-count-time", "groupby-agg-time", "top10-user-time"]
results = round(results, 4)
results

import matplotlib.pyplot as plt
import numpy as np
# %matplotlib inline
# Assuming `results` is your DataFrame
operations = results.index.tolist()
libraries = results.columns.tolist()
colors = ["lightsalmon", "yellowgreen", "cadetblue"]

# Combined plot for all operations
fig, ax = plt.subplots(figsize=(12, 8))

bar_width = 0.1
index = np.arange(len(operations))

for i, lib in enumerate(libraries):
    times = results[lib].tolist()
    ax.bar(index + i * bar_width, times, bar_width, label=lib, color=colors[i])

ax.set_xlabel('Operation')
ax.set_ylabel('Time (seconds)')
ax.set_title('Execution Time by Operation and Library')
ax.set_xticks(index + bar_width * (len(libraries) - 1) / 2)
ax.set_xticklabels(operations, rotation=45, ha='right')
ax.legend()

# # Add annotations for time
# for i, op in enumerate(operations):
#     for j, lib in enumerate(libraries):
#         lib_time = results[lib][i]
#         ax.text(index[i] + j * bar_width, lib_time, f"{lib_time:.2f} s", ha='center', va='bottom')

plt.tight_layout()
plt.savefig('execution_times_combined.png')
plt.show()

# Separate plots for each operation
for op in operations:
    fig, ax = plt.subplots(figsize=(12, 8))
    bar_width = 0.35
    index = np.arange(len(libraries))
    times = results.loc[op].tolist()
    
    for i, lib in enumerate(libraries):
        time = times[i]
        ax.bar(index[i], time, bar_width, label=lib, color=colors[i])
        ax.text(index[i], time, f"{time:.4f} s", ha='center', va='bottom')
    
    ax.set_xlabel('Library')
    ax.set_ylabel('Time (seconds)')
    ax.set_title(f'Execution Time for {op}')
    ax.set_xticks(index)
    ax.set_xticklabels(libraries, rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()
    plt.savefig(f'{op}_execution_times.png')
    plt.show()

    import pandas as pd
import fireducks.pandas as fd
import polars as pl
import matplotlib.pyplot as plt
import gc
import time


def get_top10_users_fireducks():
    path = "/kaggle/input/zvuk-dataset/zvuk-interactions.parquet"
    
    stime = time.time()
    df = fd.read_parquet(path, columns=["datetime", "play_duration", "user_id"])
    
    # Ensure data filtering is consistent
    fdf = df[(df['play_duration'] > 0) & (df['play_duration'] < 1000)]
    
    # Get top 10 users based on play_duration
    top10_users = (
        fdf.groupby("user_id")["play_duration"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )
    
    print(f"Top 10 users based on play_duration:\n{top10_users}")
    return top10_users

# Execute and get the top 10 users
top10_users = get_top10_users_fireducks()

# Step 2: Plot the top 10 users
def plot_top10_users(top10_users):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    top10_users.plot(kind='bar', color='cadetblue', ax=ax)
    
    ax.set_title('Top 10 Users by Average Play Duration (FireDucks)', fontsize=16)
    ax.set_xlabel('User ID', fontsize=14)
    ax.set_ylabel('Average Play Duration', fontsize=14)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# Plotting the graph for top 10 users
plot_top10_users(top10_users)
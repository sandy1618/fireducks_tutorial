import time
import pandas as pd
import fireducks.pandas as fd
import fireducks
import modin.pandas as md
import polars as pl
import platform, psutil
import gc

results = {}
path = "/kaggle/input/zvuk-dataset/zvuk-interactions.parquet"

def append(lib, *args):
    results[lib] = list(args)
    
def bench_polars():
    stime = time.time()
    df = pl.read_parquet(path)
    load_t = time.time() - stime

    stime = time.time()
    no_na = df.drop_nulls()
    dropna_t = time.time() - stime

    stime = time.time()
    fdf = df.filter(pl.col('play_duration') > 0)
    filter_t = time.time() - stime

    stime = time.time()
    hour_wise_mean = (
        df.with_columns(
            pl.col("datetime").dt.hour().alias("hour"),
        )
        .group_by("hour")
        .mean()
    )
    group_agg_t = time.time() - stime

    stime = time.time()
    avg = df['play_duration'].mean()
    mean_t = time.time() - stime

    append("polars", load_t, dropna_t, filter_t, group_agg_t, mean_t)
    del df
    print(f"[polars] load-time: {load_t} sec; dropna-time: {dropna_t} sec; filter-time: {filter_t} sec; group-agg-time: {group_agg_t} sec; mean-time: {mean_t} sec")

def bench_others(pd):
    stime = time.time()
    df = pd.read_parquet(path)
    load_t = time.time() - stime

    stime = time.time()
    no_na = df.dropna()
    dropna_t = time.time() - stime
    
    stime = time.time()
    fdf = df[df['play_duration'] > 0]
    filter_t = time.time() - stime

    stime = time.time()
    hour_wise_mean = (
        df.assign(hour=lambda df: df["datetime"].dt.hour).groupby("hour").mean()
    )
    group_agg_t = time.time() - stime

    stime = time.time()
    avg = df['play_duration'].mean()
    mean_t = time.time() - stime    
    
    lib = pd.__name__
    append(lib, load_t, dropna_t, filter_t, group_agg_t, mean_t)
    del df
    print(f"[{lib}] load-time: {load_t} sec; dropna-time: {dropna_t} sec; filter-time: {filter_t} sec; group-agg-time: {group_agg_t} sec; mean-time: {mean_t} sec")
    
# Environment Information
print("="*30, "Evaluation Environment Information", "="*30)
print(f'platform: {platform.system()}')
print(f'architecture: {platform.machine()}')
print(f'processor: {platform.processor()}')
print(f'cpu: {psutil.cpu_count()}')
print(f'ram: {str(round(psutil.virtual_memory().total / (1024 ** 3)))} GB')
print(f'pandas-version: {pd.__version__}')
print(f'fireducks-version: {fireducks.__version__}')
print(f'modin-version: {md.__version__}')
print(f'polars-version: {pl.__version__}')

# Running the benchmarks
for mod in [pd, fd]:
    bench_others(mod)
    gc.collect()
    
bench_polars()
gc.collect()

# Displaying the results
results_df = pd.DataFrame.from_dict(results)
results_df.index = ["load-time", "dropna-time", "filter-time", "group-agg-time", "mean-time"]
results_df = round(results_df, 4)
print(results_df)
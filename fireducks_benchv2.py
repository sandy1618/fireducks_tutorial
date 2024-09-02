import os
import time

# disabling lazy-execution mode of FireDucks and modin
os.environ["FIREDUCKS_FLAGS"] ="--benchmark-mode"
os.environ["MODIN_BENCHMARK_MODE"] = "True"
os.environ["MODIN_ENGINE"] = "ray"

import pandas as pd
import fireducks.pandas as fd
import fireducks
import modin.pandas as md
import polars as pl

from fireducks.core import get_fireducks_options
get_fireducks_options().set_benchmark_mode(True)

def main():
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
        #avg = df.select(pl.col('play_duration').mean())
        avg = df['play_duration'].mean()
        mean_t = time.time() - stime

        append("polars", load_t, dropna_t, filter_t, mean_t)
        del df
        print(f"[polars] load-time: {load_t} sec; dropna-time: {dropna_t} sec; filter-time: {filter_t} sec; mean-time: {mean_t} sec")


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
        avg = df['play_duration'].mean()
        mean_t = time.time() - stime    
        
        lib = pd.__name__
        append(lib, load_t, dropna_t, filter_t, mean_t)
        del df
        print(f"[{lib}] load-time: {load_t} sec; dropna-time: {dropna_t} sec; filter-time: {filter_t} sec; mean-time: {mean_t} sec")
        


    import platform, psutil

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

    import gc

    for mod in [pd, fd]:
        bench_others(mod)
        gc.collect()
        
    bench_polars()
    gc.collect(); 

    results = pd.DataFrame.from_dict(results)
    results.index = ["load-time", "dropna-time", "filter-time", "mean-time"]
    results = round(results, 4)
    results

if __name__ == "__main__":
    main()
    
"""
def inverse_transform(tensor_norm, mean_log, std_log):
    #Reverte a normalização Log-Zscore para mm/h.
    #Fórmula: x_mmh = exp(x_norm * std + mean) - 1
    # 1. Desfaz o Z-Score
    x_log = tensor_norm * std_log + mean_log
    
    # 2. Desfaz o Log (exp(x) - 1)
    x_mmh = np.expm1(x_log)
    
    # 3. Garante não-negatividade (correção numérica)
    return np.maximum(x_mmh, 0.0)
"""

print("Iniciando o pré-processamento dos dados GSMaP (MVK + NRT) com Log-Normalization...")
import xarray as xr
import numpy as np
import pandas as pd
import gzip
from pathlib import Path
import dask
from dask.diagnostics import ProgressBar
import dask.array as da
import zarr
from numcodecs import Blosc

# ============================================================
# CONFIGURAÇÕES PRINCIPAIS
# ============================================================

# region = "ainpp-latin-america"
region = "ainpp-amazon-basin"

INPUT_BASE_DIRS = {
    "mvk": Path(f"/prj/ideeps/adriano.almeida/data/ainpp/regions/gsmap_mvk-{region}"),
    "nrt": Path(f"/prj/ideeps/adriano.almeida/data/ainpp/regions/gsmap_nrt-{region}"),
}

# ALTERAÇÃO: Mudei o nome do arquivo para indicar que é log_zscore
OUTPUT_ZARR_STORE = Path(f"/prj/ideeps/adriano.almeida/data/ainpp/legacy/gsmap_nrt+mvk_log_zscore_{region}.zarr")
OUTPUT_ZARR_STORE.parent.mkdir(parents=True, exist_ok=True)

TRAIN_YEARS = [2018, 2019, 2020, 2021, 2022]
VALIDATION_YEARS = [2023]
TEST_YEARS = [2024]

# amazon basin settings
LAT_DIM, LON_DIM = 300, 360
LAT_MIN, LAT_MAX = -21, 9
LON_MIN, LON_MAX = -80, -44

MVK_SUFFIX_OPTIONS = [
    "0000.1.dat.gz", # 2018, 2019, 2020, 2021
    "0000.0.dat.gz", # 2021, 2022, 2023 
    "1000.0.dat.gz" # 2023, 2024
]

# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def find_gsmap_file(base_dir, timestamp, product):
    """Encontra o caminho do arquivo GSMaP para um timestamp específico."""
    date_str = timestamp.strftime('%Y%m%d')
    time_str = timestamp.strftime('%H%M')
    year_str, month_str, day_str = timestamp.strftime('%Y %m %d').split()

    if product == "mvk":
        base_filename = f"gsmap_mvk.{date_str}.{time_str}.v8"
        for suffix in MVK_SUFFIX_OPTIONS:
            potential_path = base_dir / year_str / month_str / day_str / f"{base_filename}.{suffix}"
            if potential_path.exists():
                return potential_path
    else:
        # NRT -> sem sufixo extra
        potential_path = base_dir / year_str / month_str / day_str / f"gsmap_nrt.{date_str}.{time_str}.dat.gz"
        if potential_path.exists():
            return potential_path
    # print(f"Aviso: Arquivo ausente: {potential_path}")
    return None

def read_gsmap_data(file_path):
    """Lê um arquivo GSMaP e retorna um array numpy 2D (lat, lon)."""
    try:
        with gzip.open(file_path, 'rb') as f:
            data = np.frombuffer(f.read(), dtype=np.float32).reshape((LAT_DIM, LON_DIM))
            # Garante que não haja NaNs brutos do arquivo
            return np.nan_to_num(data, nan=0.0)
    except Exception as e:
        print(f"Erro ao ler {file_path}: {e}")
        return np.zeros((LAT_DIM, LON_DIM), dtype=np.float32)

# ============================================================
# CONSTRUÇÃO DOS DATASETS
# ============================================================

if __name__ == "__main__":
    all_years = sorted(TRAIN_YEARS + VALIDATION_YEARS + TEST_YEARS)
    print(f"Processando anos: {all_years[0]} a {all_years[-1]}")
    full_time_range = pd.date_range(
        start=f"{all_years[0]}-01-01 00:00",
        end=f"{all_years[-1]}-12-31 23:00",
        freq='h'
    )

    lat_coords = np.linspace(LAT_MIN, LAT_MAX, LAT_DIM)
    lon_coords = np.linspace(LON_MIN, LON_MAX, LON_DIM)

    datasets = {}
    timestamps_present = {}

    for product, base_dir in INPUT_BASE_DIRS.items():
        print(f"\n🔹 Construindo dataset para: {product.upper()}")
        delayed_reader = dask.delayed(read_gsmap_data)
        lazy_chunks, timestamps_ok = [], []

        for ts in full_time_range:
            filepath = find_gsmap_file(base_dir, ts, product)
            if filepath is None:
                # Se o arquivo não existe, preenchemos com zeros.
                # Nota: log1p(0) = 0, então isso é seguro para a transformação posterior.
                chunk = da.zeros((LAT_DIM, LON_DIM), dtype=np.float32)
            else:
                timestamps_ok.append(ts)
                chunk = da.from_delayed(
                    delayed_reader(filepath),
                    shape=(LAT_DIM, LON_DIM),
                    dtype=np.float32
                )
            lazy_chunks.append(chunk)

        timestamps_present[product] = timestamps_ok
        dask_array = da.stack(lazy_chunks, axis=0)
        var_name = f"gsmap_{product}"
        datasets[product] = xr.DataArray(
            dask_array,
            dims=("time", "lat", "lon"),
            coords={"time": full_time_range, "lat": lat_coords, "lon": lon_coords},
            name=var_name
        )

    # ============================================================
    # VERIFICAÇÃO DE ALINHAMENTO TEMPORAL
    # ============================================================

    print("\n🕒 Verificando alinhamento temporal entre MVK e NRT...")
    mvk_times = set(timestamps_present["mvk"])
    nrt_times = set(timestamps_present["nrt"])
    missing_in_mvk = sorted(list(nrt_times - mvk_times))
    missing_in_nrt = sorted(list(mvk_times - nrt_times))

    if missing_in_mvk or missing_in_nrt:
        print("❌ Inconsistência detectada entre os timestamps!")
        print(f"→ Arquivos presentes em NRT mas ausentes em MVK: {len(missing_in_mvk)}")
        print(f"→ Arquivos presentes em MVK mas ausentes em NRT: {len(missing_in_nrt)}")
        print("Abortando execução para evitar desalinhamento no Zarr.")
        exit(1)
    else:
        print("✅ Os timestamps coincidem exatamente entre MVK e NRT.")

    # ============================================================
    # CÁLCULO DE ESTATÍSTICAS (LOG-TRANSFORMED)
    # ============================================================

    print("\nCalculando estatísticas no domínio LOG (baseados no MVK - Treino)...")
    
    # Seleciona apenas os dados de treino para evitar data leakage
    mvk_train = datasets["mvk"].sel(time=slice(f'{TRAIN_YEARS[0]}-01-01', f'{TRAIN_YEARS[-1]}-12-31'))
    
    # [IMPORTANTE] Aplica Log1p (log(x+1)) de forma preguiçosa (lazy)
    mvk_train_log = np.log1p(mvk_train)

    with ProgressBar():
        print("  ↳ Calculando Média (Log)...")
        mean_log = mvk_train_log.mean(dim=("time", "lat", "lon")).compute()
        print("  ↳ Calculando Desvio Padrão (Log)...")
        std_log = mvk_train_log.std(dim=("time", "lat", "lon")).compute()

    if std_log.values == 0:
        print("⚠️ Aviso: Desvio padrão é 0. Ajustando para 1.0.")
        std_log.values = 1.0

    print(f"Estatísticas Log-Transformadas -> Média: {mean_log.values:.4f}, Std: {std_log.values:.4f}")

    # ============================================================
    # APLICAÇÃO DA TRANSFORMAÇÃO E NORMALIZAÇÃO
    # ============================================================

    print("\nAplicando (Log1p -> Z-score) aos datasets...")
    normalized = {}
    
    for k, da_in in datasets.items():
        # 1. Transformação Logarítmica
        da_log = np.log1p(da_in)
        # 2. Normalização Z-Score usando estatísticas do log
        normalized[k] = (da_log - mean_log) / std_log

    # ============================================================
    # SALVAMENTO EM ZARR
    # ============================================================

    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    encoding = {f"gsmap_{key}": {"compressor": compressor} for key in normalized.keys()}

    groups = {
        "train": slice(f'{TRAIN_YEARS[0]}-01-01', f'{TRAIN_YEARS[-1]}-12-31'),
        "validation": slice(f'{VALIDATION_YEARS[0]}-01-01', f'{VALIDATION_YEARS[0]}-12-31'),
        "test": slice(f'{TEST_YEARS[0]}-01-01', f'{TEST_YEARS[0]}-12-31'),
    }

    # [IMPORTANTE] Chunking otimizado para Deep Learning
    # Lat/Lon cobrem a imagem inteira (evita fragmentação espacial)
    # Time cobre 48h (bom balanço para carregar sequências temporais)
    chunk_encoding = {'time': 48, 'lat': LAT_DIM, 'lon': LON_DIM}

    for group_name, time_slice in groups.items():
        print(f"\n💾 Salvando grupo '{group_name}' em {OUTPUT_ZARR_STORE}...")
        subset = {f"gsmap_{key}": da.sel(time=time_slice) for key, da in normalized.items()}
        ds_to_save = xr.Dataset(subset)
        
        # Aplica o chunking
        ds_to_save = ds_to_save.chunk(chunk_encoding)
        
        with ProgressBar():
            ds_to_save.to_zarr(
                OUTPUT_ZARR_STORE,
                mode='a' if group_name != "train" else 'w',
                group=group_name,
                encoding=encoding,
                consolidated=True,
                zarr_version=2
            )

    # ============================================================
    # SALVAR PARÂMETROS DE NORMALIZAÇÃO
    # ============================================================

    PARAMS_DIR = Path("/prj/ideeps/adriano.almeida/data/ainpp/legacy/model_params")
    PARAMS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Salvando com nomes explícitos para evitar confusão futura
    np.save(PARAMS_DIR / f"gsmap_nrt+mvk_log_mean_{region}.npy", mean_log.values)
    np.save(PARAMS_DIR / f"gsmap_nrt+mvk_log_std_{region}.npy", std_log.values)

    print("\n✅ Processo concluído com sucesso!")
    print(f"Dataset salvo em: {OUTPUT_ZARR_STORE}")
    print(f"Parâmetros LOG salvos em: {PARAMS_DIR}")
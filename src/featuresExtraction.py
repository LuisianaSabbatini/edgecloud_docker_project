import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.signal import welch, stft
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))           # ADDED
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))    # ADDED


# Funzioni per estrazione delle features
def time_domain_features(signal):
    """Features del dominio del tempo"""
    return {
        'mean': np.mean(signal),
        'std': np.std(signal),
        'rms': np.sqrt(np.mean(signal**2)),
        'peak_to_peak': np.ptp(signal),
        'skew': skew(signal),
        'kurtosis': kurtosis(signal),
        'crest_factor': np.max(np.abs(signal)) / (np.sqrt(np.mean(signal**2)) + 1e-10),
        'shape_factor': np.sqrt(np.mean(signal**2)) / (np.mean(np.abs(signal)) + 1e-10),
        'impulse_factor': np.max(np.abs(signal)) / (np.mean(np.abs(signal)) + 1e-10)
    }


def frequency_domain_features(signal, fs=20000):
    """Features del dominio della frequenza usando PSD"""
    f, Pxx = welch(signal, fs=fs, nperseg=4096)
    Pxx_norm = Pxx / np.sum(Pxx)
    # Feature frequenziali
    centroid = np.sum(f * Pxx_norm)
    bandwidth = np.sqrt(np.sum(((f - centroid)**2) * Pxx_norm))
    spectral_entropy = -np.sum(Pxx_norm * np.log(Pxx_norm + 1e-12))
    peak_freq = f[np.argmax(Pxx)]
    return {
        'freq_centroid': centroid,
        'freq_bandwidth': bandwidth,
        'spectral_entropy': spectral_entropy,
        'peak_frequency': peak_freq,
        'psd': Pxx,  # servirà per energy ratio sui bearing
        'freqs': f
    }


def time_frequency_features(signal, fs=20000):
    """Features tempo-frequenza usando STFT"""
    f, t, Zxx = stft(signal, fs=fs, nperseg=1024)
    magnitude = np.abs(Zxx)
    mean_mag = np.mean(magnitude)
    std_mag = np.std(magnitude)
    max_mag = np.max(magnitude)
    return {
        'stft_mean': mean_mag,
        'stft_std': std_mag,
        'stft_max': max_mag
    }


# --- Funzioni per features bearing-specific ---
def bearing_features(Pxx, freqs, shaft_freq_Hz=30, n_balls=9, n_poles=1, f_ratio=1.0):
    """
    Calcola energia in bande di frequenza tipiche di guasto bearing.
    shaft_freq_Hz: frequenza di rotazione albero in Hz
    n_balls: numero sfere
    """
    # Frequenze caratteristiche semplificate
    BPFO = n_balls / 2 * shaft_freq_Hz * (1 - f_ratio)
    BPFI = n_balls / 2 * shaft_freq_Hz * (1 + f_ratio)
    FTF = 0.5 * shaft_freq_Hz * (1 - f_ratio)

    # Funzione helper per energia in ±5Hz intorno alla freq caratteristica
    def band_energy(center_freq, band=5):
        idx = np.where((freqs >= center_freq - band) & (freqs <= center_freq + band))[0]
        return np.sum(Pxx[idx]) / (np.sum(Pxx) + 1e-12)

    return {
        'BPFO_energy': band_energy(BPFO),
        'BPFI_energy': band_energy(BPFI),
        'FTF_energy': band_energy(FTF)
    }


# Funzione principale che calcola tutte le features per un segnale
# --- Funzione completa di estrazione features ---
def extract_all_features(signal, fs=20000, shaft_freq_Hz=30, n_balls=9):
    features = {}

    # Tempo
    features.update(time_domain_features(signal))

    # Frequenza
    freq_feats = frequency_domain_features(signal, fs)
    features.update({k:v for k,v in freq_feats.items() if k not in ['psd','freqs']})

    # Tempo-frequenza
    features.update(time_frequency_features(signal, fs))

    # Bearing-specific
    #features.update(bearing_features(freq_feats['psd'], freq_feats['freqs'], shaft_freq_Hz, n_balls))
    return features


# ricarica i dati raw letti precedentemente dal file parquet
#df = pd.read_parquet("ims_1test_raw.parquet")
#df = pd.read_pickle("/IMS/ims_1test_raw.pkl", compression='infer', storage_options=None)
'''
messaggio = "Estrazione Features Tempo, Frequenza, Tempo-frequenza"

# --- Iterazione sul DataFrame ---
features_list = []
for idx, row in df.iterrows():
    features_x = extract_all_features(np.array(row['x']))
    features_y = extract_all_features(np.array(row['y']))
    # Prefisso per distinguere asse x e y
    features_x = {f'x_{k}': v for k, v in features_x.items()}
    features_y = {f'y_{k}': v for k, v in features_y.items()}
    # Unione
    all_features = {**features_x, **features_y}
    features_list.append(all_features)

# Aggiunta al DataFrame
features_df = pd.DataFrame(features_list)
df = pd.concat([df.reset_index(drop=True), features_df.reset_index(drop=True)], axis=1)
#df.drop(columns=['x', 'y', ], inplace=True)

df.to_parquet(
    "/Users/luisianasabbatini/Library/CloudStorage/OneDrive-UniversitàPolitecnicadelleMarche/"
    "WORK/UNIVPM/Univpm_Pubblicazioni/WIP_2025_SI_sustainability/IMSbearing_edge_cloud/"
    "pythonProject/ims_1test_features.parquet",
    engine="pyarrow",  # oppure "fastparquet"
    index=False
)


print(messaggio)'''
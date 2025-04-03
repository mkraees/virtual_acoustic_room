import numpy as np
import scipy.signal
import simulator
import os
import soundfile as sf
from scipy.signal import resample_poly
from joblib import Parallel, delayed
import multiprocessing
import time


def create_brir(src_azim, src_elev):
    room_materials = [1, 1, 1, 1, 15, 16]
    room_dim_xyz = [5, 4, 3]
    head_pos_xyz = [2, 1.4, 1.2]
    head_azim = 0
    src_elev = src_elev
    src_azim = src_azim
    src_dist = 1.4

    c = 344.5
    buffer = 0
    sr = 44100
    dur = 0.5
    use_hrtf_symmetry = True
    use_highpass = True
    use_jitter = False
    use_log_distance = False
    hrtf_locs = None
    hrtf_firs = None
    incorporate_lead_zeros = False

    brir = simulator.get_brir(
        room_materials=room_materials,
        room_dim_xyz=room_dim_xyz,
        head_pos_xyz=head_pos_xyz,
        head_azim=head_azim,
        src_azim=src_azim,
        src_elev=src_elev,
        src_dist=src_dist,
        buffer=buffer,
        sr=sr,
        dur=dur,
        use_jitter=use_jitter,
        use_hrtf_symmetry=use_hrtf_symmetry,
        incorporate_lead_zeros=incorporate_lead_zeros,
        strict=True,
    )
    return brir

def spatialise(y, brir):
    y_padded = np.pad(y, (brir.shape[0] - 1, 0))
    y_spatialized = scipy.signal.oaconvolve(
        y_padded[:, None],
        brir,
        mode="valid",
        axes=0,
    )
    return y_spatialized

def process_angle_combination(a, e, sound_path, save_folder, audio_files):
    print(f'Processing angle combination: azim={a}, elev={e}')
    brir = create_brir(a, e)
    
    for file in audio_files:
        sound = os.path.join(sound_path, file)
        y, sr = sf.read(sound)

        # use mono channel if stereo
        if len(y.shape) > 1:
            y_mono = y[:,1]
        else:
            y_mono = y

        # up-sample
        y_resampled = resample_poly(y_mono, up=48000, down=44100)
        y = y_resampled[:2*48000]

        y_spatialized = spatialise(y, brir)
        y_spatialized = np.asarray(y_spatialized, dtype=np.float32)
        filename = file.split('.')[0]
        
        np.save(os.path.join(save_folder, f"sound_{filename}_azim{round(a)}_elev{round(e)}.npy"), y_spatialized)
        # print(f"sound_{filename}_a{a}_e{e}.npy")
    
    print(f'DONE: Processing: azim={a}, elev={e}')

# Configuration
sound_path = '/mnt/lustre/work/macke/mwe234/datasets/short_audio_500'
save_folder = '/mnt/lustre/work/macke/mwe234/datasets/simulated/500_spatial_360_7/'
os.makedirs(save_folder, exist_ok=True)

audio_files = [f for f in os.listdir(sound_path) if f.endswith('.wav')]

# Spatialization parameters
azim = np.linspace(-180, 180, 72, endpoint=False)
elev = np.linspace(0, 60, 7, endpoint=True)

# Determine number of CPUs to use (use all available)
num_cpus = round(multiprocessing.cpu_count()/2) # use half only so that birir generation can use multiprocessing as well
print(f'Using {num_cpus} CPU cores for parallel processing.')

# Generate all angle combinations
angle_combinations = [(a, e) for a in azim for e in elev]

start_time = time.time()

# Parallel processing across angle combinations
Parallel(n_jobs=num_cpus)(
    delayed(process_angle_combination)(a, e, sound_path, save_folder, audio_files)
    for a, e in angle_combinations
)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Processing completed in {elapsed_time:.2f} seconds.")
import os
import sys
import pdb
import time
import numpy as np
import soundfile as sf
import scipy.signal
import scipy.interpolate


"""
The MATLAB function `acoeff_hrtf.m` maps materials to acoustic absorption coefficients.
This dictionary maps integer codes to materials for which coefficients are available.
"""
map_int_to_material = {
    # WALLS
    1: 'Brick',
    2: 'Concrete, painted',
    3: 'Window Glass',
    4: 'Marble',
    5: 'Plaster on Concrete',
    6: 'Plywood',
    7: 'Concrete block, coarse',
    8: 'Heavyweight drapery',
    9: 'Fiberglass wall treatment, 1 in',
    10: 'Fiberglass wall treatment, 7 in',
    11: 'Wood panelling on glass fiber blanket',
    # FLOORS
    12: 'Wood parquet on concrete',
    13: 'Linoleum',
    14: 'Carpet on concrete',
    15: 'Carpet on foam rubber padding',
    # CEILINGS
    16: 'Plaster, gypsum, or lime on lath',
    17: 'Acoustic tiles, 0.625", 16" below ceiling',
    18: 'Acoustic tiles, 0.5", 16" below ceiling',
    19: 'Acoustic tiles, 0.5" cemented to ceiling',
    20: 'Highly absorptive panels, 1", 16" below ceiling',
    # OTHERS
    21: 'Upholstered seats',
    22: 'Audience in upholstered seats',
    23: 'Grass',
    24: 'Soil',
    25: 'Water surface',
    26: 'Anechoic',
    27: 'Uniform (0.6) absorbtion coefficient',
    28: 'Uniform (0.2) absorbtion coefficient',
    29: 'Uniform (0.8) absorbtion coefficient',
    30: 'Uniform (0.14) absorbtion coefficient',
    31: 'Artificial - absorbs more at high freqs',
    32: 'Artificial with absorption higher in middle ranges',
    33: 'Artificial - absorbs more at low freqs',
}


def acoeff_hrtf(material, freq=[125, 250, 500, 1000, 2000, 4000]):
    """
    Python implementation of `acoeff_hrtf.m` by msaddler (2023/07)
    """
    freq = np.array(freq, dtype=float)
    freqtable = np.array([125, 250, 500, 1000, 2000, 4000], dtype=float)
    walls = [
        [0.03, 0.03, 0.03, 0.04, 0.05, 0.07],  # 1  : Brick
        [0.10, 0.05, 0.06, 0.07, 0.09, 0.08],  # 2  : Concrete, painted
        [0.35, 0.25, 0.18, 0.12, 0.07, 0.04],  # 3  : Window Glass
        [0.01, 0.01, 0.01, 0.01, 0.02, 0.02],  # 4  : Marble
        [0.12, 0.09, 0.07, 0.05, 0.05, 0.04],  # 5  : Plaster on Concrete
        [0.28, 0.22, 0.17, 0.09, 0.10, 0.11],  # 6  : Plywood
        [0.36, 0.44, 0.31, 0.29, 0.39, 0.25],  # 7  : Concrete block, coarse
        [0.14, 0.35, 0.55, 0.72, 0.70, 0.65],  # 8  : Heavyweight drapery
        [0.08, 0.32, 0.99, 0.76, 0.34, 0.12],  # 9  : Fiberglass wall treatment, 1 in
        [0.86, 0.99, 0.99, 0.99, 0.99, 0.99],  # 10 : Fiberglass wall treatment, 7 in
        [0.40, 0.90, 0.80, 0.50, 0.40, 0.30],  # 11 : Wood panelling on glass fiber blanket
    ]
    floors = [
        [0.04, 0.04, 0.07, 0.06, 0.06, 0.07],  # 12 : Wood parquet on concrete
        [0.02, 0.03, 0.03, 0.03, 0.03, 0.02],  # 13 : Linoleum
        [0.02, 0.06, 0.14, 0.37, 0.60, 0.65],  # 14 : Carpet on concrete
        [0.08, 0.24, 0.57, 0.69, 0.71, 0.73],  # 15 : Carpet on foam rubber padding
    ]
    ceilings = [
        [0.14, 0.10, 0.06, 0.05, 0.04, 0.03],  # 16 : Plaster, gypsum, or lime on lath
        [0.25, 0.28, 0.46, 0.71, 0.86, 0.93],  # 17 : Acoustic tiles, 0.625", 16" below ceiling
        [0.52, 0.37, 0.50, 0.69, 0.79, 0.78],  # 18 : Acoustic tiles, 0.5", 16" below ceiling
        [0.10, 0.22, 0.61, 0.66, 0.74, 0.72],  # 19 : Acoustic tiles, 0.5" cemented to ceiling
        [0.58, 0.88, 0.75, 0.99, 1.00, 0.96],  # 20 : Highly absorptive panels, 1", 16" below ceiling
    ]
    others = [
        [0.19, 0.37, 0.56, 0.67, 0.61, 0.59],  # 21 : Upholstered seats
        [0.39, 0.57, 0.80, 0.94, 0.92, 0.87],  # 22 : Audience in upholstered seats
        [0.11, 0.26, 0.60, 0.69, 0.92, 0.99],  # 23 : Grass
        [0.15, 0.25, 0.40, 0.55, 0.60, 0.60],  # 24 : Soil
        [0.01, 0.01, 0.01, 0.02, 0.02, 0.03],  # 25 : Water surface
        [1.00, 1.00, 1.00, 1.00, 1.00, 1.00],  # 26 : Anechoic
        # 27 : Uniform (0.6) absorbtion coefficient
        [0.60, 0.60, 0.60, 0.60, 0.60, 0.60],
        # 28 : Uniform (0.2) absorbtion coefficient
        [0.20, 0.20, 0.20, 0.20, 0.20, 0.20],
        # 29 : Uniform (0.8) absorbtion coefficient
        [0.80, 0.80, 0.80, 0.80, 0.80, 0.80],
        # 30 : Uniform (0.14) absorbtion coefficient
        [0.14, 0.14, 0.14, 0.14, 0.14, 0.14],
        # 31 : Artificial - absorbs more at high freqs
        [0.08, 0.08, 0.10, 0.10, 0.12, 0.12],
        # 32 : Artificial with absorption higher in middle ranges
        [0.05, 0.05, 0.20, 0.20, 0.10, 0.10],
        # 33 : Artificial  - absorbs more at low freqs
        [0.12, 0.12, 0.10, 0.10, 0.08, 0.08],
    ]
    atable = np.array(walls + floors + ceilings + others, dtype=float)
    if (material >= 0) and (material < 1):
        # If 0 <= material < 1, walls are set to uniform absorption
        # with a coefficient value equal to `material`
        alpha = material * np.ones_like(freq, dtype=float)
    else:
        alpha = np.zeros_like(freq, dtype=float)
        for itr, f in enumerate(freq):
            if f == 0:
                alpha[itr] = 0
            else:
                alpha[itr] = np.interp(
                    f,
                    freqtable,
                    atable[material - 1],
                    left=atable[material - 1, 0],
                    right=atable[material - 1, -1])
    return alpha, freq


def shapedfilter_hrtf(sdelay, freq, gain, sr, ctap, ctap2):
    """
    Python implementation of `shapedfilter_hrtf.m` by msaddler (2023/07)
    """
    sdelay = sdelay.reshape((-1, 1))  # Ensure sdelay is an M x 1 matrix
    assert np.all(sdelay >= 0), "The sample delay must be positive"
    ntaps = 2 * ctap - 1
    N = ctap - 1
    fc = 0.9
    # Design the non-integer delay filter
    x = np.ones((1, ntaps))
    x[0, 0] = 0
    x = np.matmul(np.ones(sdelay.shape), np.arange(-N, N +
                  1).reshape((1, -1))) - np.matmul(sdelay, x)
    h = 0.5 * fc * (1 + np.cos(np.pi * x / N)) * np.sinc(fc * x)
    freq = freq.reshape((-1))  # Ensure freq is a vector
    if ctap2 > 1:
        # Determine FFT points
        df = np.arange(0, ctap2) * (np.pi / (ctap2 - 1))
        freq = np.array([-np.spacing(1)] + list(2 * np.pi * freq) + [np.pi])
        gain = np.concatenate([gain[:, :1], gain, gain[:, -1:]], axis=1)
        # Interpolate reflection frequency-dependence to get gains at FFT points
        G = scipy.interpolate.interp1d(freq.reshape([-1]), gain)(df)
        # Combine the non-integer delay filter and the wall/sphere filter
        G[:, ctap2-1] = np.real(G[:, ctap2-1])
        # Transform into appropriate wall transfer function
        G = np.concatenate([G, np.conj(G[:, 1:ctap2-1])[:, ::-1]], axis=1)
        gt = np.real(np.fft.ifft(G.T, axis=0))
        # Zero-pad and FFT
        g = np.concatenate([
            0.5 * gt[(ctap2-1):(ctap2), :],
            gt[ctap2: (2 * ctap2 - 2), :],
            gt[0: ctap2 - 1, :],
            0.5 * gt[(ctap2-1):(ctap2), :],
            np.zeros((2 * ctap - 2, gt.shape[1])),
        ], axis=0)
        G = np.fft.fft(g, axis=0)
        # Zero-pad and FFT the delay filter
        H = np.fft.fft(
            np.concatenate([
                h.T,
                np.zeros((2 * ctap2 - 2, gt.shape[1])),
            ], axis=0),
            axis=0
        )
        # Convolve wall transfer function and delay filter
        HOUT = H * G
        # Obtain total impulse response
        hout = np.real(np.fft.ifft(HOUT, axis=0)).T
    else:
        # Scale impulse response only if wall reflections are
        # frequency-independent and sphere is not present
        hout = h * np.matmul(gain[:, 0], np.ones(h[0:1, :].shape))
    return hout


def impulse_generate_hrtf(
        h=None,
        head_cent=None,
        head_azim=None,
        s_locations=None,
        s_reflections=None,
        m_locs=None,
        m_locs_xyz=None,
        m_locs_xyz_logdist=None,
        m_files=None,
        m_delay=None,
        sr=None,
        c=None,
        ntaps=None,
        ctap=None,
        ctap2=None,
        fgains=None,
        nfreq=None,
        lead_zeros=None,
        use_m_sym=None,
        use_log_distance=None,
        use_jitter=None):
    """
    Python implementation of `impulse_generate_hrtf.m` by msaddler (2023/07)
    """
    jitter_reflects = 5

    """
    Part I: Form variables to be used in impulse response generation
    """
    # Determine overall source gains (based on number of reflections
    # through each wall) for each source location.
    gains = np.ones((s_locations.shape[0], nfreq.shape[0]), dtype=float)
    for itr_wall in range(6):
        gains = gains * \
            np.power(fgains[itr_wall:itr_wall + 1, :],
                     s_reflections[:, itr_wall:itr_wall + 1])
    # If use_m_sym is active, convert 180° to 360° sources to 0° to 180° sources
    s_locations_relh = s_locations - head_cent.reshape((1, -1))
    s_locations_pol = np.zeros_like(s_locations)
    s_locations_pol[:, 0] = np.sqrt(
        np.sum(np.square(s_locations_relh), axis=1))
    s_locations_pol[:, 1] = np.rad2deg(
        np.angle(s_locations_relh[:, 0] - 1j * s_locations_relh[:, 1])) - head_azim
    s_locations_pol[:, 2] = np.rad2deg(
        np.arcsin(s_locations_relh[:, 2] / s_locations_pol[:, 0]))
    if use_m_sym:
        flip = s_locations_pol[:, 1] < 0
        s_locations_pol[:, 1] = np.abs(s_locations_pol[:, 1])
        r = s_locations_pol[:, 0]
        s_locations = np.stack([
            r * np.cos(np.deg2rad(s_locations_pol[:, 1] + head_azim)) * np.cos(
                np.deg2rad(s_locations_pol[:, 2])),
            r * -np.sin(np.deg2rad(s_locations_pol[:, 1] + head_azim)) * np.cos(
                np.deg2rad(s_locations_pol[:, 2])),
            r * np.sin(np.deg2rad(s_locations_pol[:, 2])),
        ], axis=1)
        s_locations = s_locations + head_cent.reshape((1, -1))
    else:
        flip = np.zeros((s_locations.shape[0]), dtype=bool)

    # If use_log_distance is active, form s_locations_logdist
    if use_log_distance:
        r = np.log(s_locations_pol[:, 0]) - np.log(0.05)
        s_locations_logdist = np.stack([
            r * np.cos(np.deg2rad(s_locations_pol[:, 1] + head_azim)) * np.cos(
                np.deg2rad(s_locations_pol[:, 2])),
            r * -np.sin(np.deg2rad(s_locations_pol[:, 1] + head_azim)) * np.cos(
                np.deg2rad(s_locations_pol[:, 2])),
            r * np.sin(np.deg2rad(s_locations_pol[:, 2])),
        ], axis=1)
        s_locations_logdist = s_locations_logdist + head_cent.reshape((1, -1))
        D = m_locs_xyz_logdist[:, np.newaxis, :] - \
            s_locations_logdist[np.newaxis, :, :]
    else:
        D = m_locs_xyz[:, np.newaxis, :] - s_locations[np.newaxis, :, :]
    # For each source, determine the closest measurement spot
    D = np.sqrt(np.sum(np.square(D), axis=2))
    near_m_loc = np.argmin(D, axis=0)

    """
    Part II: Based on the center of the head, introduce a 
    1 percent jitter to add into all source-to-mic distances
    that are reflected by more than 5 walls (if use_jitter)
    """
    if use_jitter:
        jitt = np.random.randn(s_locations_pol.shape[0])
        jitt[s_reflections.sum(axis=1) < jitter_reflects] = 0
        s_locations_pol[:, 0] = s_locations_pol[:, 0] + jitt
    # Calculate the relative additional distance between each
    # (jittered) source and the corresponding measurement location
    rel_dist = s_locations_pol[:, 0] - m_locs[near_m_loc, 0]

    """
    Part III: For each measurement location, generate impulse
    response from corresponding sources to meas loc.  Then
    incorporate HRTFs.  Treat flips and no flips accordingly.
    """
    hrtf_temp = np.zeros((m_locs.shape[0], *h.shape), dtype=float)
    for l in range(hrtf_temp.shape[0]):
        IDX_l = near_m_loc == l
        if IDX_l.sum() > 0:
            IDX_noflip = np.logical_and(IDX_l, ~flip)
            IDX_flip = np.logical_and(IDX_l, flip)
            h_noflip = np.zeros_like(h)
            h_flip = np.zeros_like(h)

            # Treat non-flipped sources
            if IDX_noflip.sum() > 0:
                # Get sample delays to the measured location
                thit = ctap + ctap2 - lead_zeros + \
                    m_delay[l] + (rel_dist[IDX_noflip] * sr / c)
                ihit = np.floor(thit)
                fhit = thit - ihit
                gains_noflip = gains[IDX_noflip, :]
                # Get scale factors to account for distance traveled
                m_sc = 1 / m_locs[near_m_loc[IDX_noflip], 0]
                s_sc = 1 / s_locations_pol[IDX_noflip, 0]
                rel_sc = s_sc / m_sc
                # Eliminate locations that are too far away to enter into impulse response
                v = ihit <= ntaps + ctap + ctap2
                if v.sum() > 0:
                    # Initialize temporary impulse response vector
                    ht = np.zeros(
                        (h.shape[0] + ctap + 1 + ctap2 + 1, 1), dtype=float)
                    # Indices into ht. Each row corresonds to one source image location, with the center
                    # determined by ihit. Within a row, there are (2 * ctap - 1) + (2 * ctap2 - 1) - 1 values
                    # that account for non-integer dela, fhit, and for frequency-dependent wall reflections /
                    # sphere diffraction
                    ht_ind = ihit[v].reshape(-1, 1) * \
                        np.ones((1, 2 * ctap - 1 + 2 * ctap2 - 1 - 1))
                    ht_ind = ht_ind + \
                        np.arange(-ctap - ctap2 + 1 + 1, ctap +
                                  ctap2 - 1).reshape((1, -1))
                    ht_ind = ht_ind.astype(int)
                    # For each source location, determine the impulse response (generate filter to
                    # incorporate frequency gains, non-integer delay and scattering off rigid sphere
                    h_temp = rel_sc[v].reshape(-1, 1) * shapedfilter_hrtf(
                        fhit[v],
                        nfreq,
                        gains_noflip[v],
                        sr,
                        ctap,
                        ctap2)
                    # Add impulse response segments into the overall impulse response
                    for k in range(v.sum()):
                        ht[ht_ind[k], 0] = ht[ht_ind[k], 0] + h_temp[k, :]
                    # Incorporate HRTF impulse response and add into overall impulse response matrix
                    hrtf, sr_hrtf = sf.read(
                        m_files[l].replace('\\', '/').replace(' ', ''))
                    assert sr == sr_hrtf, "sampling rate does not match HRTF"
                    new_vals = np.stack([
                        scipy.signal.fftconvolve(
                            ht[:h.shape[0], 0], hrtf[:, 0], mode='full'),
                        scipy.signal.fftconvolve(
                            ht[:h.shape[0], 0], hrtf[:, 1], mode='full'),
                    ], axis=1)
                    hrtf_temp[l, :, :] = hrtf_temp[l, :, :] + \
                        new_vals[:hrtf_temp.shape[1]]

            # Treat flipped sources
            if IDX_flip.sum() > 0:
                # Get sample delays to the measured location
                thit = ctap + ctap2 - lead_zeros + \
                    m_delay[l] + (rel_dist[IDX_flip] * sr / c)
                ihit = np.floor(thit)
                fhit = thit - ihit
                gains_flip = gains[IDX_flip, :]
                # Get scale factors to account for distance traveled
                m_sc = 1 / m_locs[near_m_loc[IDX_flip], 0]
                s_sc = 1 / s_locations_pol[IDX_flip, 0]
                rel_sc = s_sc / m_sc
                # Eliminate locations that are too far away to enter into impulse response
                v = ihit <= ntaps + ctap + ctap2
                if v.sum() > 0:
                    # Initialize temporary impulse response vector
                    ht = np.zeros(
                        (h.shape[0] + ctap + 1 + ctap2 + 1, 1), dtype=float)
                    # Indices into ht. Each row corresonds to one source image location, with the center
                    # determined by ihit. Within a row, there are (2 * ctap - 1) + (2 * ctap2 - 1) - 1 values
                    # that account for non-integer dela, fhit, and for frequency-dependent wall reflections /
                    # sphere diffraction
                    ht_ind = ihit[v].reshape(-1, 1) * \
                        np.ones((1, 2 * ctap - 1 + 2 * ctap2 - 1 - 1))
                    ht_ind = ht_ind + \
                        np.arange(-ctap - ctap2 + 1 + 1, ctap +
                                  ctap2 - 1).reshape((1, -1))
                    ht_ind = ht_ind.astype(int)
                    # For each source location, determine the impulse response (generate filter to
                    # incorporate frequency gains, non-integer delay and scattering off rigid sphere
                    h_temp = rel_sc[v].reshape(-1, 1) * shapedfilter_hrtf(
                        fhit[v],
                        nfreq,
                        gains_flip[v],
                        sr,
                        ctap,
                        ctap2)
                    # Add impulse response segments into the overall impulse response
                    for k in range(v.sum()):
                        ht[ht_ind[k], 0] = ht[ht_ind[k], 0] + h_temp[k, :]
                    # Incorporate HRTF impulse response and add into overall impulse response matrix
                    hrtf, sr_hrtf = sf.read(
                        m_files[l].replace('\\', '/').replace(' ', ''))
                    assert sr == sr_hrtf, "sampling rate does not match HRTF"
                    new_vals = np.stack([
                        scipy.signal.fftconvolve(
                            ht[:h.shape[0], 0], hrtf[:, 1], mode='full'),
                        scipy.signal.fftconvolve(
                            ht[:h.shape[0], 0], hrtf[:, 0], mode='full'),
                    ], axis=1)
                    hrtf_temp[l, :, :] = hrtf_temp[l, :, :] + \
                        new_vals[:hrtf_temp.shape[1]]

    h = h + hrtf_temp.sum(axis=0)
    return h, s_locations


def room_impulse_hrtf(
        src_loc=[5, 5, 5],
        head_cent=[2, 2, 2],
        head_azim=0,
        walls=[10, 10, 10],
        wtypes=[3, 3, 3, 3, 3, 3],
        sr=44100,
        c=344.5,
        dur=0.5,
        use_m_sym=True,
        use_log_distance=False,
        use_jitter=True,
        use_highpass=True):
    """
    Python implementation of `room_impulse_hrtf.m` by msaddler (2023/07)
    """
    src_loc = np.array(src_loc, dtype=float)
    head_cent = np.array(head_cent, dtype=float)
    head_azim = np.array(head_azim, dtype=float)
    m_files = scipy.io.loadmat('HRTFs/file_names.mat')['gardnermartin_file']
    m_locs = scipy.io.loadmat('HRTFs/data_locs.mat')['locs_gardnermartin']
    m_locs = np.array(m_locs, dtype=float)
    m_delay = (np.sqrt(np.sum(np.square(src_loc - head_cent))) /
               c) * np.ones((m_locs.shape[0],))
    use_m_sym = use_m_sym

    # Frequency-dependent reflection coefficients for each wall
    fgains = np.zeros((6, 6), dtype=float)
    for itr_wall, material in enumerate(wtypes):
        alpha, freq = acoeff_hrtf(material=material)
        fgains[itr_wall, :] = np.sqrt(1 - alpha)
    # True when fgains is frequency-independent
    uniform_walls = len(np.unique(fgains)) == 1

    nfreq = freq / sr  # Frequencies as a fraction of sampling rate
    ntaps = int(sr * dur)  # Number of taps in output BRIR

    """
    Part I: Initialization
    """
    ctap = 11  # Center tap of lowpass to create non-integer delay impulse (as in Peterson)
    if uniform_walls:
        ctap2 = 1  # If walls are uniform, use a single-tap filter to scale gain
    else:
        ctap2 = 33  # For frequency-dependent wall reflections, use a longer filter

    # Convert measured HRTF locations into room (xyz) coordinates (and log distance locations)
    m_locs_xyz = np.ones_like(m_locs)
    r = m_locs[:, 0]
    m_locs_xyz[:, 0] = r * np.cos(np.deg2rad(m_locs[:, 1] + head_azim)
                                  ) * np.cos(np.deg2rad(m_locs[:, 2]))
    m_locs_xyz[:, 1] = r * -np.sin(np.deg2rad(m_locs[:, 1] + head_azim)
                                   ) * np.cos(np.deg2rad(m_locs[:, 2]))
    m_locs_xyz[:, 2] = r * np.sin(np.deg2rad(m_locs[:, 2]))
    m_locs_xyz = m_locs_xyz + head_cent.reshape([1, -1])
    m_locs_xyz_logdist = np.ones_like(m_locs)
    r = (np.log(m_locs[:, 0]) - np.log(0.05))
    m_locs_xyz_logdist[:, 0] = r * np.cos(np.deg2rad(
        m_locs[:, 1] + head_azim)) * np.cos(np.deg2rad(m_locs[:, 2]))
    m_locs_xyz_logdist[:, 1] = r * -np.sin(np.deg2rad(
        m_locs[:, 1] + head_azim)) * np.cos(np.deg2rad(m_locs[:, 2]))
    m_locs_xyz_logdist[:, 2] = r * np.sin(np.deg2rad(m_locs[:, 2]))
    m_locs_xyz_logdist = m_locs_xyz_logdist + head_cent.reshape((1, -1))

    # Calculate the number of lead zeros to strip
    idx_min = np.argmin(
        np.sqrt(np.sum(np.square(src_loc.reshape((1, -1)) - m_locs_xyz), axis=1)))
    src_mloc = m_locs_xyz[idx_min, :]  # Nearest measured loc or direct path
    rel_dist = np.linalg.norm(src_loc - head_cent, 2) - \
        np.linalg.norm(src_mloc - head_cent, 2)
    lead_zeros = m_delay[idx_min] + np.floor(sr * rel_dist / c)

    # Initialize output matrix (will later truncate to exactly ntaps in length)
    hrtf, sr_hrtf = sf.read(m_files[0].replace('\\', '/').replace(' ', ''))
    assert sr == sr_hrtf, "sampling rate does not match HRTF"
    h = np.zeros(
        (ntaps + ctap + ctap2 + hrtf.shape[0], 2), dtype=float)  # 2 ears

    """
    Part II: determine source image locations and corresponding impulse
    response contribution from each source.  To speed up process yet ease
    the computational burden, for every 10000 source images, break off and
    determine impulse response.
    
    The algorithm for determining source images is as follows:
    1. Calculate maximum distance which provides relevant sources
        (i.e., those that arrive within the imp_resp duration)
    2. By looping through the X dimension, generate images of
        the (0,0,0) corner of the room, restricting the
        distance below the presecribed level.
    3. Use the coordinates of each (0,0,0) image to generate 8
        source images
    4. Generate corresponding number of reflections from each wall
        for each source image.
    """
    # Maximum source distance to be in impulse response
    dmax = np.ceil((ntaps + lead_zeros) * c / sr + np.max(walls))
    # Initialize locations matrix
    s_locations = np.ones((20000, 3), dtype=float)
    # Initialize reflections matrix
    s_reflections = np.ones((20000, 6), dtype=float)
    # Vector to get locations from the (0, 0, 0) corner images
    src_pts = np.array([
        [1,  1,  1],
        [1,  1, -1],
        [1, -1,  1],
        [1, -1, -1],
        [-1,  1,  1],
        [-1,  1, -1],
        [-1, -1,  1],
        [-1, -1, -1],
    ], dtype=float) * src_loc.reshape((1, -1))
    Nx = np.ceil(dmax / (2 * walls[0]))  # Appropriate number of (0, 0, 0)
    loc_num = 0
    for nx in np.arange(Nx, -1, -1, dtype=int):
        if nx < Nx:
            ny = int(np.ceil(np.sqrt(np.square(dmax) -
                     np.square(nx * 2 * walls[0])) / (2 * walls[1])))
            nz = int(np.ceil(np.sqrt(np.square(dmax) -
                     np.square(nx * 2 * walls[0])) / (2 * walls[2])))
        else:
            ny = 0
            nz = 0
        X = nx * np.ones(((2 * ny + 1) * (2 * nz + 1), 1),
                         dtype=float)  # Form images of (0,0,0)
        Y = np.matmul(
            np.arange(-ny, ny + 1, dtype=float).reshape((-1, 1)),
            np.ones((1, 2 * nz + 1), dtype=float)).reshape((-1, 1))
        Z = np.matmul(
            np.ones((2 * ny + 1, 1), dtype=float),
            np.arange(-nz, nz + 1, dtype=float).reshape((1, -1))).reshape((-1, 1))
        if nx != 0:
            # If nx !=0, do both +nx and -nx
            X = np.concatenate([-X, X], axis=0)  # Images of (0, 0, 0)
            Y = np.concatenate([Y, Y], axis=0)
            Z = np.concatenate([Z, Z], axis=0)
        Xw = 2 * walls[0] * X
        Yw = 2 * walls[1] * Y
        Zw = 2 * walls[2] * Z

        # For each image of (0, 0, 0), get the 8 source images and number of reflections at each wall
        for k in range(8):
            s_refs = np.zeros((X.shape[0], 6), dtype=float)
            s_locs = np.concatenate(
                [Xw, Yw, Zw], axis=1) + src_pts[k, :].reshape((1, -1))
            s_refs[:, 0:1] = (src_pts[k, 0] > 0) * np.abs(X) + \
                (src_pts[k, 0] < 0) * np.abs(X - 1)
            s_refs[:, 1:2] = np.abs(X)
            s_refs[:, 2:3] = (src_pts[k, 1] > 0) * np.abs(Y) + \
                (src_pts[k, 1] < 0) * np.abs(Y - 1)
            s_refs[:, 3:4] = np.abs(Y)
            s_refs[:, 4:5] = (src_pts[k, 2] > 0) * np.abs(Z) + \
                (src_pts[k, 2] < 0) * np.abs(Z - 1)
            s_refs[:, 5:6] = np.abs(Z)

            while (loc_num + s_locs.shape[0]) > 20000:
                m = 20000 - loc_num
                s_locations[slice(loc_num, loc_num + m),
                            :] = s_locs[slice(0, m), :]
                s_reflections[slice(loc_num, loc_num + m),
                              :] = s_refs[slice(0, m), :]
                # Get impulse response contributions
                h, s_locations = impulse_generate_hrtf(
                    h=h,
                    head_cent=head_cent,
                    head_azim=head_azim,
                    s_locations=s_locations,
                    s_reflections=s_reflections,
                    m_locs=m_locs,
                    m_locs_xyz=m_locs_xyz,
                    m_locs_xyz_logdist=m_locs_xyz_logdist,
                    m_files=m_files,
                    m_delay=m_delay,
                    sr=sr,
                    c=c,
                    ntaps=ntaps,
                    ctap=ctap,
                    ctap2=ctap2,
                    fgains=fgains,
                    nfreq=nfreq,
                    lead_zeros=lead_zeros,
                    use_m_sym=use_m_sym,
                    use_log_distance=use_log_distance,
                    use_jitter=use_jitter)
                loc_num = 0  # Reset loc_num counter and continue
                s_locs = s_locs[slice(m, s_locs.shape[0]), :]
                s_refs = s_refs[slice(m, s_refs.shape[0]), :]

            s_locations[slice(loc_num, loc_num + s_locs.shape[0]), :] = s_locs
            s_reflections[slice(loc_num, loc_num + s_refs.shape[0]), :] = s_refs
            loc_num = loc_num + s_locs.shape[0]

    # When all locations have been generated, process the final ones
    s_locations = s_locations[0:loc_num, :]
    s_reflections = s_reflections[0:loc_num, :]
    h, s_locations = impulse_generate_hrtf(
        h=h,
        head_cent=head_cent,
        head_azim=head_azim,
        s_locations=s_locations,
        s_reflections=s_reflections,
        m_locs=m_locs,
        m_locs_xyz=m_locs_xyz,
        m_locs_xyz_logdist=m_locs_xyz_logdist,
        m_files=m_files,
        m_delay=m_delay,
        sr=sr,
        c=c,
        ntaps=ntaps,
        ctap=ctap,
        ctap2=ctap2,
        fgains=fgains,
        nfreq=nfreq,
        lead_zeros=lead_zeros,
        use_m_sym=use_m_sym,
        use_log_distance=use_log_distance,
        use_jitter=use_jitter)

    """
    Part III: Finalize output
    """
    if use_highpass:
        # Highpass filter if desired
        b, a = scipy.signal.butter(2, 0.005, btype='high')
        h = scipy.signal.lfilter(b, a, h, axis=0)
    # Restrict to `ntaps` in length
    hout = h[:ntaps, :]
    return hout, lead_zeros


def get_brir(
        room_materials=[26, 26, 26, 26, 26, 26],
        room_dim_xyz=[10, 10, 3],
        head_pos_xyz=[5, 5, 1.5],
        head_azim=0,
        src_azim=0,
        src_elev=0,
        src_dist=1.4,
        buffer_pos=0,
        sr=44100,
        c=344.5,
        dur=0.5,
        use_m_sym=True,
        use_log_distance=False,
        use_jitter=True,
        use_highpass=True,
        incorporate_lead_zeros=True,
        verbose=True):
    """
    Main function to generate binaural room impulse response (BRIR) from
    a room description, a listener position, and a source position.
    """
    room_materials = np.array(room_materials)
    msg = "room_materials shape: [wall_x0, wall_x, wall_y0, wall_y, floor, ceiling]"
    assert room_materials.shape == (6,), msg
    room_dim_xyz = np.array(room_dim_xyz)
    msg = "room_dim_xyz shape: [x_len (length), y_len (width), z_len (height)]"
    assert room_dim_xyz.shape == (3,), msg
    head_pos_xyz = np.array(head_pos_xyz)
    msg = "head_pos_xyz shape: [x_head, y_head, z_head]"
    assert head_pos_xyz.shape == (3,), msg
    src_pos_xyz = np.array([
        src_dist * np.cos(np.deg2rad(src_elev)) * np.cos(np.deg2rad(src_azim + head_azim)) + head_pos_xyz[0],
        src_dist * np.cos(np.deg2rad(src_elev)) * np.sin(np.deg2rad(src_azim + head_azim)) + head_pos_xyz[1],
        src_dist * np.sin(np.deg2rad(src_elev)) + head_pos_xyz[2],
    ])
    if verbose:
        print("[get_brir] head_pos: {}, src_pos: {}, room_dim: {}".format(
            head_pos_xyz.tolist(),
            src_pos_xyz.tolist(),
            room_dim_xyz.tolist()))
    t0 = time.time()

    h_out, lead_zeros = room_impulse_hrtf(
        src_loc=src_pos_xyz,
        head_cent=head_pos_xyz,
        head_azim=-head_azim, # `room_impulse_hrtf` convention is positive azimuth = clockwise
        walls=room_dim_xyz,
        wtypes=room_materials,
        sr=sr,
        c=c,
        dur=dur,
        use_m_sym=use_m_sym,
        use_log_distance=use_log_distance,
        use_jitter=use_jitter,
        use_highpass=use_highpass)
    if verbose:
        print(f'[get_brir] time elapsed: {time.time() - t0} seconds')
    if incorporate_lead_zeros:
        lead_zeros = int(np.round(lead_zeros))
        print(f'[get_brir] incorporated {lead_zeros} leading zeros')
        if lead_zeros >= 0:
            h_out = np.pad(h_out, ((lead_zeros, 0), (0, 0)))
            brir = h_out[:int(dur * sr)]
        else:
            h_out = np.pad(h_out, ((0, -lead_zeros), (0, 0)))
            brir = h_out[-lead_zeros:]
    else:
        brir = h_out
    return brir

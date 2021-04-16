import os
import sys
import logging

import numpy as np
import scipy.io as sio
import pandas as pd

from utils.intanutils import read_amplifier
from utils.filter import bandpass_filter, notch_filter
from utils import find_nearest


def get_spike_times(num, date, raw_dir, proc_dir, f_sampling, n_channels, f_low, f_high, noise_threshold, save_waveform=False):
    # Get names of all directories with the specified 'date'.
    with os.scandir(raw_dir) as it:
        dirs = [entry.name for entry in it if (entry.is_dir() and entry.name.find(date) != -1)]
    dirs.sort()
    logging.debug(dirs)

    for d in dirs:
        # Get all raw neural data files
        with os.scandir(os.path.join(raw_dir, d)) as it:
            files = [entry.name for entry in it if (entry.is_file() and entry.name.find('amp') != -1)]
        files.sort()  # The files are randomly loaded, so sort them

        assert len(files) == n_channels  # Check if number of files matches number of channels

        # Create spikeTime directory if it does not already exist
        if not os.path.isdir(os.path.join(proc_dir, d, 'spikeTime')):
            os.makedirs(os.path.join(proc_dir, d, 'spikeTime'))

        if not os.path.isfile(os.path.join(proc_dir, d, 'spikeTime') + '/' + files[num][:-4] + '_spk.mat'):
            # Get amplifier channel data and apply 60 Hz notch filter
            v = read_amplifier(os.path.join(raw_dir, d, files[num]))  # In microvolts
            v = notch_filter(v, f_sampling=f_sampling, f_notch=60, bandwidth=10)

            nrSegments = 10
            nrPerSegment = int(np.ceil(len(v) / nrSegments))
            spike_times_ms = []

            waveform_time_ms, waveform_uv = [], []

            for i in range(nrSegments):
                # print( i*nrPerSegment, (i+1)*nrPerSegment )
                timeIdxs = np.arange(i * nrPerSegment, (i + 1) * nrPerSegment) / f_sampling * 1000.  # In ms
                # print(timeIdxs)

                # Apply bandpass (IIR) filter
                v1 = bandpass_filter(v[i * nrPerSegment:(i + 1) * nrPerSegment], f_sampling, f_low, f_high)
                v2 = v1 - np.nanmean(v1)

                # We threshold at noise_threshold * std (generally we use noise_threshold=3)
                # The denominator 0.6745 comes from some old neuroscience
                # paper which shows that when you have spikey data, correcting
                # with this number is better than just using plain standard
                # deviation value
                noiseLevel = -noise_threshold * np.median(np.abs(v2)) / 0.6745
                outside = np.array(v2) < noiseLevel  # Spits a logical array
                outside = outside.astype(int)  # Convert logical array to int array for diff to work

                cross = np.concatenate(([outside[0]], np.diff(outside, n=1) > 0))

                idxs = np.nonzero(cross)[0]
                spike_times_ms.extend(timeIdxs[idxs])

                # Get waveforms (-1ms to 2ms)
                if save_waveform:
                    for spk_time in timeIdxs[idxs]:
                        start_index = find_nearest(timeIdxs, spk_time - 1)
                        stop_index = find_nearest(timeIdxs, spk_time + 2)
                        x_axis = timeIdxs[start_index:stop_index] - spk_time
                        y_axis = v2[start_index:stop_index]

                        waveform_time_ms.append(x_axis)
                        waveform_uv.append(y_axis)

            # Save spikeTime and waveforms
            spikeTime = {'spike_time_ms': spike_times_ms, 'waveform': {'time_ms': waveform_time_ms, 'amplitude_uv': waveform_uv}}
            sio.savemat(os.path.join(proc_dir, d, 'spikeTime') + '/' + files[num][:-4] + '_spk.mat', spikeTime,
                        oned_as='column')
    return


def get_psth(num, date, proc_dir, start_time, stop_time, timebin):
    # Get names of all directories with the specified 'date'.
    with os.scandir(proc_dir) as it:
        dirs = [entry.name for entry in it if (entry.is_dir() and entry.name.find(date) != -1)]
    dirs.sort()

    # Get all behavior files
    mwk_dir = proc_dir[:proc_dir.rfind('/')] + '/mworksproc'
    assert os.path.isdir(mwk_dir)
    mwk_files = [f for f in os.listdir(mwk_dir) if not f.startswith('.') and f.find(date) != -1]
    mwk_files.sort()

    logging.debug(f'{mwk_files}, {dirs}')
    assert len(mwk_files) == len(dirs)

    for mwk_file, d in zip(mwk_files, dirs):
        logging.debug(f'{mwk_file}, {d}')

        # Load trial times
        mwk_data = pd.read_csv(os.path.join(mwk_dir, mwk_file))
        mwk_data = mwk_data[mwk_data.fixation_correct == 1]
        if 'photodiode_on_us' in mwk_data.keys():
            samp_on_ms = np.asarray(mwk_data['photodiode_on_us']) / 1000.
            logging.info('Using photodiode signal for sample on time')
        else:
            samp_on_ms = np.asarray(mwk_data['samp_on_us']) / 1000.
            logging.info('Using MWorks digital signal for sample on time')

        # trial_times = sio.loadmat(os.path.join(proc_dir, d) + '/' + d + '_trialTimes.mat', squeeze_me=True)
        # if 'photodiode_on' in trial_times.keys():
        #     print('Using photodiode signal for trial times')
        #     trial_times = trial_times['photodiode_on']
        # else:
        #     print('Using MWorks digital signal for trial times')
        #     trial_times = trial_times['samp_on']

        # Create psth directory is it does not already exist
        if not os.path.isdir(os.path.join(proc_dir, d, 'psth')):
            os.mkdir(os.path.join(proc_dir, d, 'psth'))

        # Get all spikeTime files
        with os.scandir(os.path.join(proc_dir, d, 'spikeTime')) as it:
            files = [entry.name[:entry.name.find('_spk.mat')] for entry in it if (entry.is_file() and entry.name.startswith('amp') and
                entry.name.find('_spk') != -1)]
        files.sort()  # The files are randomly loaded, so sort them

        assert len(files) >= (num + 1)  # Check if there are at least as many files as current channel being processed
        if not len(files) >= (num + 1):
            exit()
        logging.debug(files[num])

        if not os.path.isfile(os.path.join(proc_dir, d, 'spikeTime') + '/' + files[num] + '_spk.mat'):
            exit()

        if not os.path.isfile(os.path.join(proc_dir, d, 'psth') + '/' + files[num] + '_psth.mat'):
            print('Starting to estimate PSTH')

            # Load spikeTime file for current channel
            spikeTime = \
            sio.loadmat(os.path.join(proc_dir, d, 'spikeTime') + '/' + files[num] + '_spk.mat', squeeze_me=True,
                        variable_names='spike_time_ms')['spike_time_ms']

            timebase = np.arange(start_time, stop_time, timebin)

            psth_matrix = np.full((len(samp_on_ms), len(timebase)), np.nan)

            for i in range(len(samp_on_ms)):
                for j in range(len(timebase)):
                    # logging.debug(f'{timebase[j]} {timebase[j] + timebin}')
                    psth_matrix[i, j] = np.where((spikeTime > (samp_on_ms[i] + timebase[j])) & (
                                spikeTime <= (samp_on_ms[i] + timebase[j] + timebin)))[0].size
            # print(psth_matrix)

            # Re-order the psth to image x reps
            max_number_of_reps = max(np.bincount(mwk_data['stimulus_presented']))  # Max reps obtained for any image
            mwk_data['stimulus_presented'] = mwk_data['stimulus_presented'].astype(int)  # To avoid indexing errors
            image_numbers = np.unique(mwk_data['stimulus_presented'])
            psth = np.full((len(image_numbers), max_number_of_reps, len(timebase)), np.nan)  # Re-ordered PSTH

            for i, image_num in enumerate(image_numbers):
                index_in_table = np.where(mwk_data.stimulus_presented == image_num)[0]
                selected_cells = psth_matrix[index_in_table, :]
                # Use i instead of image_num for indexing psth as image_num can be 0-25, or 1-26(!)
                psth[i, :selected_cells.shape[0], :] = selected_cells

            logging.info(psth.shape)
            # Save psth data
            meta = {'start_time_ms': start_time, 'stop_time_ms': stop_time, 'tb_ms': timebin}
            psth = {'psth': psth, 'meta': meta}

            sio.savemat(os.path.join(proc_dir, d, 'psth') + '/' + files[num] + '_psth.mat', psth)
    return


def combine_channels(proc_dir, num_channels=288):
    dirs = [_ for _ in os.listdir(proc_dir) if not _.startswith('.')]
    for d in dirs:
        filename = d + '_psth.mat'
        if not os.path.isfile(os.path.join(proc_dir, d, filename)):
            psth_dir = os.path.join(proc_dir, d, 'psth')
            if not os.path.isdir(psth_dir):  # Skip if no psth directory
                continue
            ch_files = os.listdir(psth_dir)
            ch_files.sort()
            logging.debug(f'{len(ch_files)} files in {d}')
            if not len(ch_files) == num_channels:  # Skip if not all channel files present
                continue
            psth = [sio.loadmat(os.path.join(psth_dir, f), squeeze_me=True, variable_names='psth')['psth'] for f in ch_files]
            psth = np.asarray(psth)  # channels x stimuli x reps x timebins
            psth = np.moveaxis(psth, 0, -1)  # stimuli x reps x timebins x channels
            logging.debug(psth.shape)

            meta = sio.loadmat(os.path.join(psth_dir, ch_files[0]), squeeze_me=True, variable_names='meta')['meta']
            psth = {'psth': psth, 'meta': meta}

            sio.savemat(os.path.join(proc_dir, d, filename), psth)
    return


def combine_sessions(dates, proc_dir, output_dir, normalize=False):
    assert isinstance(dates, (list, np.ndarray))
    dirs = [_ for _ in os.listdir(proc_dir) if not _.startswith('.')]
    dirs = [_ for _ in dirs if any(date in _ for date in dates)]  # Filter for given dates
    dirs.sort()
    if not dirs:  # Exit if empty
        logging.debug(f'No directories for dates {dates}')
        return

    if normalize:  # TODO: incomplete -- finish this
        # Locate the normalizer directory for this animal
        normalizer_dir = proc_dir[:proc_dir.find('projects/')+len('projects/')] + 'normalizers' + proc_dir[proc_dir.find('/monkeys/'):]

        subdirs = [_ for _ in os.listdir(normalizer_dir) if not _.startswith('.') and any(date in _ for date in dates)]
        subdirs.sort()
        # Pick the first normalizer run for each of the dates (the normalizers are generally run at the start and
        # end of each recording day
        n_dirs = []
        for date in dates:
            selected_dirs = list(filter(lambda x: date in x, subdirs))
            assert len(selected_dirs) > 0, f'No normalizers found for date {date}'
            n_dirs.append(selected_dirs[0])
        logging.debug(f'Normalizer directories are {n_dirs}')

        # Get the normalizer PSTH files (with their complete path)
        n_files = [os.path.join(normalizer_dir, d, d + '_psth.mat') for d in n_dirs]
        assert np.all([os.path.isfile(f) for f in n_files]), 'Normalizer PSTH file(s) missing'

    # Loop through all files, and concatenate the (normalized) PSTH
    psth_matrix, normalizer_matrix = [], []
    prev_date = None  # Used to track whether to re-load normalizer or not
    for i, d in enumerate(dirs):
        date = d.split('_')[-2]  # Get the date from the directory name
        file = [f for f in os.listdir(os.path.join(proc_dir, d)) if f.endswith('psth.mat') and not f.startswith('.')]
        assert len(file) == 1, f'No PSTH file found in {d}'
        file = file[0]
        p = sio.loadmat(os.path.join(proc_dir, d, file), squeeze_me=True, variable_names='psth')['psth']
        if len(p.shape) == 3:  # If there's only 1 repetition, then you need to manually add the repetition axis
            p = np.expand_dims(p, axis=1)
        logging.debug(f'{d} - {p.shape}')

        # Normalize
        if normalize:
            if date != prev_date:
                n_file = next(filter(lambda x: date in x, n_files))
                normalizer_p = sio.loadmat(n_file, squeeze_me=True, variable_names='psth')['psth']
                normalizer_meta = sio.loadmat(n_file, squeeze_me=True, variable_names='meta')['meta']
                assert len(normalizer_p.shape) == 4  # num_images x num_repetitions x num_timebins x num_channels
                assert normalizer_p.shape[0] == 26  # 25 normalizers images + 1 gray image

                timebase = np.arange(int(normalizer_meta['start_time_ms']), int(normalizer_meta['stop_time_ms']), int(normalizer_meta['tb_ms']))
                t_cols = np.where((timebase >= 70) & (timebase < 170))[0]
                n_p = np.nanmean(normalizer_p[:-1, :, t_cols, :], 2)  # Select first 25 images, and then mean 70-170 time bins

                n_p = n_p.reshape(-1, normalizer_p.shape[-1])  # Reshape so that first two axes collapse into one

                mean_response_normalizer = np.nanmean(n_p, 0)  # Mean across images x reps
                std_response_normalizer = np.nanstd(n_p, 0)  # Std across images x reps

            p = np.subtract(p, mean_response_normalizer[np.newaxis, np.newaxis, np.newaxis, :])
            p = np.divide(p, std_response_normalizer[np.newaxis, np.newaxis, np.newaxis, :],
                          where=std_response_normalizer!=0)

        if i == 0:
            meta = sio.loadmat(os.path.join(proc_dir, d, file), squeeze_me=True, variable_names='meta')['meta']
            psth_matrix = p
            if normalize:
                normalizer_matrix = normalizer_p
                prev_date = date
            continue
        psth_matrix = np.hstack((psth_matrix, p))
        if normalize and date != prev_date:
            normalizer_matrix = np.hstack((normalizer_matrix, normalizer_p))

    psth_matrix = _shift_nans(psth_matrix.copy())  # TODO check if copy() necessary
    logging.info(f'Combining sessions - {psth_matrix.shape}')

    psth_matrix = _remove_nan_cols(psth_matrix.copy())  # TODO check if copy() necessary
    logging.info(f'Trimming all NaN repetition columns - {psth_matrix.shape}')

    experiment_name = proc_dir.split('/monkeys/')[0].split('/')[-1]
    monkey_name = proc_dir.split('/monkeys/')[1].split('/')[0]

    # Save experiment PSTH
    filename = f'{monkey_name}.rsvp.{experiment_name}.experiment_psth.mat'
    if not os.path.isfile(os.path.join(output_dir, filename)):
        # Save psth data
        meta = {'start_time_ms': float(meta['start_time_ms']),
                'stop_time_ms': float(meta['stop_time_ms']),
                'tb_ms':  float(meta['tb_ms'])}
        psth = {'psth': psth_matrix, 'meta': meta}

        sio.savemat(os.path.join(output_dir, filename), psth)

    bin_counts = np.zeros(psth_matrix.shape[1] + 1)
    for image in range(psth_matrix.shape[0]):
        num_repetitions = np.count_nonzero(~np.isnan(psth_matrix[image, :, 0, 0]))
        bin_counts[num_repetitions] += 1
    logging.info(f'Bin counts - {[(index, count)for index, count in enumerate(bin_counts)]}')

    # Save normalizer PSTH
    if normalize:
        normalizer_matrix = _shift_nans(normalizer_matrix.copy())  # TODO check if copy() necessary
        normalizer_matrix = _remove_nan_cols(normalizer_matrix.copy())  # TODO check if copy() necessary
        logging.debug(f'Normalizer - {normalizer_matrix.shape}')
        filename = f'{monkey_name}.rsvp.{experiment_name}.normalizer_psth.mat'
        if not os.path.isfile(os.path.join(output_dir, filename)):
            # Save psth data
            meta = {'start_time_ms': float(normalizer_meta['start_time_ms']),
                    'stop_time_ms': float(normalizer_meta['stop_time_ms']),
                    'tb_ms': float(normalizer_meta['tb_ms'])}
            psth = {'psth': normalizer_matrix, 'meta': meta}

            sio.savemat(os.path.join(output_dir, filename), psth)

    return


def _shift_nans(data):
    for channel in range(data.shape[3]):
        for image in range(data.shape[0]):
            sub_matrix = data[image, :, :, channel]
            sub_matrix = sub_matrix.T

            # Solution from https://stackoverflow.com/questions/63031346/shift-nan-to-the-beginning-of-an-array-in-python
            mask = np.isnan(sub_matrix)  # Find the position of nan items in `psth_matrix`
            nan_pos = np.sort(mask)  # Put them at the end of the matrix
            not_nan_pos = ~nan_pos  # New position of non_non items

            result = np.empty(sub_matrix.shape)
            result[nan_pos] = np.nan
            result[not_nan_pos] = sub_matrix[~mask]

            result = result.T

            data[image, :, :, channel] = result
    return data


def _remove_nan_cols(data):
    while True:
        if np.isnan(data[:, -1, :, :]).all():
            data = data[:, :-1, :, :]
        else:
            break
    return data

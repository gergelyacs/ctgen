import os
import pandas as pd
import numpy as np
from tqdm import tqdm
#create pytables 
import tables
import omegaconf
from utils import normalize_to_neg_one_to_one, unnormalize_to_zero_to_one
import matplotlib.pyplot as plt

# STORED SHAPE: (time, channels), THE DATA IS SCALED TO [-1, 1] RANGE FOR DIFFUSION MODEL
# (first divide by max value to normalize the data, then scale to [-1, 1] range with normalize_to_neg_one_to_one)
# YOU NEED TO TRANSPOSE THE DATA TO (channels, time) WHEN LOADING with PyTablesDataset, since PyTorch expects (C, T) shape
# AND YOU HAVE TO STORE THE GENERATED DATA IN THE SAME SHAPE FOR EVALUATION!!

# create omegaconf object for sote dataset (freq, max_break, min_ts_length, stride, training_folder, input_folder)
# the time difference corresponding to the distance between two datapoints is 0.24244238252020353 seconds.
sote = omegaconf.OmegaConf.create({
    "freq": 4,
    "max_break_sec": 15,
    "min_ts_length_min": 30,
    "stride_sec": 204, # STRIDE = MIN_TS_LENGTH // 10 -> 8192 // 10 -> 819 / freq = 204 sec
    "fhr_max": 210,
    "fhr_min": 60,
    "uc_max": 100,
    "uc_min": 0,
    "training_folder": "training_data/",
    "input_folder": ["training_data/raw_sote_1/", "training_data/raw_sote_2/"],
    # h5 file name for unprocessed data
    "unproc_h5": "training_data/sote_unproc.h5",
    # h5 file name for processed data
    "proc_h5": "training_data/sote_proc.h5"
})

# create omegaconf object for czech dataset (freq, max_break, min_ts_length, stride, training_folder, input_folder)
czech = omegaconf.OmegaConf.create({
    "freq": 1,
    "max_break_sec": 15,
    "min_ts_length_min": 30,
    "stride_sec": 30,
    "fhr_max": 240,
    "fhr_min": 60,
    "uc_max": 100,
    "uc_min": 0,
    "training_folder": "training_data/",
    "input_folder": ["training_data/raw_czech/"],
    # h5 file name for unprocessed data
    "unproc_h5": "training_data/czech_unproc_2.h5",
    # h5 file name for processed data
    "proc_h5": "training_data/czech_proc_2.h5"
})

cf = czech

# freq: ca. 4 Hz -> 1/4 s
#FREQ = 4 if DATASET == "sote" else 1
#MAX_BREAK_SEC = 15
#MIN_TS_LENGTH_MIN = 60
#STRIDE_SEC = 30

max_break = cf.freq * cf.max_break_sec
MIN_TS_LENGTH = cf.freq * 60 * cf.min_ts_length_min  
STRIDE = cf.freq * cf.stride_sec 

# take the power of 2 that is greater than the MIN_TS_LENGTH with formula
MIN_TS_LENGTH = int(2 ** np.ceil(np.log2(MIN_TS_LENGTH)))

print (f"MIN_TS_LENGTH: {MIN_TS_LENGTH}")
print (f"MAX_BREAK: {max_break}")
print (f"STRIDE: {STRIDE}")

training_folder = cf.training_folder
input_folder = cf.input_folder

if not os.path.exists(cf.training_folder):
    os.makedirs(cf.training_folder)

def create_sliding_windows(patient_data, stride):
    series = []
    for patient in patient_data:
        for i in range(0, len(patient) - MIN_TS_LENGTH, stride):
            series.append(patient[i:i + MIN_TS_LENGTH])
    return series

def plot_patient_data(patient_data, title=""):
    plt.figure()
    fig, axs = plt.subplots(2, 1)
    # get column names
    cols = patient_data.columns

    # plot first column
    axs[0].plot(patient_data[cols[0]])
    #axs[0].set_title(cols[0])
    axs[1].plot(patient_data[cols[1]], color="orange")
    # swtich off the x-axis labels and y-axis labels for the bottom subplot
    #axs[1].set_xticklabels([])
    # fix y scal into uc_min uc_max
    #axs[1].set_ylim(cf.uc_min, cf.uc_max)
    #axs[0].set_ylim(cf.fhr_min, cf.fhr_max)
   
    #axs[1].set_title(cols[1])
    # remove x-axis labels from the top subplot
    plt.setp(axs[0].get_xticklabels(), visible=False)
  
    plt.suptitle(title)
    plt.show()

# read each csv file from the input folder where file name is the patient id
patient_ids = []
patient_data = []

# csv files
patient_ids = set()
files = []
duplicated = 0
for dir in input_folder:
    for file in os.listdir(dir):
        if not file.endswith(".csv"):
            continue

        patient_id = int(file.split(".")[0])
        if patient_id in patient_ids:
            duplicated += 1
            continue
        patient_ids.add(int(file.split(".")[0]))
        # make path to the file with os.path.join
        files.append((dir, file))

print ("Number of files: ", len(files))
print ("Number of duplicated patient ids: ", duplicated)

# create a new HDF5 file
hdf5_file = tables.open_file(cf.unproc_h5, mode="w")
# store windows in a new group
group = hdf5_file.create_group("/", "samples", "Training samples")
# create an array to store the windows with blosc compression, high compression level
array = hdf5_file.create_earray(group, "windows", tables.Float16Atom(), filters=tables.Filters(complib="blosc", complevel=7), shape=(0, MIN_TS_LENGTH, 2))
# create an array to store the labels with blosc compression, label is patient id
labels = hdf5_file.create_earray(group, "labels", tables.Int32Atom(),filters=tables.Filters(complib="blosc"), shape=(0,))

# iterate over the files: One file per patient
skipped_patients = []

for dir, file in tqdm(files):
    # process a single patient
    # extract the patient id from the file name
    patient_id = int(file.split(".")[0])
    # create empty pandas dataframe with cols "FHR" and "UC"
    patient = pd.read_csv(os.path.join(dir, file))

    # replace all NaN values with 0
    patient.fillna(0, inplace=True)

    # drop seconds column for czech data
    if cf == czech:
        patient.drop(columns=["seconds"], inplace=True)

    # rename the column with larger max to "FHR" and the other to "UC". the column name is not known
    if patient.iloc[:, 0].max() > patient.iloc[:, 1].max():
        patient.columns = ["FHR", "UC"]
    else:
        patient.columns = ["UC", "FHR"]

    # print statistics of the data
    #print(f"Data shape: {df.shape}")
    #print(f"Data head: {df.head()}")
    #print(f"Data describe: {df.describe()}")

    # find all indices where both FHR and UC are zero
    #nonzero_idxs = patient[(patient["FHR"] != 0) | (patient["UC"] != 0)].index.to_numpy()
    # find all indices where only FHR is zero
    nonzero_idxs = patient[patient["FHR"] != 0].index.to_numpy()
    # find the indices where the difference between consecutive indices is larger than max_break
    last_i = 0
    series = []
    for i in range(len(nonzero_idxs) - 1):
        if nonzero_idxs[i + 1] - nonzero_idxs[i] > max_break:
            # only keep the window if it is larger than MIN_TS_LENGTH
            if i - last_i > MIN_TS_LENGTH:
                series.append(patient[nonzero_idxs[last_i]:nonzero_idxs[i] + 1])
            last_i = i + 1
    # add the last window
    if nonzero_idxs[-1] - nonzero_idxs[last_i] > MIN_TS_LENGTH:
        series.append(patient[nonzero_idxs[last_i]:nonzero_idxs[-1] + 1])

    # SOTE:
    #series = create_sliding_windows(series, stride=MIN_TS_LENGTH // 10)
    # czech: 3 seconds stride
    series = create_sliding_windows(series, stride=STRIDE)

    # check series for nan
    for s in series:
        # check is nan pandas
        if s.isnull().values.any():
            print(f"NaN values in {patient_id}")
            print (s)
            continue

    if len(series) == 0:
        skipped_patients.append(patient_id)
        continue

    try:
        array.append(series)
    except ValueError:
        print(f"Value error in {patient_id}")
        print(series)
        continue
    # add the labels to the array
    labels.append([patient_id] * len(series))

# close the HDF5 file
hdf5_file.close()

print (f"Skipped patients: {skipped_patients}")
print(f"Total number of patients: {len(files)}")
print(f"Number of skipped patients: {len(skipped_patients)}")

# open the HDF5 file
hdf5_file = tables.open_file(cf.unproc_h5, mode="r")
# read the windows and labels
array = hdf5_file.root.samples.windows
labels = hdf5_file.root.samples.labels

# number of patients
num_patients = len(files)

# number of windows
num_windows = len(array)

# print the number of patients and windows
print(f"Number of patients: {num_patients:,}")
print(f"Number of windows: {num_windows:,}")

# get max min values of FHR and UC for numpy arrays
#max_values = np.max(array, axis=(1, 0))
#min_values = np.min(array, axis=(1, 0))
#print(f"Max values: {max_values}")
#print(f"Min values: {min_values}")

# compute the sum of both FHR and UC values for each window in for loop

## check the data

sum_values = np.zeros((num_windows, 2))
for i in tqdm(range(num_windows)):
    sum_values[i] = np.sum(array[i].astype(np.float32), axis=0)
print(f"Sum values: {sum_values}")

# does sum values has any nan values
if np.isnan(sum_values).any():
    print("Nan values in sum values")
    print(sum_values)

# does sum values has any inf values
if np.isinf(sum_values).any():
    print("Inf values in sum values")
    print(sum_values)

# does sum values has any negative values
if np.any(sum_values < 0):
    print("Negative values in sum values")
    print(sum_values)

# does sum values has any 0 for FHR
if np.any(sum_values[:, 0] == 0):
    print("Zero values in FHR")
    print(sum_values)

uc_max = cf.uc_max
fhr_max = cf.fhr_max
fhr_min = cf.fhr_min
uc_min = cf.uc_min

# open the HDF5 file
hdf5_file = tables.open_file(cf.unproc_h5, mode="r")
# read the windows and labels
array = hdf5_file.root.samples.windows
labels = hdf5_file.root.samples.labels

print(f"Max values; FHR: {fhr_max}, UC: {uc_max}")
print(f"Min values: FHR: {fhr_min}, UC: {uc_min}")

# print max min values

# create new HDF5 file to store the processed data.
# each window is clipped

hdf5_file_proc = tables.open_file(cf.proc_h5, mode="w")
# store windows in a new group
group = hdf5_file_proc.create_group("/", "samples", "Training samples")
# create an array to store the windows with blosc compression, high compression level
array_proc = hdf5_file_proc.create_earray(group, "windows", tables.Float16Atom(), filters=tables.Filters(complib="blosc", complevel=7), shape=(0, MIN_TS_LENGTH, 2))
# create an array to store the labels with blosc compression, label is patient id
labels_proc = hdf5_file_proc.create_earray(group, "labels", tables.Int32Atom(),filters=tables.Filters(complib="blosc"), shape=(0,))
print ("HDF5 file created.")

# copy the labels to the new HDF5 file
labels_proc.append(labels[:])
print ("Labels copied.")

# clip array into the array_proc
for i in tqdm(range(len(array))):
    # clip patient data to the 95th percentile of the max values
    patient_data_proc = np.clip(array[i], [fhr_min, uc_min], [fhr_max, uc_max])
    # Normalize the data by scaling it to the range [0, 1]
    # divide by the max value to normalize the data
    patient_data_proc = (patient_data_proc - [fhr_min, uc_min]) / [fhr_max - fhr_min, uc_max - uc_min]
    patient_data_proc = normalize_to_neg_one_to_one(patient_data_proc.astype(np.float16))
    array_proc.append([patient_data_proc])

# close the HDF5 file
hdf5_file_proc.close()
hdf5_file.close()

# open the HDF5 file
hdf5_file_proc = tables.open_file(cf.proc_h5, mode="r")
# read the windows and labels
array_proc = hdf5_file_proc.root.samples.windows
labels_proc = hdf5_file_proc.root.samples.labels

# select 3 random patients and plot the data
random_patients = np.random.choice(len(array_proc), 3)

for i in random_patients:
    # scale back the data to the original range
    tmp = unnormalize_to_zero_to_one(array_proc[i])
    tmp = tmp * [cf.fhr_max - cf.fhr_min, cf.uc_max - cf.uc_min] + [cf.fhr_min, cf.uc_min]
    plot_patient_data(pd.DataFrame(tmp, columns=["FHR", "UC"]), title=f"Patient {labels_proc[i]} (normalized)")

hdf5_file_proc.close()




import matplotlib.pyplot as plt
import torch
import numpy as np
from collections import defaultdict
import scipy.fftpack as spfft
import torch_dct
import einops
import pandas as pd
from pathlib import Path

calc_conv_out_size = lambda L_in, kernel_size, stride, dilation, padding: int((L_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1)

def hash_tensor(tensor):
    return hash(tensor.cpu().numpy().tobytes())

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob
    
def load_classes(file, classes=None):
    sample_labels = {}
    class_nums = []

    # load pandas classes
    df = pd.read_csv(file, index_col=0)
    # keep only classes attributes
    if exists(classes):
        df = df[classes]
    # iterate over the dataframe
    for i, row in df.iterrows():
        assert int(i) not in sample_labels, "Duplicate user id!"
        sample_labels[int(i)] = tuple(row.values)
    
    class_nums = [np.unique(df[c].values).shape[0] for c in classes]     
    
    return sample_labels, class_nums


# get the index of the class in the class vector
# multiply the class by the cummulative product of the class nums and divide by the class nums
# this will give the index of the class in the class vector
# example: 
# classes [1, 2, 2], class_nums [2, 3, 3] => index 1 + 2 * 2 + 2 * (3 * 2) = 17
# classes [0, 1, 0], class_nums [2, 3, 3] => index 0 + 1 * 2 + 0 * (3 * 2) = 2
def get_class_index(classes, class_nums):
    class_nums = einops.repeat(class_nums, 'c -> b c', b=classes.shape[0]).to(classes.device)
    assert class_nums.size() == classes.size(), "Class nums do not match"
    assert (classes >= 0).all(), "Classes are negative" 
    assert (classes < class_nums).all(), "Classes are not in the range of class nums"

    cm = torch.cumprod(class_nums, dim=1)
    return torch.sum(classes * cm // class_nums, dim=1)

# do the inverse: get the class vector from the index
# divide the index by the last class_num and get the remainder.
# then divide the result by the one before the last class_num and get the remainder, etc.
def get_class_vector(idxs, class_nums):
    class_nums = einops.repeat(class_nums, 'c -> b c', b=idxs.shape[0]).to(idxs.device)
    cm = torch.cumprod(class_nums, dim=1)
    res = torch.zeros((idxs.size(0), class_nums.size(1)), device=idxs.device)
    remainder = idxs.clone()
    for i in range(class_nums.size(1) - 1, 0, -1):
        res[:, i] = remainder // cm[:, i - 1]
        remainder = idxs % cm[:, i - 1]
    res[:, 0] = remainder
    return res
    
def create_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
   
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class dict2class:
    def __init__(self, **entries):
        self.__dict__.update(entries)

class Aggregator:
    def __init__(self):
        """ Aggregate  """
        self.dict = defaultdict(list)

    def update(self, dict):
        for k, v in dict.items():
            self.dict[k].append(v)

    # __getitem__ is used to access the dict
    def __getitem__(self, key):
        return self.dict[key]

    def get_avg(self, prefix=""):   
        avg_dict = {}
        for k, v in self.dict.items():
            avg_dict[prefix + k] = np.mean(v)
        return avg_dict

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

def unscale(signals, factors):
    signals = unnormalize_to_zero_to_one(signals)
    signals[..., 0, :] = signals[..., 0, :] * (factors.fhr_max - factors.fhr_min) + factors.fhr_min
    signals[..., 1, :] = signals[..., 1, :] * (factors.uc_max - factors.uc_min) + factors.uc_min

    return signals

def downsample_time_series(ts_data, sampling_rate, shift=0):
    return ts_data[..., shift::sampling_rate]

def idct1(x):
    """Return inverse 1D discrete cosine transform.
    """
    return spfft.idct(x, norm='ortho')
    #return spfft.idct(x)
        
def dct1(x):
    """Return 1D discrete cosine transform.
    """
    return spfft.dct(x, norm='ortho')
    #return spfft.dct(x)

# ts_data.shape:: b x c x ts_len
def upsample_time_series(ts_data, sampling_rate):
    # if ts_data is torch tensor
    if isinstance(ts_data, torch.Tensor):
        ts_data = ts_data.cpu().detach().numpy()
    b, c, ts_len = ts_data.shape
    new_ts_data = []
    new_ts_len = ts_len * sampling_rate
    x_new = np.arange(new_ts_len)
    x = x_new[::sampling_rate]
    
    for ts in ts_data:
        new_ts = np.zeros((c, new_ts_len))
        for i in range(c):
            # linear interpolation of missing values
            new_ts[i, :] = np.interp(x_new, x, ts[i, :])
        new_ts_data.append(new_ts)
    return np.array(new_ts_data).astype(np.float32)

def merge_time_series(ts_data, sampling_rate):
    # if ts_data is torch tensor
    if isinstance(ts_data, torch.Tensor):
        ts_data = ts_data.cpu().detach().numpy()
    b, c, ts_len = ts_data.shape
    new_ts_data = []
    new_ts_len = ts_len * sampling_rate
    x_new = np.arange(new_ts_len)
    x = x_new[::sampling_rate//2]
   
    for i in range(b // 2):
        # interleave ts_data[i] and ts_data[b//2+1]
        tmp = np.empty((c, 2 * ts_len))
        tmp[:, ::2] = ts_data[i]
        tmp[:, 1::2] = ts_data[b//2+i]

        new_ts = np.zeros((c, new_ts_len))
        for j in range(c):
            # linear interpolation of missing values
            new_ts[j, :] = np.interp(x_new, x, tmp[j, :])
        new_ts_data.append(new_ts)
    return np.array(new_ts_data).astype(np.float32)

def down_sample_ts(x, sampling_rate):
    return torch.from_numpy(upsample_time_series(downsample_time_series(x, sampling_rate), sampling_rate)).float()

def dct_decompose(x, cutoff):
    x_fft = torch_dct.dct(x, norm='ortho')
    
    # low fequency components
    component = int(cutoff * x_fft.shape[-1])
    x_fft_low = x_fft.clone()
    x_fft_high = x_fft.clone()
    
    x_fft_low[:, component:] = 0 
    x_fft_high[:, :component] = 0
    # zero out the high frequency components
    x_low = torch_dct.idct(x_fft_low, norm='ortho')
    x_high = torch_dct.idct(x_fft_high, norm='ortho')

    return x_low, x_high

def merge_freqs(x_hat):
    return x_hat[:, :x_hat.shape[1] // 2, :] + x_hat[:, x_hat.shape[1] // 2:, :]

# expects a tensor of shape (batch_size, channel_num, ts_len)
def save_signals(signals, path, classes=None, cols=5):
    if isinstance(signals, torch.Tensor):
        signals = signals.to('cpu').detach().numpy()
    dim = signals.shape[1]
    rows = signals.shape[0] // cols
    fig, axs = plt.subplots(rows, cols, figsize=(20,5))
    for i in range(rows):
        for j in range(cols):
            for k in range(dim):
                axs[i, j].plot(signals[i*cols+j][k][:])
                # set title
                if classes is not None:
                    axs[i, j].set_title(f"Class: {classes[i*cols+j].cpu().detach().numpy()}")
                    # set title size to small
                    axs[i, j].title.set_size(8)
                # switch off x labes for all but the last row
                if i != rows - 1:
                    axs[i, j].set_xticklabels([])
            #axs[i, j].set_title(f'{labels[i*5+j]}')
    plt.savefig(path, format="jpeg")
    plt.close()

def plot_syn_data(patient_data_orig, patient_data_syn, title="", labels=["Original", "Synthetic"]):
    plt.figure()
    
    fig, axs = plt.subplots(2, 1)
    error = abs(patient_data_orig - patient_data_syn)
    axs[0].plot(patient_data_orig[0, :], color="blue", label=labels[0])
    # synthetic data is in red and dashed
    axs[0].plot(patient_data_syn[0, :], color="red", linestyle="--", label=labels[1])
    axs[0].set_title("FHR (error: {:.2f})".format(error[:, 0].mean()))
    # put legen outside the plot
    axs[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    axs[1].plot(patient_data_orig[1, :], color="orange", label=labels[0])
    axs[1].plot(patient_data_syn[1, :], color="green", linestyle="--", label=labels[1])
    axs[1].set_title("UC (error: {:.2f})".format(error[1, :].mean()))
     # put legen outside the plot
    axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # remove x-axis labels from the top subplot
    plt.setp(axs[0].get_xticklabels(), visible=False)
    plt.suptitle(title)
    # save the plot
    plt.savefig(f"{title}.png")
    plt.close()


def sdct_torch(signals, frame_length, frame_step, window=torch.hamming_window):
    """Compute Short-Time Discrete Cosine Transform of `signals`.

    No padding is applied to the signals.

    Parameters
    ----------
    signal : Time-domain input signal(s), a `[..., n_samples]` tensor.

    frame_length : Window length and DCT frame length in samples.

    frame_step : Number of samples between adjacent DCT columns.

    window : Window to use for DCT.  Either a window tensor (see documentation for `torch.stft`),
        or a window tensor constructor, `window(frame_length) -> Tensor`.
        Default: hamming window.

    Returns
    -------
    dct : Real-valued F-T domain DCT matrix/matrixes, a `[..., frame_length, n_frames]` tensor.
    """
    framed = signals.unfold(-1, frame_length, frame_step)
    if callable(window):
        window = window(frame_length).to(framed)
    if window is not None:
        framed = framed * window
    return torch_dct.dct(framed, norm="ortho").transpose(-1, -2)


def isdct_torch(dcts, *, frame_step, frame_length=None, window=torch.hamming_window):
    """Compute Inverse Short-Time Discrete Cosine Transform of `dct`.

    Parameters other than `dcts` are keyword-only.

    Parameters
    ----------
    dcts : DCT matrix/matrices from `sdct_torch`

    frame_step : Number of samples between adjacent DCT columns (should be the
        same value that was passed to `sdct_torch`).

    frame_length : Ignored.  Window length and DCT frame length in samples.
        Can be None (default) or same value as passed to `sdct_torch`.

    window : Window to use for DCT.  Either a window tensor (see documentation for `torch.stft`),
        or a window tensor constructor, `window(frame_length) -> Tensor`.
        Default: hamming window.

    Returns
    -------
    signals : Time-domain signal(s) reconstructed from `dcts`, a `[..., n_samples]` tensor.
        Note that `n_samples` may be different from the original signals' lengths as passed to `sdct_torch`,
        because no padding is applied.
    """
    *_, frame_length2, n_frames = dcts.shape
    assert frame_length in {None, frame_length2}
    signals = torch_overlap_add(
        torch_dct.idct(dcts.transpose(-1, -2), norm="ortho").transpose(-1, -2),
        frame_step=frame_step,
    )
    if callable(window):
        window = window(frame_length2).to(signals)
    if window is not None:
        window_frames = window[:, None].expand(-1, n_frames)
        window_signal = torch_overlap_add(window_frames, frame_step=frame_step)
        signals = signals / window_signal
    return signals


def torch_overlap_add(framed, *, frame_step, frame_length=None):
    """Overlap-add ("deframe") a framed signal.

    Parameters other than `framed` are keyword-only.

    Parameters
    ----------
    framed : Tensor of shape `(..., frame_length, n_frames)`.

    frame_step : Overlap to use when adding frames.

    frame_length : Ignored.  Window length and DCT frame length in samples.
        Can be None (default) or same value as passed to `sdct_torch`.

    Returns
    -------
    deframed : Overlap-add ("deframed") signal.
        Tensor of shape `(..., (n_frames - 1) * frame_step + frame_length)`.
    """
    *rest, frame_length2, n_frames = framed.shape
    assert frame_length in {None, frame_length2}
    return torch.nn.functional.fold(
        framed.reshape(-1, frame_length2, n_frames),
        output_size=(((n_frames - 1) * frame_step + frame_length2), 1),
        kernel_size=(frame_length2, 1),
        stride=(frame_step, 1),
    ).reshape(*rest, -1)




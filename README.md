# Georgia Tech Robotic Musicianship Prosody Audio Analysis

## Background
The Robotic Musicianship group aims to facilitate meaningful musical interactions between humans and machines, leading to novel musical experiences and outcomes. In our research, we combine computational modeling approaches for perception, interaction, and improvisation, with novel approaches for generating acoustic responses in physical and visual manners.

The motivation for this work is based on the hypothesis that real-time collaboration between human and robotic players can capitalize on the combination of their unique strengths to produce new and compelling music. Our goal is to combine human qualities, such as musical expression and emotions, with robotic traits, such as powerful processing, the ability to perform sophisticated mathematical transformations, robust long-term memory, and the capacity to play accurately without practice.

## Purpose
The purpose of this repository is to provide documentation for the work performed by the Prosody Machine Learning Team. Human emotion is conveyed via many mediums, one of which is prosody. In linguistics, prosody is concerned with those elements of speech that are not individual phonetic segments but are properties of syllables and larger units of speech, including linguistic functions such as intonation, tone, stress, and rhythm. The main goals for this lab are 1. to classify emotion based on prosody and 2. to generate prosody for specific emotions.

## Methodology
During the summer of 2020, a 4.4 hour long dataset containing hundreds of audio clips labeled with a specific emotion was collected. This dataset, and more to come, will serve as the backbone for the models and algorithms generated this semester of Fall 2020. 

## Results from PyAudioAnalysis Feature Extraction and Model Training
During the first week with this dataset, our team decided to explore the accuracy of various classification models implemented by the popular python library PyAudioAnalysis. We trained and tested the following types of models: KNN, SVM, Random Forests, Extra Trees, and Gradient Boosting. Additionally we trained with two types of classification: All 20 emotions separated individually and the 20 emotions divided into 4 categories determined by the intersection of control and valance. The categories can be visualized as follows: 
| High Control, Negative Valance | High Control, Positive Valance | Low Control, Negative Valance | Low Control, Positive Valance |
|---|---|---|---|
| Anger | Amusement | Disappointment | Admiration |
| Contempt | Interest | Fear | Compassion |
| Disgust | Joy | Guilt | Contentment |
| Hate | Pleasure | Sadness | Love |
| Regret | Pride | Shame | Relief |

All models were trained with the following parameters consistent:
- Mid-term Window Step = 1.0 seconds
- Mid-term Window Size = 1.0 seconds
- Short-term Window Step = 0.05 seconds
- Short-term Window Size = 0.05 seconds
- computeBeat = False

The models achieved the following accuracies and f1 scores:
| Model | Classification | Best Accuracy | Best F1 | Best Hyperparameter |
|---|---|---|---|---|
| KNN | Big4 | 56.1 | 56.2 | C=11 |
| SVM | Big4  | 66.5 | 65.3 | C=1.0 |
| Extra Trees |  Big4 | 64.6 | 64.3 | C=100 |
| Gradient Boosting | Big4  | 67.0 | 66.7 | C=500 |
| Random Forest | Big4  | 63.5 | 63.2 | C=200 |
| KNN | Individual | 33.8 | 32.1 | C=15 |
| SVM | Individual | 49.1 | 48.1 | C=5.0 |
| Extra Trees | Individual | 44.3 | 42.8 | C=500 |
| Gradient Boosting | Individual | 47.2 | 46.6 | C=200 |
| Random Forest | Individual | 43.8 | 42.3 | C=200 |
|SVM|Love Vs Disgust|98.9|98.9|C=.01|

### Big4 Classification
The confusion matrix for Gradient Boosting on the Big4 Classification has been shown below. Notice the difficulty the model has in distinguishing between Low Control Positive Valance and High Control Positive Valance.

||HCN|HCP|LCN|LCP|
|---|---|---|---|---|
|HCN|20.82|2.38|2.46|1.39|
|HCP|2.21|15.16|2.38|5.66|
|LCN|2.87|2.13|15.66|2.30|
|LCP|1.39|5.16|2.62|15.41|

### Individual Classification
The confusion matrix for SVM on the Individual Classification has been shown below. Although this can be hard to read, it is useful to visualize which emotions are being confused for each other. A few interesting observations include: 

- Disgust is rarely confused with other emotions. 
- Fear and Guilt are the two most common pair of emotions to confuse for one another. 
- Pleasure is the hardest emotion to correctly classify.

||Adm|Amu|Ang|Com|Con|Con|Disa|Disg|Fear|Gui|Hate|Int|Joy|Love|Ple|Pri|Reg|Rel|Sad|Sha|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|Adm|2.29|0.41|0.08|0.00|0.08|0.00|0.24|0.00|0.00|0.00|0.16|0.24|0.24|0.08|0.16|0.00|0.00|0.08|0.00|0.00|
|Amu|0.16|3.18|0.00|0.00|0.00|0.00|0.00|0.16|0.08|0.00|0.08|0.33|0.33|0.00|0.00|0.08|0.00|0.08|0.00|0.00|
|Ang|0.00|0.08|2.94|0.00|0.41|0.08|0.00|0.16|0.16|0.08|0.41|0.00|0.08|0.08|0.08|0.08|0.00|0.08|0.00|0.16|
|Com|0.08|0.00|0.00|2.20|0.08|0.41|0.49|0.00|0.16|0.16|0.00|0.16|0.00|0.16|0.00|0.41|0.00|0.00|0.00|0.16|
|Con|0.33|0.08|0.65|0.00|1.88|0.00|0.16|0.16|0.49|0.33|0.33|0.08|0.08|0.00|0.16|0.00|0.41|0.08|0.00|0.08|
|Con|0.49|0.16|0.08|0.98|0.08|1.71|0.08|0.00|0.08|0.16|0.00|0.08|0.00|0.08|0.33|0.08|0.16|0.00|0.24|0.08|
|Disa|0.08|0.00|0.16|0.16|0.00|0.00|2.45|0.16|0.08|0.33|0.00|0.08|0.08|0.00|0.08|0.08|0.00|0.16|0.00|0.16|
|Disg|0.08|0.33|0.24|0.00|0.08|0.00|0.16|5.31|0.16|0.00|0.08|0.00|0.00|0.00|0.00|0.00|0.00|0.08|0.00|0.00|
|Fear|0.08|0.00|0.24|0.08|0.33|0.00|0.08|0.08|2.53|0.49|0.16|0.00|0.00|0.08|0.24|0.08|0.08|0.08|0.08|0.16|
|Gui|0.08|0.08|0.08|0.08|0.08|0.24|0.08|0.00|1.14|2.45|0.00|0.00|0.00|0.00|0.24|0.24|0.24|0.00|0.08|0.16|
|Hate|0.00|0.16|0.57|0.00|0.33|0.00|0.00|0.24|0.16|0.00|3.35|0.08|0.33|0.24|0.00|0.00|0.00|0.08|0.16|0.00|
|Int|0.24|0.82|0.00|0.24|0.16|0.65|0.16|0.00|0.00|0.00|0.00|2.12|0.00|0.08|0.24|0.16|0.08|0.08|0.16|0.08|
|Joy|0.49|0.57|0.08|0.16|0.16|0.00|0.08|0.00|0.08|0.00|0.33|0.08|2.53|0.33|0.41|0.08|0.08|0.24|0.00|0.00|
|Love|0.16|0.16|0.08|0.82|0.08|0.41|0.00|0.00|0.16|0.24|0.08|0.24|0.16|1.96|0.33|0.33|0.00|0.00|0.08|0.00|
|Ple|0.41|0.00|0.00|0.73|0.16|0.49|0.24|0.08|0.00|0.41|0.00|0.41|0.00|0.82|1.06|0.16|0.24|0.16|0.08|0.24|
|Pri|0.24|0.00|0.16|0.16|0.08|0.16|0.08|0.00|0.08|0.08|0.00|0.00|0.08|0.33|0.08|2.37|0.00|0.16|0.00|0.00|
|Reg|0.08|0.00|0.24|0.41|0.08|0.16|0.16|0.08|0.08|0.49|0.16|0.08|0.00|0.00|0.41|0.16|1.47|0.16|0.08|0.16|
|Rel|0.16|0.00|0.08|0.16|0.16|0.08|0.16|0.33|0.00|0.08|0.24|0.16|0.16|0.08|0.08|0.00|0.00|4.00|0.16|0.00|
|Sad|0.00|0.00|0.33|0.08|0.00|0.16|0.24|0.16|0.00|0.00|0.24|0.24|0.00|0.08|0.00|0.00|0.00|0.24|1.71|0.16|
|Sha|0.08|0.00|0.08|0.16|0.16|0.33|0.24|0.08|0.65|0.33|0.16|0.00|0.00|0.16|0.08|0.33|0.33|0.08|0.08|1.55|

### Disgust vs Love Classification
When tasked with categorizing between two emotions, our model performs extremely well. Achieving accuracies as high as 98.9% with a f1 of 98.9 in the distinction between Love and Disgust using a SVM. This reinforces the intuition that by reducing the number of emotional categories we can achieve exceptionally high accuracies for identification. The confusion matrix is shown below:
||Dis|Love|
|---|---|---|
|Dis|54.03|1.14|
|Love|0.00|44.83|


## Understanding Feature Extraction 
//TODO: Add the following sections to each of the features listed below: Description, Why is this feature important?, Algorithm for extraction
### Zero Crossing Rate
#### Description
Zero Crossing Rate is the rate of sign-changes across a signal (the rate at which the signal changes from positive negative or from negative to positive). 
#### Why is this important?
This feature is key in classifying percussive sounds. //TODO expand this section
#### Algorithm
![](https://wikimedia.org/api/rest_v1/media/math/render/svg/cda9c9054c26350f517732f08eb791d232b7c1ba)

where 's' is a signal of length 'T' and '1_r<0' is an indicator function

In Python:

    def zero_crossing_rate(frame):
        """Computes zero crossing rate of frame"""
        count = len(frame)
        count_zero = np.sum(np.abs(np.diff(np.sign(frame)))) / 2
        return np.float64(count_zero) / np.float64(count - 1.0)

#### Example Extraction
![](example_extraction_graphs/zcr.png)
### Energy
#### Description
Energy is defined as the area under the squared magnitude of the considered signal
#### Why is this important?
//TODO
#### Algorithm
![](https://wikimedia.org/api/rest_v1/media/math/render/svg/41b72bb00407a25b74769848f2920bb99a0fb1be)

Unit of E_s will be (unit of signal)^2.

In Python:

    def energy(frame):
        """Computes signal energy of frame"""
        return np.sum(frame ** 2) / np.float64(len(frame))

#### Example Extraction
![](example_extraction_graphs/energy.png)
### Entropy of Energy
#### Description
The entropy of energy is defined as the average level of "information" or "uncertainty" inherent within a signal's energy
#### Why is this important?
//TODO
#### Algorithm
![](https://wikimedia.org/api/rest_v1/media/math/render/svg/bfe3616dee43f6287d4a4e2a557de8d48ad24926)

In Python:

    def energy_entropy(frame, n_short_blocks=10):
        """Computes entropy of energy"""
        # total frame energy
        frame_energy = np.sum(frame ** 2)
        frame_length = len(frame)
        sub_win_len = int(np.floor(frame_length / n_short_blocks))
        if frame_length != sub_win_len * n_short_blocks:
            frame = frame[0:sub_win_len * n_short_blocks]

        # sub_wins is of size [n_short_blocks x L]
        sub_wins = frame.reshape(sub_win_len, n_short_blocks, order='F').copy()

        # Compute normalized sub-frame energies:
        s = np.sum(sub_wins ** 2, axis=0) / (frame_energy + eps)

        # Compute entropy of the normalized sub-frame energies:
        entropy = -np.sum(s * np.log2(s + eps))
        return entropy

#### Example Extraction
![](example_extraction_graphs/energy_entropy.png)
### Spectral Centroid
#### Description
The spectral centroid characterizes a signal's spectrum, the amount of vibration at each individual frequency, defined by the spectrum's center of mass. 

#### Why is this important?
Perceptually, a spectral centroid has a connection with a sound's brightness. It follows, that this parameter serves as an indicator of musical timbre. //TODO expand this section

#### Algorithm
![](https://wikimedia.org/api/rest_v1/media/math/render/svg/0a62c839a6ceafd854138264d81b2986d8cdaff1)

where x(n) represents the weighted frequency magnitude of frame number n and f(n) represents the center frequency of that frame.

In Python:

    def spectral_centroid_spread(fft_magnitude, sampling_rate):
        """Computes spectral centroid of frame (given abs(FFT))"""
        ind = (np.arange(1, len(fft_magnitude) + 1)) * \
            (sampling_rate / (2.0 * len(fft_magnitude)))

        Xt = fft_magnitude.copy()
        Xt = Xt / Xt.max()
        NUM = np.sum(ind * Xt)
        DEN = np.sum(Xt) + eps

        # Centroid:
        centroid = (NUM / DEN)

        # Spread:
        spread = np.sqrt(np.sum(((ind - centroid) ** 2) * Xt) / DEN)

        # Normalize:
        centroid = centroid / (sampling_rate / 2.0)
        spread = spread / (sampling_rate / 2.0)

        return centroid, spread

#### Example Extraction
![](example_extraction_graphs/spectral_centriod.png)

### Spectral Entropy
#### Description
Spectral Entropy is defined to be the entropy of the power spectral density of a signal. The power spectral density of a signal describes the distribution of power into frequency components composing that signal.
#### Why is this important?
//TODO
#### Algorithm
Power Spectral Density:

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/65720eba3cb58f844db7c7ec2eb67b821ff13713)

where 

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/0b8957c65771a48df900e1a9503ccf29750e04bc)

In Python:

    def spectral_entropy(signal, n_short_blocks=10):
        """Computes the spectral entropy"""
        # number of frame samples
        num_frames = len(signal)

        # total spectral energy
        total_energy = np.sum(signal ** 2)

        # length of sub-frame
        sub_win_len = int(np.floor(num_frames / n_short_blocks))
        if num_frames != sub_win_len * n_short_blocks:
            signal = signal[0:sub_win_len * n_short_blocks]

        # define sub-frames (using matrix reshape)
        sub_wins = signal.reshape(sub_win_len, n_short_blocks, order='F').copy()

        # compute spectral sub-energies
        s = np.sum(sub_wins ** 2, axis=0) / (total_energy + eps)

        # compute spectral entropy
        entropy = -np.sum(s * np.log2(s + eps))

        return entropy


#### Example Extraction
![](example_extraction_graphs/spectral_entropy.png)

### Spectral Flux
#### Description
Spectral flux is a measure of the rate of change of the power spectrum of a signal. More precisely, spectral flux is usually calculated as the Euclidean distance between sequential frames.

#### Why is this important?
//TODO
#### Algorithm
//TODO Write formal definition

In Python:

    def spectral_flux(fft_magnitude, previous_fft_magnitude):
        """
        Computes the spectral flux feature of the current frame
        ARGUMENTS:
            fft_magnitude:            the abs(fft) of the current frame
            previous_fft_magnitude:        the abs(fft) of the previous frame
        """
        # compute the spectral flux as the sum of square distances:
        fft_sum = np.sum(fft_magnitude + eps)
        previous_fft_sum = np.sum(previous_fft_magnitude + eps)
        sp_flux = np.sum(
            (fft_magnitude / fft_sum - previous_fft_magnitude /
            previous_fft_sum) ** 2)

        return sp_flux

#### Example Extraction
![](example_extraction_graphs/spectral_flux.png)

### Spectral Roll-Off
#### Description
The spectral roll-off is defined as the frequency under which some percentage of the total energy of the spectrum is contained.

#### Why is this important?
Spectral Roll-off helps differentiate between harmonic content, characterized below the roll-off, and noisy sounds, characterized above the roll-off. //TODO Expand

#### Algorithm


In Python:

    def spectral_rolloff(signal, c):
        """Computes spectral roll-off"""
        energy = np.sum(signal ** 2)
        fft_length = len(signal)
        threshold = c * energy
        # Ffind the spectral rolloff as the frequency position 
        # where the respective spectral energy is equal to c*totalEnergy
        cumulative_sum = np.cumsum(signal ** 2) + eps
        a = np.nonzero(cumulative_sum > threshold)[0]
        if len(a) > 0:
            sp_rolloff = np.float64(a[0]) / (float(fft_length))
        else:
            sp_rolloff = 0.0
        return sp_rolloff

#### Example Extraction
![](example_extraction_graphs/spectral_rolloff.png)

### Mel Frequency Cepstral Coefficients
#### Description
Mel-Frequency Cepstral Coefficient (MFCC) is an important and powerful analytical tool used in speech analysis. The significant of the MFCC is that it the distribution of analyzed frequency bands are non-linear and resembles the logarithmic hearing of humans.
#### Why is this important?
Many studies have linked the importance of MFCC analysis to emotion recognition [1][2][3]. 
#### Algorithm
{\rm Mel}({\rm f})=2595\log_{10}(1+{\rm f}/700)\eqno{\hbox{(1)}}
#### Example Extraction
![](example_extraction_graphs/mfcc_1.png)
![](example_extraction_graphs/mfcc_2.png)
![](example_extraction_graphs/mfcc_3.png)
![](example_extraction_graphs/mfcc_4.png)
![](example_extraction_graphs/mfcc_5.png)
![](example_extraction_graphs/mfcc_6.png)
![](example_extraction_graphs/mfcc_7.png)
![](example_extraction_graphs/mfcc_8.png)
![](example_extraction_graphs/mfcc_9.png)
![](example_extraction_graphs/mfcc_10.png)
![](example_extraction_graphs/mfcc_11.png)
![](example_extraction_graphs/mfcc_12.png)
![](example_extraction_graphs/mfcc_13.png)
### Chroma Vector
#### Description
A chroma vector is an approximation of the pitch class profiles present within a given frame. In music, this can be thought of as a classification of twelve tones.

#### Why is this important?
Chroma vectors allow for the capture of harmonic and melodic characteristics while remaining robust toward changes in timbre and instrumentation.

#### Algorithm

#### Example Extraction
![](example_extraction_graphs/chroma_1.png)
![](example_extraction_graphs/chroma_2.png)
![](example_extraction_graphs/chroma_3.png)
![](example_extraction_graphs/chroma_4.png)
![](example_extraction_graphs/chroma_5.png)
![](example_extraction_graphs/chroma_6.png)
![](example_extraction_graphs/chroma_7.png)
![](example_extraction_graphs/chroma_8.png)
![](example_extraction_graphs/chroma_9.png)
![](example_extraction_graphs/chroma_10.png)
![](example_extraction_graphs/chroma_11.png)
![](example_extraction_graphs/chroma_12.png)

### Chroma Deviation

#### Description
The standard deviation of the 12 chroma coefficients.
#### Why is this important?
This gives insight into the distribution and spread of the spectral energy in the chroma vector.
#### Algorithm

#### Example Extraction
![](example_extraction_graphs/chroma_std.png)

### Chroma Deviation
## Works Cited (WIP)

1.	F. S. A., V. K. V.R., R. S. A., A. Jayakumar and B. A. P., "Speaker Independent Automatic Emotion Recognition from Speech: A Comparison of MFCCs and Discrete Wavelet Transforms," 2009 International Conference on Advances in Recent Technologies in Communication and Computing, Kottayam, Kerala, 2009, pp. 528-531, doi: 10.1109/ARTCom.2009.231.
2.	K. V. Krishna Kishore and P. Krishna Satish, "Emotion recognition in speech using MFCC and wavelet features," 2013 3rd IEEE International Advance Computing Conference (IACC), Ghaziabad, 2013, pp. 842-847, doi: 10.1109/IAdCC.2013.6514336.
3.	https://doi.org/10.1016/j.procs.2015.10.020


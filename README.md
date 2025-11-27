# ITE351-GOYO

## An AI-Based Active Noise Control System for Smart Home Environments

## Members
| Name | Organization | Email |
|------|---------------|-------|
| Taerim Kim | Department of Information Systems, Hanyang University | [trnara5375@gmail.com](mailto:trnara5375@gmail.com) |
| Wongyu Lee | Department of Information Systems, Hanyang University | [onew2370@hanyang.ac.kr](mailto:onew2370@hanyang.ac.kr) |
| Junill Jang | Department of Information Systems, Hanyang University | [jang1161@hanyang.ac.kr](mailto:jang1161@hanyang.ac.kr) |
| Hoyoung Chung | Department of Information Systems, Hanyang University | [sydney010716@gmail.com](mailto:sydney010716@gmail.com) |
## I. Introduction

In the era of increasing telecommuting, residential acoustic comfort has become a critical factor determining the quality of life. Disruptive sounds, such as inter-floor noise or home appliance noise, are factor that degrades the quality of attention in the modern residential environment. To solve this problem, ANC (Active Noise Cancelling) technology using earphones or headsets has been popularized. But the existing wearable-based methods have two critical limitations to use at home.

**First, Physical constraints:** Current ANC technology is limited to the functions through wearable equipment like earphones or headsets. Prolonged usage of such equipment causes physical fatigue, ear pressure, and discomfort. Furthermore, this is impractical during sleep or relaxation, resulting in users discomfort.

**Second, Indiscriminate blocking:** Conventional ANC algorithms block all incoming sounds regardless of the type of noise. This indiscriminate suppression isolates users from their environment, filtering the essential information such as emergency alarms or family members calling. This disconnection poses potential safety hazards and blocks necessary communication.

Therefore, we propose a new form of spatial ANC that creates a quiet zone without the need for wearable devices. Our system intelligently selects and removes only inconvenient noises while preserving essential sounds.

## II. Description of datasets

We constructed a dataset by aggregating high-quality samples from multiple verified open-source datasets(using Freesound.org API Key) to maximize classification accuracy. 

### Data Acquisition Strategy
+ **Air Conditioner:** Samples were extracted from the UrbanSound8K dataset, a widely used benchmark containing 8,732 labeled sound excerpts from urban environments. We selectively used the relevant class to ensure our model is trained on realistic background noise profiles.

+ **Vacuum Cleaner:** Samples were sourced from ESC-50, a collection of 2,000 environmental audio recordings, providing clear and distinct sound signatures essential for accurate detection.

+ **Microwave, Hair Dryer, & Refrigerator:** Due to the scarcity of these specific classes in standard datasets, we collected samples using the Freesound API. We filtered for files with CC0 or CC-BY licenses to ensure copyright compliance.

+ **The Non-target Class:** To prevent false detection in smart home environments, we defined an 'Others' class comprising common sounds that should not trigger the ANC system. This class includes speech, TV audio, and other frequent non-appliance household noises, collected via Freesound to represent a realistic backdrop.

### Spectral Analysis & Validation
To validate the viability of our classification model, we analyzed the spectral signatures of the some collected data.

<img width="522" height="400" alt="image" src="https://github.com/user-attachments/assets/3faabb3d-781f-4242-9bc8-ed8590bd2418" />


As illustrated in above picture, our dataset exhibits clear distinctions that our fine-tuned model can learn:

+ **Target Noises:** Appliances like Hair Dryers and Refrigerators show continuous and stationary patterns in the spectrogram. This indicates consistent energy distribution over time.

+ **Non-Target Sounds:** The 'Others' class exhibits transient and irregular patterns with unstable energy levels.


## III. Methodology

This section describes algorithms and systems designed to accurately categorize household appliance noise in resource-limited edge device environments.

### Core Architecture: MobileNetV1 (YAMNet)
---
Minimizing inference latency is the most critical requirement for our real-time ANC system. Consequently, we selected the MobileNetV1-based **YAMNet architecture** as our backbone. This model replaces traditional heavy CNNs to significantly reduce computational overhead on edge devices.

+ **Algorithmic Efficiency**: MobileNetV1 employs Depthwise Separable Convolutions, a technique that factorizes standard convolutions into two lightweight operations: a depthwise convolution and a pointwise convolution. This structural optimization significantly reduces both the number of parameters and Multiply-Accumulate operations (MACs). This efficiency is crucial for processing 0.5-second audio chunks with minimal latency.
  
+ **Feature Extraction:** We utilized pre-trained YAMNet weights from TensorFlow Hub as a robust feature extractor. To seamlessly integrate this into our pipeline, we encapsulated the model within a custom Keras layer (YAMNetLayer), adapting the architecture classification with our specific audio classes.

```python
class YAMNetLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(YAMNetLayer, self).__init__(**kwargs)
        self.yamnet_tf_function = hub.load('https://tfhub.dev/google/yamnet/1')
        # self.trainable = False

    def call(self, inputs):
        def run_yamnet_on_sample(waveform_1d):
            outputs_tuple = self.yamnet_tf_function(waveform_1d)
            return outputs_tuple[1]

        batch_embeddings = tf.map_fn(
            fn=run_yamnet_on_sample,
            elems=inputs,
            fn_output_signature=tf.TensorSpec(shape=(1, 1024), dtype=tf.float32)
        )
        return batch_embeddings

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1, 1024)
```

### Training Strategy: Imbalance Handling & 2-Phase Optimization
---

Because data within the specialized domain of household appliance noise is inherently sparse, we excountered significant challenges in constructing large-scale datasets to train the model. To overcome the limitations associated with small-scale datasets (i.e., data scarcity and class imbalance) and to enhance the model's generalization capabilities, we designed customized training algorithms as detailed below.

#### A. Class-Aware Augmentation & Weighting

Instead of applying uniform random augmentation, we implemented a conditional augmentation logic that adjusts intensity based on specific class characteristics.

+ **Target Classes (Appliances):** We applied strong augmentation techniques, such as pitch shifting and noise addition. This strategy effectively enriched data diversity and prevented the overfitting caused by limited data.

+ **Non-Target Class (Others)**: Since this class already contains approximately four times more data than each target class and represents the background environment, we employed conservative augmentation to preserve the original acoustic features.

Simultaneously, we computed class weights inversely proportional to the number of samples and applied them to the Cross-Entropy Loss function. This mathematically adjusts the imbalance by ensuring that the model prioritizes learning from underrepresented minority classes.

```python
#Augmentation depending on size of datasets.
def add_noise(audio_data, noise_factor=0.005):
    noise = np.random.randn(len(audio_data))
    augmented_data = audio_data + (np.random.uniform(0.001, noise_factor)) * noise
    augmented_data = augmented_data.astype(type(audio_data[0]))
    return augmented_data

def pitch_shift(audio_data, sample_rate, n_steps=2):
    return librosa.effects.pitch_shift(y=audio_data, sr=sample_rate, n_steps=n_steps)
    
def mask_time(audio_data, t_width_max=1000):
    augmented_data = audio_data.copy()
    num_masks = np.random.randint(1,5)
    for _ in range(num_masks):
        t = np.random.randint(0, augmented_data.shape[0])
        t_width = np.random.randint(1, t_width_max + 1)

        if t + t_width > augmented_data.shape[0]:
            t_width = augmented_data.shape[0] - t
        augmented_data[t:t+t_width] = 0
    return augmented_data

def mask_freq(wav_data, f_width_max=10):
    stft = librosa.stft(wav_data)
    f_count_max = stft.shape[0]
    num_masks = np.random.randint(1, 5)
    
    for _ in range(num_masks):
        f = np.random.randint(0, f_count_max)
        f_width = np.random.randint(1, f_width_max + 1)
    
        if f + f_width > stft.shape[0]:
            f_width = stft.shape[0] - f
        stft[f:f+f_width, :] = 0
    return librosa.istft(stft)
```

```python
if self.augment:
  current_class = self.class_names[label]

  if current_class != 'Others':
      if np.random.rand() > 0.5:
          wav_data = add_noise(wav_data)
      if np.random.rand() > 0.5:
          wav_data = pitch_shift(wav_data, self.sample_rate, n_steps=np.random.randint(-2, 3))
      if np.random.rand() > 0.7:
          wav_data = mask_time(wav_data)
      if np.random.rand() > 0.7:
          wav_data = mask_freq(wav_data)

  else:
      if np.random.rand() > 0.8:
          wav_data = add_noise(wav_data)
      if np.random.rand() > 0.8:
          wav_data = pitch_shift(wav_data, self.sample_rate, n_steps=np.random.randint(-2, 3))
      if np.random.rand() > 0.8:
          wav_data = mask_time(wav_data)
      if np.random.rand() > 0.8:
          wav_data = mask_freq(wav_data)
```
```python
#weighting depending on size of datasets of each classes.
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(train_labels),
    y=train_labels
)
class_weight_dict = {i : class_weights[i] for i in range(len(class_weights))}
```

#### B. 2-Phase Fine-Tuning Protocol
To prevent the risk of **Catastrophic Forgetting** inherent in transfer learning, we implemented a strict two-phase optimization protocol.

+ **Phase 1 (First 20epochs):** We freeze the backbone parameters and train only the custom classifier head using a relatively high learning rate ($10^{-3}$). This step initializes the weights of the dense layers before adjusting the feature extractor.
+ 
+ **Phase 2:** We unfreeze the backbone for end-to-end adaptation. Crucially, we keep the Batch Normalization (BN) layers frozen while applying a very low learning rate ($10^{-5}$). This strategy preserves the robust statistical feature distributions learned from the large-scale source dataset while effectively adapting the model to the specific acoustic signatures of household appliances.
 
```python
#2-Phase Fine-Tuning
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

model.fit(
    train_generator,
    epochs=20,
    class_weight=class_weight_dict,
    validation_data=val_generator,
    callbacks=[checkpoint_cb],
    steps_per_epoch=len(train_generator),
    validation_steps=len(val_generator)
)

print("[Phase 2](Unfreeze Backbone)")

yamnet_found = False
for layer in model.layers:
    if 'yamnet' in layer.name.lower() or 'YAMNetLayer' in str(type(layer)):
        layer.trainable = True
        yamnet_found = True
        print(f"-> Unfrozen Layer: {layer.name}")

if not yamnet_found:
    print("error: YAMNet 레이어를 찾지 못했습니다.")
    model.trainable = True

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

model.fit(
    train_generator,
    initial_epoch=20,
    epochs=100,
    class_weight=class_weight_dict,
    validation_data=val_generator,
    callbacks=[checkpoint_cb, early_stop],
    steps_per_epoch=len(train_generator),
    validation_steps=len(val_generator)
)
```

### Real-time Inference & Control Algorithm
---
In dynamic real-world environments, relying solely on the instantaneous probability of a single inference step poses a high risk of false positives. To address this, we designed a Dual-Stage Filtering Pipeline to statistically verify the reliability of the system.

#### A. Optimization: Decibel-based VAD Gating (Pre-filtering)
To maximize the efficiency of resource-constrained edge devices, we layed out a Db VAD(Voice Activity Detection) state machine prior to the deep learning inference.

+ The input audio stream is monitored in real-time using 0.5s chunks, and the average decibel level is calculated for each chunk. If the calculated dB value is lower than a predefined threshold (e.g., 55dB), the signal is considered insignificant 'background noise' and is immediately dropped. This acts as a computational gate, preventing the deep learning model from running during silence, thereby conserving computational resources and power.

#### B. Stability: Spatio-Temporal Consistency Filtering
For valid signals passing the VAD gate, we applied a Spatio-Temporal Consistency Algorithm that integrates spatial location information with temporal continuity to generate the final control signal.
+ Sliding Window Buffering
  + The system utilizes a Sliding Window technique to capture continuous context rather than relying on a single data point.
  + By striding every $0.5s$, it generates $1.0s$ overlapping windows and loads them into a FIFO (First-In-First-Out) queue, constructing a time-series buffer ($N=5$ chunks) representing the most recent 3 seconds.
+ Majority Voting Logic (Temporal Consistency):
  + The five independent audio chunks stored in the buffer are each processed by the deep learning model, returning five predicted classes ($C_{pred}$).
  + The system compares these predictions with the Spatial ID (Target Class, $C_{target}$) assigned to the specific microphone to calculate the number of matches.
  + Finally, a control signal is generated only if the condition "Is the same target noise detected in 4 or more out of 5 independent trials ($\ge 80\%$) within a 3-second window ($T=3s$)?" is met.


> $$Trigger = \begin{cases} \text{True (ON)} & \text{if } \sum_{i=1}^{5} \mathbb{I}(C_{pred}^{(i)} == C_{target}) \ge 4 \\ 
\text{False (OFF)} & \text{otherwise} \end{cases}$$

## IV. Evaluation & Analysis
### Comparative Experiment Setup
To determine the optimal backbone for our real-time ANC system, we implemented and evaluated two candidate models: YAMNet (MobileNetV1-based) and PANNs (Cnn14-based). Both models were fine-tuned under identical conditions using our custom dataset, and their performance was measured across four key engineering metrics: Accuracy, Latency, Model Parameters, and Storage Size.
#### Quantitative Results & Analysis
<img width="790" height="516" alt="image" src="https://github.com/user-attachments/assets/517e1227-3ac0-4d18-aa95-a8b70207f16c" />

As illustrated Performance Benchmark, the experimental results clearly demonstrate the trade-offs between the two architectures.

**Accuracy:** YAMNet achieved a validation accuracy of 84.21%, outperforming PANNs (63.54%) by approximately **20.7%p**. This indicates that YAMNet is more effective at extracting features from short-duration audio clips typical of appliance noise.

**Inference Latency:** In terms of inference speed, YAMNet recorded an inference time of 34.17 ms, which is **6.3 times faster** than PANNs (215.12 ms). This low latency is critical for ensuring the ANC system reacts instantaneously to noise events.

**Model Efficiency:**
+ Parameters: YAMNet (4.01 M) has **20 times fewer** parameters compared to PANNs (81.29 M).
+ Storage Size: Consequently, the model file size of YAMNet is only 3.05 MB, making it **102 times lighter** than PANNs (311.2 MB).

#### Why we chose YAMNet?
The significant performance gap can be attributed to the architectural differences and input length mismatch.

**Input Sensitivity:** PANNs is originally designed for long-context audio (~10s). When fed with our system's short 1-second buffers, it struggles to capture sufficient temporal context, leading to lower accuracy despite its larger capacity.

**Computational Load:** The standard CNN architecture of PANNs incurs a heavy computational burden, resulting in high latency (~215ms) that exceeds the real-time processing budget. In contrast, YAMNet's Depthwise Separable Convolutions efficiently handle the workload, maintaining high accuracy with minimal delay.

---

### Qualitative Analysis: Confusion Matrix & Limitation

To verify the model's reliability, we analyzed the ‘Confusion Matrix’ to understand the decision boundaries between classes.
<img width="624" height="517" alt="image" src="https://github.com/user-attachments/assets/949a85ae-3d6d-4f0f-8bb2-980781f39364" />

**High Safety Assurance (Non-target Class):** The most critical requirement for our system is safety, preventing false triggers on human speech or  ambient sounds. The model achieved a high accuracy of 89.7% for the 'Others' class. This explains a robust capability to distinguish essential environmental sounds from target noise.

**Analysis of Misclassification (Air Conditioner and Refrigerator):** As observed in the matrix, there is a noticeable confusion between the 'Air Conditioner' and 'Refrigerator' classes (20-25% misclassification rate).

+ **Reasoning:** This ambiguity is caused by the high spectral similarity between the two sources. Due to the strong spectral similarity in their low-frequency hums, these appliances lack distinguishing features, making them acoustically ambiguous to both the model and human.

+ **Engineering Impact:** While both classes share unrelieved low-frequency characteristics, precise distinction remains a critical objective for maximizing noise cancellation efficiency. Differentiating between two classes would enable the deployment of device-specific ANC profiles, targeting the unique harmonic peaks of each appliance. Therefore, resolving this spectral ambiguity through advanced feature extraction or increased data diversity is identified as a key direction for future system optimization.

## V. Related Work
### Foundational Studies (Theoretical Background)
---
This project is built upon state-of-the-art research in efficient deep learning and audio event classification. We leveraged transfer learning from large-scale pre-trained models to ensure both reliability and real-time performance.
#### MobileNets (Backbone Architecture)
**[1] A. G. Howard et al., "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications," arXiv preprint arXiv:1704.04861, 2017.**
+ Relevance: This paper introduces the MobileNetV1 architecture, which utilizes depthwise separable convolutions to drastically reduce computational cost ($MACs$). We cited this work to justify our choice of YAMNet as a lightweight backbone suitable for low-latency inference on edge devices.
#### PANNs (Comparative Benchmark)
**[2] Q. Kong et al., "PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition," IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 28, pp. 2880-2894, 2020.**
+ Relevance: This study proposes Cnn14 (PANNs), a state-of-the-art audio classification model. We utilized this model as a comparative benchmark to analyze the trade-offs between model size, accuracy, and latency, ultimately confirming that lighter models are more effective for our specific ANC application.
#### AudioSet (Pre-training Dataset)
**[3] J. F. Gemmeke et al., "Audio Set: An ontology and human-labeled dataset for audio events," in Proc. IEEE ICASSP, 2017, pp. 776-780.**
+ Relevance: This paper details the large-scale dataset used to pre-train the YAMNet model. Referencing this validates that our model possesses a robust foundational understanding of general audio features before we applied transfer learning with our custom appliance dataset.


### Implementation Tools & Libraries
---
The system implementation relies on standard open-source libraries for deep learning and audio signal processing to ensure reproducibility and stability.
#### Deep Learning Frameworks
+ **TensorFlow & Keras:** Utilized as the primary framework for building the custom classifier head, managing the 2-Phase Fine-Tuning loop, and executing model optimization for edge deployment.
+ **PyTorch:** Used to implement and evaluate the PANNs (Cnn14) model for performance benchmarking.
#### Audio Processing
+ **Librosa:** Utilized for core audio preprocessing tasks, including loading audio files, resampling to 16kHz, trimming silence ($top\_db=30$), and generating spectrograms for analysis.
+ **SoundDevice:** Integrated for real-time low-latency audio stream acquisition from distributed reference microphones.
#### Data Acquisition
+ **Freesound API:** Used to programmatically crawl and filter high-quality datasets based on specific query tags and CC0/CC-BY licenses to address data scarcity.
## VI. Conclusion

In this project, we successfully developed a robust real-time audio classification framework for selective Active Noise Cancelling (ANC) in smart home environments.

By leveraging Transfer Learning with the MobileNetV1 (YAMNet) architecture, we addressed the limitations of traditional ANC systems, achieving high classification accuracy (84.2%) while maintaining low latency (32ms) suitable for edge devices. We overcame data scarcity through Class-Aware Augmentation and 2-Phase Fine-Tuning, ensuring the model's generalization capability. Furthermore, the integration of a Multi-stage Filtering algorithm significantly enhanced system reliability by virtually eliminating false positives caused by general environment noise.

This work demonstrates that selective noise cancellation is not only feasible but also highly efficient when combining optimized deep learning models with rigorous system-level filtering logic. Future work may involve expanding the target class categories and deploying the system on various hardware platforms to further validate its scalability.


**Conclusion:** This visual evidence confirms that our dataset contains sufficient feature disparities for the model to effectively distinguish between constant mechanical hums and dynamic environmental sounds.

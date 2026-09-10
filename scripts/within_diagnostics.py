"""Controlled window/stride experiments and traceable within-subject failures."""
import joblib
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits

# %% Configuration
Config.WINDOWS_MS = [200, 400, 600]
Config.STRIDES_MS = [50, 100, 200]
Config.SMOKE = False
Config.GATE_CNN_THRESHOLD = 0.70
Config.GATE_EMG_THRESHOLD = 0.80
Config.EVALUATION_GRID_MS = 50
Config.REFIT_ON_TRAIN_PLUS_VAL = False


def load_diagnostic_subject(subject, output):
    """Read one recording; preserve raw signals, labels and repetition provenance."""
    loader = SubjectLoader()
    parts, acc_channels = loader._load_raw_parts_from_source(subject)
    assert len(parts) == 1 and acc_channels == 36
    part = parts[0]
    raw_emg, acc, labels = part['emg'], part['acc'], part['labels']
    assert len(raw_emg) == len(acc) == len(labels)
    assert np.isfinite(raw_emg).all() and np.isfinite(acc).all()
    assert set(np.unique(labels)) == set(range(18))
    extras = io.loadmat(loader._selected_files(subject)[0], variable_names=['stimulus', 'subject', 'exercise'])
    stimulus = np.asarray(extras['stimulus']).reshape(-1)
    assert len(stimulus) == len(labels)
    rows = []
    signal = np.empty((len(labels), 48), dtype=np.float32)
    signal[:, :12] = 0  # Rest is never sampled by WindowDataset.
    signal[:, 12:] = acc
    for gesture, repetitions in loader._collect_repetitions(parts).items():
        assert len(repetitions) == 6
        train_ids, val_ids, test_ids = loader._split_repetition_indices(subject, gesture)
        native_seen = set()
        for index, repetition in enumerate(repetitions):
            start, end = repetition['start'], repetition['end']
            native = np.unique(part['native_repetition'][start:end])
            assert len(native) == 1 and 1 <= int(native[0]) <= 6 and int(native[0]) not in native_seen
            native_seen.add(int(native[0]))
            split = 'train' if index in train_ids else 'validation' if index in val_ids else 'test'
            signal[start:end, :12] = loader.rep_filter.apply(raw_emg[start:end])
            rows.append(dict(subject=subject, gesture=gesture, native_repetition=int(native[0]),
                             split=split, file_name=part['file_name'],
                             run_start=start, run_end=end))
    metadata = pd.DataFrame(rows)
    metadata.to_csv(output/'repetition_inventory.csv', index=False)
    json_write(output/'identity.json', dict(folder_subject=subject,
               internal_subject=int(extras['subject'].item()), exercise=int(extras['exercise'].item()),
               signal_shape=list(signal.shape), label_key=Config.LBL_KEY,
               raw_emg_sha256=hashlib.sha256(raw_emg.tobytes()).hexdigest()))
    boundaries = np.unique(np.r_[0, np.flatnonzero(np.diff(labels))+1,
                                 np.flatnonzero(np.diff(stimulus))+1, len(labels)])
    pd.DataFrame(dict(start=boundaries[:-1], end=boundaries[1:],
                     stimulus=stimulus[boundaries[:-1]], restimulus=labels[boundaries[:-1]])
                ).to_csv(output/'stimulus_restimulus_intervals.csv', index=False)
    return dict(signal=signal, raw_emg=raw_emg, stimulus=stimulus,
                restimulus=labels, metadata=metadata)


class WindowDataset(Dataset):
    """One subject, split and window/stride configuration; no split crosses a repetition."""
    def __init__(self, recording, split, window_ms, stride_ms, center_acc=False):
        self.recording = recording
        self.window_ms = int(window_ms)
        self.stride_ms = int(stride_ms)
        self.samples = 2 * self.window_ms
        self.center_acc = bool(center_acc)
        self.mean = np.zeros((48, 1), np.float32)
        self.std = np.ones((48, 1), np.float32)
        rows = []
        for repetition in recording['metadata'].to_dict('records'):
            if repetition['split'] != split:
                continue
            first_end = repetition['run_start'] + 200 + self.samples
            last_end = repetition['run_end'] - 200
            assert first_end <= last_end
            for end in range(first_end, last_end + 1, 2 * self.stride_ms):
                start = end - self.samples
                rows.append(dict(**repetition, window_start=start, window_end=end,
                    window_ms=self.window_ms, grid_stride_ms=self.stride_ms,
                    endpoint_ms=(end-repetition['run_start'])/2,
                    phase_fraction=((start+end)/2-repetition['run_start']) /
                                   (repetition['run_end']-repetition['run_start']),
                    stimulus_disagreement_fraction=float(np.mean(
                        recording['stimulus'][start:end] != repetition['gesture']))))
        self.meta = pd.DataFrame(rows)
        self.meta['phase'] = pd.cut(self.meta.phase_fraction, [0, 1/3, 2/3, 1],
                                    labels=['early', 'middle', 'late'], include_lowest=True).astype(str)
        self.y = self.meta.gesture.to_numpy(dtype=np.int64) - 1
        self.starts = self.meta.window_start.to_numpy(dtype=np.int64)

    def __len__(self):
        return len(self.y)

    def physical(self, index):
        start = self.starts[index]
        return self.recording['signal'][start:start+self.samples].T.copy()

    def raw(self, index):
        signal = self.physical(index)
        if self.center_acc:
            signal[12:] -= signal[12:].mean(axis=1, keepdims=True)
        return signal

    def __getitem__(self, index):
        signal = ((self.raw(index)-self.mean)/self.std).astype(np.float32)
        return torch.from_numpy(signal), int(self.y[index])


def fit_window_normalizer(training):
    """Fit only on actual training input windows, after optional ACC centering."""
    total = np.zeros(48, np.float64)
    squares = np.zeros(48, np.float64)
    count = 0
    for begin in range(0, len(training), 64):
        signal = np.stack([training.raw(i) for i in range(begin, min(begin+64, len(training)))]).astype(np.float64)
        total += signal.sum(axis=(0, 2))
        squares += np.square(signal).sum(axis=(0, 2))
        count += signal.shape[0]*signal.shape[2]
    mean = total/count
    std = np.sqrt(np.maximum(squares/count-mean*mean, 0))+1e-8
    return mean.astype(np.float32)[:, None], std.astype(np.float32)[:, None]

# %% Physical signal features and classical EMG classifier
def diagnostic_features(dataset):
    """Keep the first 72 features identical to the previous EMG gate classifier."""
    blocks = []
    for begin in range(0, len(dataset), 64):
        ids = range(begin, min(begin+64, len(dataset)))
        signal = np.stack([dataset.physical(i) for i in ids]).astype(np.float64)
        emg, acc = signal[:, :12], signal[:, 12:]
        difference = np.diff(emg, axis=-1)
        power = np.abs(np.fft.rfft(emg, axis=-1))**2
        frequency = np.fft.rfftfreq(emg.shape[-1], 1/Config.EMG_FS)
        total_power = power.sum(-1)+1e-30
        rms = np.sqrt(np.mean(emg**2, -1))
        features = [rms, np.mean(abs(emg), -1), np.mean(abs(difference), -1),
            np.mean(emg[:, :, :-1]*emg[:, :, 1:] < 0, -1),
            (power*frequency).sum(-1)/total_power,
            frequency[np.argmax(np.cumsum(power, axis=-1) >= total_power[:, :, None]/2, axis=-1)],
            acc.mean(-1), acc.std(-1), np.mean(abs(np.diff(acc, axis=-1)), -1),
            rms/(rms.sum(1, keepdims=True)+1e-30),
            np.max(abs(emg), -1)/(rms+1e-30),
            power[:, :, (frequency >= 20)&(frequency < 60)].sum(-1)/total_power,
            power[:, :, (frequency >= 250)&(frequency <= 450)].sum(-1)/total_power]
        raw = np.stack([dataset.recording['raw_emg'][dataset.starts[i]:dataset.starts[i]+dataset.samples].T for i in ids]).astype(np.float64)
        raw_rms = np.sqrt(np.mean(raw**2, -1))
        features.extend([raw_rms, np.max(abs(raw), -1),
                         np.mean(np.diff(raw, axis=-1) == 0, -1),
                         np.mean(raw == raw.max(-1, keepdims=True), -1),
                         np.sqrt(np.mean(emg[:, :, :emg.shape[-1]//2]**2, -1))/(rms+1e-30),
                         np.sqrt(np.mean(emg[:, :, emg.shape[-1]//2:]**2, -1))/(rms+1e-30)])
        blocks.append(np.concatenate(features, axis=1).astype(np.float32))
    emg_names = ['rms', 'mav', 'wl_mean', 'zc_fraction', 'mean_frequency', 'median_frequency']
    names = [f'emg_{name}_{ch+1:02}' for name in emg_names for ch in range(12)]
    names += [f'acc_{name}_{ch+1:02}' for name in ['mean', 'std', 'mean_abs_diff'] for ch in range(36)]
    extras = ['relative_rms', 'crest_factor', 'power_20_60_fraction', 'power_250_450_fraction',
              'raw_rms', 'raw_abs_peak', 'raw_flat_fraction', 'raw_max_repeat_fraction',
              'first_half_rms_ratio', 'second_half_rms_ratio']
    names += [f'emg_{name}_{ch+1:02}' for name in extras for ch in range(12)]
    features = np.concatenate(blocks)
    assert features.shape[1] == len(names) and np.isfinite(features).all()
    return features, names


def fit_emg_classifier(training_features, training_labels):
    classifier = make_pipeline(StandardScaler(),
        LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
    with threadpool_limits(limits=2):
        classifier.fit(training_features[:, :72], training_labels)
    assert np.array_equal(classifier.classes_, np.arange(17))
    return classifier

# %% CNN fit and evaluation (same architecture and checkpoint rule)
def fit_diagnostic_cnn(training, validation, output, subject):
    seed = 42 + 1009*subject
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    mean, std = fit_window_normalizer(training)
    training.mean = validation.mean = mean
    training.std = validation.std = std
    np.savez_compressed(output/'normalizer.npz', mean=mean, std=std)
    model = OriginalMultiKernelAttention1DCNN(48, 17, Config.DROPOUT)
    trainer = DiagnosticTrainer(model, output/'best_model.pt', 17)
    training_loader = DataLoader(training, batch_size=128, shuffle=True,
                                generator=torch.Generator().manual_seed(seed))
    validation_loader = DataLoader(validation, batch_size=128, shuffle=False)
    trainer.fit(training_loader, validation_loader)
    learning_report(trainer, output)
    model.load_state_dict(torch.load(output/'best_model.pt', map_location=Config.DEVICE, weights_only=True))
    json_write(output/'fit_manifest.json', dict(subject=subject, window_ms=training.window_ms,
        training_stride_ms=training.stride_ms, acc_centering=training.center_acc, seed=seed,
        train_windows=len(training), validation_windows=len(validation),
        selected_epoch=trainer.best_epoch, epochs_run=len(trainer.epoch_rows),
        optimizer_steps=len(training_loader)*len(trainer.epoch_rows),
        parameters=sum(parameter.numel() for parameter in model.parameters()),
        no_refit=True, window_selection=False, gate_thresholds=[.70, .80]))
    return model


def predict_with_diagnostics(model, dataset, retain_attention=False):
    """Evaluate the checkpoint; collect embeddings and optionally attention on every window."""
    model.eval()
    captured, probabilities, embeddings, attention = {}, [], [], {}
    def capture(name):
        def hook(module, args, result):
            captured[name] = result.detach()
        return hook
    handles = [model.temporal_pool.register_forward_hook(capture('embedding'))]
    if retain_attention:
        handles += [model.temporal_pool.score.register_forward_hook(capture('time_scores'))]
        for stage in [1, 2, 3]:
            handles.append(getattr(model, f'attn{stage}').attention.register_forward_hook(capture(f'channel_stage{stage}')))
    try:
        with torch.no_grad():
            for signal, _ in DataLoader(dataset, batch_size=128, shuffle=False):
                logits = model(signal.to(Config.DEVICE))
                probabilities.append(logits.softmax(1).cpu().numpy())
                embeddings.append(captured['embedding'].cpu().numpy())
                if retain_attention:
                    for name in ['time_scores', 'channel_stage1', 'channel_stage2', 'channel_stage3']:
                        values = captured[name].softmax(-1) if name == 'time_scores' else captured[name]
                        attention.setdefault(name, []).append(values.squeeze(1 if name == 'time_scores' else -1).cpu().numpy())
    finally:
        for handle in handles:
            handle.remove()
    return np.concatenate(probabilities), np.concatenate(embeddings), {k:np.concatenate(v) for k,v in attention.items()}


def evaluation_masks(metadata, window_ms, training_stride):
    elapsed = metadata.endpoint_ms.to_numpy() - (100+window_ms)
    return dict(primary=(elapsed % training_stride == 0),
                common=(metadata.endpoint_ms.to_numpy() >= 700) &
                       ((metadata.endpoint_ms.to_numpy()-700) % 200 == 0),
                dense=np.ones(len(metadata), dtype=bool))


def save_predictions(dataset, cnn_probability, emg_probability, embeddings, attention,
                     features, feature_names, output, training_stride, split):
    output.mkdir(parents=True, exist_ok=True)
    assert np.isfinite(cnn_probability).all() and np.isfinite(emg_probability).all()
    assert np.allclose(cnn_probability.sum(1), 1, atol=1e-5)
    switch = (cnn_probability.max(1) < Config.GATE_CNN_THRESHOLD) & (emg_probability.max(1) > Config.GATE_EMG_THRESHOLD)
    gate_probability = np.where(switch[:, None], emg_probability, cnn_probability)
    frame = dataset.meta.copy()
    arms = {'cnn':cnn_probability, 'emg':emg_probability, 'gate':gate_probability}
    for arm, probability in arms.items():
        frame[f'{arm}_prediction'] = probability.argmax(1)+1
        frame[f'{arm}_confidence'] = probability.max(1)
        frame[f'{arm}_correct'] = probability.argmax(1) == dataset.y
        frame[f'{arm}_entropy'] = -(probability*np.log(np.maximum(probability, 1e-12))).sum(1)
    ordered = np.sort(cnn_probability, axis=1)
    frame['cnn_top2_margin'] = ordered[:, -1]-ordered[:, -2]
    frame['switch_to_emg'] = switch
    frame['recovered'] = ~frame.cnn_correct & frame.gate_correct
    frame['harmed'] = frame.cnn_correct & ~frame.gate_correct
    frame['both_wrong'] = ~frame.cnn_correct & ~frame.emg_correct
    masks = evaluation_masks(frame, dataset.window_ms, training_stride) if split == 'test' else {'primary':np.ones(len(frame), bool)}
    for name, mask in masks.items():
        frame[f'evaluate_{name}'] = mask
    frame.to_csv(output/'predictions.csv', index=False)
    np.savez_compressed(output/'probabilities.npz', y_true=dataset.y, **arms)
    np.savez_compressed(output/'features_embeddings.npz', features=features,
                        feature_names=np.asarray(feature_names), embeddings=embeddings)
    if attention:
        np.savez_compressed(output/'attention.npz', **attention)
    rows, gestures = [], []
    for grid, mask in masks.items():
        for arm, probability in arms.items():
            metrics, calibration = probability_metrics(dataset.y[mask], probability[mask])
            rows.append(dict(grid=grid, arm=arm, **metrics))
            pd.DataFrame(calibration).to_csv(output/f'calibration_{grid}_{arm}.csv', index=False)
            pd.DataFrame(confusion_matrix(dataset.y[mask], probability[mask].argmax(1), labels=range(17)),
                         index=range(1,18), columns=range(1,18)).to_csv(output/f'confusion_{grid}_{arm}.csv')
        for gesture, part in frame[mask].groupby('gesture'):
            wrong = part[~part.cnn_correct].cnn_prediction
            gestures.append(dict(grid=grid, gesture=gesture, windows=len(part),
                cnn_correct=int(part.cnn_correct.sum()), cnn_wrong=int((~part.cnn_correct).sum()),
                cnn_error_percent=100*float((~part.cnn_correct).mean()),
                gate_correct=int(part.gate_correct.sum()), gate_wrong=int((~part.gate_correct).sum()),
                gate_error_percent=100*float((~part.gate_correct).mean()),
                recovered=int(part.recovered.sum()), harmed=int(part.harmed.sum()),
                both_wrong=int(part.both_wrong.sum()),
                most_common_wrong_gesture=int(wrong.mode().iloc[0]) if len(wrong) else None))
    pd.DataFrame(rows).to_csv(output/'metrics.csv', index=False)
    pd.DataFrame(gestures).to_csv(output/'gesture_errors.csv', index=False)
    if split == 'test':
        primary = pd.DataFrame(gestures).query("grid == 'primary'")
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        for ax, arm in zip(axes, ['cnn', 'gate']):
            ax.bar(primary.gesture, primary[f'{arm}_correct'], label='Correct', color='#25867a')
            ax.bar(primary.gesture, primary[f'{arm}_wrong'], bottom=primary[f'{arm}_correct'], label='Wrong', color='#ce604c')
            ax.set(title=f'{arm.upper()}: test windows per gesture', xlabel='Gesture', ylabel='Window count', xticks=range(1,18))
            ax.legend()
        save_figure(fig, output/'gesture_error_bars.png')
    return frame

# %% Failure diagnosis: comparisons within the same subject, gesture and phase
def feature_success_failure_comparison(frame, features, names, training_features, output):
    """Descriptive effects, not significance tests on correlated overlapping windows."""
    rows = []
    train_scale = training_features.std(0)+1e-12
    primary = frame[frame.evaluate_primary]
    for phase in ['all', 'early', 'middle', 'late']:
        selected = primary if phase == 'all' else primary[primary.phase == phase]
        for gesture, part in selected.groupby('gesture'):
            good_ids = part.index[part.cnn_correct].to_numpy()
            bad_ids = part.index[~part.cnn_correct].to_numpy()
            if not len(good_ids) or not len(bad_ids):
                continue
            good, bad = features[good_ids], features[bad_ids]
            for column, name in enumerate(names):
                good_median, bad_median = np.median(good[:, column]), np.median(bad[:, column])
                rows.append(dict(gesture=gesture, phase=phase, feature=name,
                    correct_windows=len(good_ids), wrong_windows=len(bad_ids),
                    correct_mean=float(good[:, column].mean()), wrong_mean=float(bad[:, column].mean()),
                    correct_median=float(good_median), wrong_median=float(bad_median),
                    median_difference_training_sd=float((bad_median-good_median)/train_scale[column])))
    pd.DataFrame(rows, columns=['gesture','phase','feature','correct_windows','wrong_windows',
        'correct_mean','wrong_mean','correct_median','wrong_median','median_difference_training_sd']).to_csv(output/'feature_correct_wrong.csv', index=False)


def nearest_training_examples(training_values, test_values, train_meta, test_frame, space):
    """Distances use training-fitted scaling; true-label queries are diagnostic only."""
    scaler = StandardScaler().fit(training_values)
    train = scaler.transform(training_values)
    test = scaler.transform(test_values)
    all_distances, all_indices = [], []
    for gesture in range(1, 18):
        ids = np.flatnonzero(train_meta.gesture.to_numpy() == gesture)
        nearest = NearestNeighbors(n_neighbors=1, algorithm='brute', n_jobs=1).fit(train[ids])
        distances, neighbors = nearest.kneighbors(test)
        all_distances.append(distances[:, 0])
        all_indices.append(ids[neighbors[:, 0]])
    distances = np.stack(all_distances, axis=1)
    indices = np.stack(all_indices, axis=1)
    row_ids = np.arange(len(test))
    true = test_frame.gesture.to_numpy()-1
    predicted = test_frame.cnn_prediction.to_numpy()-1
    nearest_class = distances.argmin(1)
    nearest_ids = indices[row_ids, nearest_class]
    result = test_frame[['subject','gesture','native_repetition','window_start','window_end','cnn_correct','evaluate_primary','phase']].copy()
    result['space'] = space
    result['true_gesture_distance'] = distances[row_ids, true]
    result['cnn_predicted_gesture_distance'] = distances[row_ids, predicted]
    result['nearest_training_gesture'] = nearest_class+1
    result['nearest_training_row'] = nearest_ids
    result['nearest_training_repetition'] = train_meta.iloc[nearest_ids].native_repetition.to_numpy()
    result['nearest_training_window_start'] = train_meta.iloc[nearest_ids].window_start.to_numpy()
    other = distances.copy()
    other[row_ids, true] = np.inf
    result['nearest_other_distance'] = other.min(1)
    result['true_vs_other_distance_margin'] = result.nearest_other_distance-result.true_gesture_distance
    return result


def error_timing_and_stride(frame, window_ms, output):
    phase_rows, streak_rows, stride_rows = [], [], []
    primary = frame[frame.evaluate_primary]
    for (gesture, phase), part in primary.groupby(['gesture', 'phase']):
        phase_rows.append(dict(gesture=gesture, phase=phase, windows=len(part),
            cnn_wrong=int((~part.cnn_correct).sum()), gate_wrong=int((~part.gate_correct).sum()),
            both_wrong=int(part.both_wrong.sum())))
    for (gesture, repetition), part in primary.groupby(['gesture','native_repetition']):
        part = part.sort_values('window_end')
        longest = current = 0
        for wrong in (~part.cnn_correct).to_numpy():
            current = current+1 if wrong else 0
            longest = max(longest, current)
        step_ms = float(np.diff(part.endpoint_ms).min()) if len(part)>1 else 0
        streak_rows.append(dict(gesture=gesture, repetition=repetition, windows=len(part),
            cnn_error_fraction=float((~part.cnn_correct).mean()), both_wrong_fraction=float(part.both_wrong.mean()),
            longest_error_streak_windows=longest, longest_error_endpoint_span_ms=max(0,longest-1)*step_ms,
            all_cnn_wrong=bool((~part.cnn_correct).all())))
    for stride in [50,100,200]:
        for offset in range(0,stride,50):
            mask = ((frame.endpoint_ms-(100+window_ms)-offset) % stride == 0)
            part = frame[mask]
            stride_rows.append(dict(output_stride_ms=stride, offset_ms=offset, windows=len(part),
                cnn_accuracy=float(part.cnn_correct.mean()), gate_accuracy=float(part.gate_correct.mean()),
                note='Same fitted model; output subsampling only'))
    pd.DataFrame(phase_rows).to_csv(output/'phase_errors.csv', index=False)
    pd.DataFrame(streak_rows).to_csv(output/'repetition_failures.csv', index=False)
    pd.DataFrame(stride_rows).to_csv(output/'output_stride_offsets.csv', index=False)


def export_waveform_cases(dataset, frame, output):
    """One correct and one wrong primary-grid example per gesture, selected by confidence."""
    signals, raw, stimuli, labels, rows = [], [], [], [], []
    for gesture, part in frame[frame.evaluate_primary].groupby('gesture'):
        for correct in [True, False]:
            candidates = part[part.cnn_correct == correct].sort_values(['cnn_confidence','window_start'], ascending=[False,True])
            if candidates.empty:
                continue
            index = int(candidates.index[0])
            start = int(dataset.starts[index])
            signals.append(dataset.physical(index))
            raw.append(dataset.recording['raw_emg'][start:start+dataset.samples].T)
            stimuli.append(dataset.recording['stimulus'][start:start+dataset.samples])
            labels.append(dataset.recording['restimulus'][start:start+dataset.samples])
            rows.append(dict(case=len(rows), test_row=index, **frame.iloc[index].to_dict()))
    np.savez_compressed(output/'waveform_cases.npz', filtered_emg_and_acc=np.stack(signals),
                        raw_emg=np.stack(raw), stimulus=np.stack(stimuli), restimulus=np.stack(labels))
    pd.DataFrame(rows).to_csv(output/'waveform_case_index.csv', index=False)

# %% Complete experiment for one subject, one window and one training stride
def run_subject_configuration(recording, subject, window, stride, output):
    datasets = {split:WindowDataset(recording, split, window,
                Config.EVALUATION_GRID_MS if split == 'test' else stride)
                for split in ['train','validation','test']}
    features = {}
    for split, dataset in datasets.items():
        features[split], feature_names = diagnostic_features(dataset)
    classifier = fit_emg_classifier(features['train'], datasets['train'].y)
    joblib.dump(classifier, output/'emg_classifier.joblib')
    emg_probability = {split:classifier.predict_proba(values[:, :72]) for split,values in features.items()}
    variant_frames = {}
    for variant, centered in [('original_acc', False), ('centered_acc', True)]:
        folder = output/variant
        folder.mkdir()
        for dataset in datasets.values():
            dataset.center_acc = centered
        model = fit_diagnostic_cnn(datasets['train'], datasets['validation'], folder, subject)
        datasets['test'].mean = datasets['train'].mean
        datasets['test'].std = datasets['train'].std
        frames, embeddings = {}, {}
        for split, dataset in datasets.items():
            probability, embeddings[split], attention = predict_with_diagnostics(model, dataset, retain_attention=split=='test')
            frames[split] = save_predictions(dataset, probability, emg_probability[split], embeddings[split],
                attention, features[split], feature_names, folder/split, stride, split)
        diagnostic_dir = folder/'test'
        test_frame = frames['test']
        feature_success_failure_comparison(test_frame, features['test'], feature_names, features['train'], diagnostic_dir)
        with threadpool_limits(limits=2):
            physical = nearest_training_examples(features['train'], features['test'], datasets['train'].meta, test_frame, 'physical_features')
            learned = nearest_training_examples(embeddings['train'], embeddings['test'], datasets['train'].meta, test_frame, 'cnn_embedding')
        pd.concat([physical, learned]).to_csv(diagnostic_dir/'training_similarity.csv', index=False)
        error_timing_and_stride(test_frame, window, diagnostic_dir)
        export_waveform_cases(datasets['test'], test_frame, diagnostic_dir)
        activation_audit(model, datasets, folder)
        variant_frames[variant] = test_frame
        del model, embeddings
        gc.collect()
        torch.cuda.empty_cache()
    reference, candidate = variant_frames['original_acc'], variant_frames['centered_acc']
    assert np.array_equal(reference.window_end, candidate.window_end)
    paired = candidate[['subject','gesture','native_repetition','window_start','window_end','evaluate_primary','evaluate_common']].copy()
    paired['centering_recovered'] = ~reference.cnn_correct & candidate.cnn_correct
    paired['centering_harmed'] = reference.cnn_correct & ~candidate.cnn_correct
    paired.to_csv(output/'centering_paired_recovery.csv', index=False)


def summarize_diagnostic_job(root):
    metrics, gestures = [], []
    for manifest_path in root.glob('w*/s*/S*/*/fit_manifest.json'):
        folder = manifest_path.parent
        fit = json.loads(manifest_path.read_text())
        for split in ['train','validation','test']:
            for name, destination in [('metrics.csv', metrics), ('gesture_errors.csv', gestures)]:
                if not (folder/split/name).exists():
                    continue  # Preserve useful partial outputs if a later diagnostic fails.
                data = pd.read_csv(folder/split/name)
                data['subject'] = fit['subject']
                data['window_ms'] = fit['window_ms']
                data['training_stride_ms'] = fit['training_stride_ms']
                data['variant'] = folder.name
                data['split'] = split
                destination.append(data)
    if not metrics:
        return
    metrics = pd.concat(metrics, ignore_index=True)
    metrics.to_csv(root/'all_subject_metrics.csv', index=False)
    metrics.groupby(['window_ms','training_stride_ms','variant','split','grid','arm'])[['accuracy','f1_macro']].mean().to_csv(root/'mean_subject_metrics.csv')
    if not gestures:
        return
    errors = pd.concat(gestures, ignore_index=True)
    errors.to_csv(root/'all_gesture_errors.csv', index=False)
    for (window,stride,variant), part in errors.query("split == 'test' and grid == 'primary'").groupby(['window_ms','training_stride_ms','variant']):
        matrix = part.pivot(index='subject',columns='gesture',values='cnn_error_percent')
        fig, ax = plt.subplots(figsize=(12, max(3, len(matrix)*.28)))
        sns.heatmap(matrix, ax=ax, cmap='Reds', vmin=0, vmax=100, annot=True, fmt='.0f', cbar_kws={'label':'CNN test error %'})
        ax.set_title(f'{window} ms window / {stride} ms training stride / {variant}')
        save_figure(fig, root/f'errors_w{window}_s{stride}_{variant}.png')


def run_diagnostic_experiments():
    assert Config.DEVICE.type == 'cuda', 'Enable a Kaggle GPU.'
    assert Config.EXERCISE_IDS == (1,) and Config.GESTURE_MIN == 1 and Config.GESTURE_MAX == 17
    assert not Config.REFIT_ON_TRAIN_PLUS_VAL and not Config.AUGMENT_TRAIN
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    root = Config.KAGGLE_WORKING/f'db7_diagnostic_study_{stamp}'
    root.mkdir(parents=True)
    subjects = [1] if Config.SMOKE else list(range(1,23))
    if Config.SMOKE:
        Config.MIN_EPOCHS, Config.MAX_EPOCHS, Config.PATIENCE = 1, 2, 2
    manifest = dict(automation=Config.AUTOMATION, subjects=subjects, smoke=Config.SMOKE,
        windows_ms=Config.WINDOWS_MS, strides_ms=Config.STRIDES_MS, labels=list(range(1,18)),
        protocol='Within-subject 4 train / 1 validation / 1 test repetition; no refit',
        window_selection=False, checkpoint_selection='Minimum validation loss within each fit',
        native_grid='First endpoint = 100 ms trim + window length; same stride for train and validation. Test primary matches training stride.',
        dense_test_grid_ms=50, common_test_grid='Endpoints 700 ms + multiples of 200 ms in every configuration',
        gate='CNN confidence <0.70 AND EMG confidence >0.80; frozen; EMG StandardScaler + shrinkage LDA on 72 features',
        feature_diagnostics='300 physical features; no significance tests treating overlapping windows as independent',
        case_selection='Highest-confidence correct and wrong example per gesture; examples not representative random samples',
        limits='One seed. Existing test data explored before; future changes require independent confirmation. Rest excluded. Zero-phase per-repetition filtering is not online causal. Internal S18 identity and physical channel semantics unresolved. Flat/repeated extrema are indicators, not proven ADC clipping.',
        versions=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__))
    json_write(root/'run_manifest.json', manifest)
    completed = []
    try:
        for window in Config.WINDOWS_MS:
            for stride in Config.STRIDES_MS:
                assert window in [200,400,600] and stride in [50,100,200]
                for subject in subjects:
                    folder = root/f'w{window}'/f's{stride}'/f'S{subject:02}'
                    folder.mkdir(parents=True)
                    print(f'RUN: window={window} stride={stride} subject={subject}', flush=True)
                    recording = load_diagnostic_subject(subject, folder)
                    run_subject_configuration(recording, subject, window, stride, folder)
                    completed.append(dict(subject=subject,window_ms=window,stride_ms=stride))
                    json_write(root/'progress.json', completed)
                    del recording
                    gc.collect()
                summarize_diagnostic_job(root)
        expected = len(subjects)*len(Config.WINDOWS_MS)*len(Config.STRIDES_MS)*2
        assert len(list(root.glob('w*/s*/S*/*/fit_manifest.json'))) == expected
        json_write(root/'completion.json',dict(success=True,cnn_fits=expected,completed=completed,smoke=Config.SMOKE))
    except Exception:
        (root/'FAILURE.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise
    finally:
        try:
            summarize_diagnostic_job(root)
        except Exception:
            (root/'SUMMARY_FAILURE.txt').write_text(traceback.format_exc(), encoding='utf-8')
        finally:
            print('OUTPUT ZIP:',shutil.make_archive(str(root),'zip',root_dir=root),flush=True)
    return root

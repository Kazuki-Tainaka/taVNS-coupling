function paired_beats = pair_rri_sbp_beats_updated(rri_path, sbp_path)
% PAIR_RRI_SBP_BEATS_UPDATED RRIとsBPのビートイベントを、HR時間がRRIの中間点であることを考慮してペアリングする
%
% 出力:
%   paired_beats: (t_R_start_ms, RRI_ms, t_SBP_ms, SBP_mmHg, PTT_ms)
%
% ---------------------------------------------------------------------------
% Change vs. pair_rri_sbp_beats.m (2026-08-05)
%
%   The original advanced the search cursor with
%       if best_sbp_idx > sbp_start_idx, sbp_start_idx = best_sbp_idx; end
%   which leaves the cursor ON the matched SBP beat, so the same SBP beat can
%   be assigned again to the next R wave. Duplicate t_SBP_ms values break the
%   spline resampling in calc_spectral_brs_paired.m / data_preprocessing.m
%   ("first input must contain unique values").
%
%   Fixed to
%       sbp_start_idx = best_sbp_idx + 1;
%   which enforces strictly one-to-one R-wave <-> SBP-beat pairing.
%
% Verification (all 18 subjects, raw_HR_XX.csv + raw_sBP_XX.csv in data/v.2):
%   - the original function reproduces the archived paired_beats_XX.csv exactly;
%   - this updated function returns output IDENTICAL to the original for every
%     subject (max absolute difference 0.0 across all five columns, same row
%     counts), so no published result changes.
%   - the fix only takes effect on beat series dense enough for two R waves to
%     compete for one SBP beat, e.g. subject 13 re-detected with
%     MinPeakDistance = 0.3 s: 1417 beats with 2 duplicate SBP assignments
%     becomes 1415 beats with 0.
% ---------------------------------------------------------------------------

    % --- 設定 ---
    % PTT (Pulse Transit Time) の許容範囲 [ms]
    PTT_MIN_MS = 0;   % 生理学的に考えられる最小PTT [ms]
    PTT_MAX_MS = 500;  % 生理学的に考えられる最大PTT [ms]

    % --- 1. データロードとRRI計算 ---
    rri_data = readtable(rri_path, 'ReadVariableNames', false, 'Format', '%f%f');
    sbp_data = readtable(sbp_path, 'ReadVariableNames', false, 'Format', '%f%f');

    % 変数名の設定 (仮定: 1列目=時間[s], 2列目=値)
    rri_data.Properties.VariableNames = {'t_HR_mid_s', 'HR_bpm'};
    sbp_data.Properties.VariableNames = {'t_SBP_peak_s', 'SBP_mmHg'};

    % 時間を秒[s]からミリ秒[ms]に変換
    rri_data.t_HR_mid_ms = rri_data.t_HR_mid_s * 1000;
    sbp_data.t_SBP_ms = sbp_data.t_SBP_peak_s * 1000;

    % HR[bpm]をRRI[ms]に変換
    rri_data.RRI_ms = 60000.0 ./ rri_data.HR_bpm;

    % 欠損値/無限大値の除去 (RRIが0以下やNaNになる場合を除去)
    rri_data(isinf(rri_data.RRI_ms) | isnan(rri_data.RRI_ms) | rri_data.RRI_ms <= 0, :) = [];
    sbp_data(isnan(sbp_data.SBP_mmHg), :) = [];

    % --- 2. R-ピーク時間 (サイクル開始時間) の推定 ---
    % t_R_start = t_HR_mid - RRI / 2
    rri_data.t_R_start_ms = rri_data.t_HR_mid_ms - rri_data.RRI_ms / 2;

    % --- 3. ペアリング処理 ---

    % 結果格納用テーブルの初期化
    paired_beats = table('Size', [0, 5], 'VariableTypes', ...
                         {'double', 'double', 'double', 'double', 'double'}, ...
                         'VariableNames', {'t_R_start_ms', 'RRI_ms', 't_SBP_ms', 'SBP_mmHg', 'PTT_ms'});

    % SBPデータから検索を始めるインデックス (検索効率化)
    sbp_start_idx = 1;

    % RRIをアンカーとしてループ
    for i = 1:height(rri_data)
        t_R_start = rri_data.t_R_start_ms(i);
        RRI_val = rri_data.RRI_ms(i);

        % SBPの検索範囲 (R-peak時間 t_R_start を基準にPTTの生理学的範囲で設定)
        search_start_time = t_R_start + PTT_MIN_MS;
        search_end_time = t_R_start + PTT_MAX_MS;

        % 検索効率化: sbp_start_idx を search_start_time を超える最初のインデックスまで進める
        while sbp_start_idx <= height(sbp_data) && sbp_data.t_SBP_ms(sbp_start_idx) < search_start_time
            sbp_start_idx = sbp_start_idx + 1;
        end

        % SBPデータのうち、現在のRRIビートに対応する可能性がある範囲を抽出
        sbp_indices = sbp_start_idx:height(sbp_data);

        % 検索ウィンドウ内のSBPビートを特定
        found_mask = (sbp_data.t_SBP_ms(sbp_indices) >= search_start_time) & ...
                     (sbp_data.t_SBP_ms(sbp_indices) <= search_end_time);

        found_indices_relative = find(found_mask);

        if isempty(found_indices_relative)
            continue; % ペアが見つからなかった場合、RRI拍をスキップ
        else
            % 該当するSBPビートが見つかった場合

            % 絶対インデックスに変換
            best_sbp_idx_abs = sbp_indices(found_indices_relative);

            % PTTを計算: SBPピーク時間 - R-peak時間
            PTTs = sbp_data.t_SBP_ms(best_sbp_idx_abs) - t_R_start;

            % 最小PTTを持つSBPを選択（最もR-peakに近いSBPを選択）
            [~, min_idx_in_found] = min(PTTs);
            best_sbp_idx = best_sbp_idx_abs(min_idx_in_found);

            t_SBP = sbp_data.t_SBP_ms(best_sbp_idx);
            SBP_val = sbp_data.SBP_mmHg(best_sbp_idx);
            PTT_val = PTTs(min_idx_in_found);

            % 結果テーブルに行を追加
            new_row = table(t_R_start, RRI_val, t_SBP, SBP_val, PTT_val, ...
                            'VariableNames', paired_beats.Properties.VariableNames);
            paired_beats = [paired_beats; new_row];

            % 次のRRIの検索を効率化するため、検索開始インデックスを更新
            % 【修正】マッチしたSBP拍の「次」へ進めることで、同一SBP拍が
            % 連続する2つのR波に割り当てられるのを防ぐ (1対1ペアリングの保証)
            sbp_start_idx = best_sbp_idx + 1;
        end
    end
end

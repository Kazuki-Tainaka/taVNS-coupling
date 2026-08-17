% データの読み込み
paired_data = readtable('paired_beats_04.csv');
t_R_start = paired_data.t_R_start_ms;
t_SBP = paired_data.t_SBP_ms;

% --------------------------------------------------------
% 1. 単一拍内での順序確認 (t_R_start < t_SBP)
% これはPTT > 0ms の設定により保証されているはずですが、確認します。
% --------------------------------------------------------
intra_beat_order_check = all(t_R_start < t_SBP);

% --------------------------------------------------------
% 2. 連続拍間での順序確認 (t_SBP(i) < t_R_start(i+1))
% SBPのピークが次のRRIサイクルの開始より前に発生しているか
% --------------------------------------------------------
% SBPのピーク時間（最後の1つを除く）
t_SBP_prev = t_SBP(1:end-1); 
% 次のRRIの開始時間（最初の1つを除く）
t_R_start_next = t_R_start(2:end);

% 連続するペアで t_SBP(i) < t_R_start(i+1) が全て真か確認
inter_beat_order_check = all(t_SBP_prev < t_R_start_next);

% --------------------------------------------------------
% 結果の表示
% --------------------------------------------------------
if intra_beat_order_check
    fprintf('1. 単一拍内順序 (t_R_start < t_SBP): OK (全てのPTT > 0)\n');
else
    fprintf('1. ERROR: 単一拍内順序が崩れている箇所があります。\n');
end

if inter_beat_order_check
    fprintf('2. 連続拍間順序 (t_SBP(i) < t_R_start(i+1)): OK (次のRRIサイクルと混同なし)\n');
else
    fprintf('2. ERROR: SBPピークが次のRRIサイクルに跨っている可能性があります。\n');
end
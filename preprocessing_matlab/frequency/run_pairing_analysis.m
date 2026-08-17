function run_pairing_analysis()
% RUN_PAIRING_ANALYSIS.M - 全被験者のRRIとsBPのビートペアリングを実行するメインスクリプト

    % --- 1. 設定 ---
    DATA_DIR = '../../data/v.2';  
    OUT_DIR = '../../data/v.2/paired_beats'; 
    SUBJECT_IDS = 1:18; % ここに解析対象の被験者番号をリストアップしてください（例: 1:10, [1, 3, 5, 8]）
    
    % 出力ディレクトリの作成
    if ~exist(OUT_DIR, 'dir')
        mkdir(OUT_DIR);
        fprintf('Created output directory: %s\n', OUT_DIR);
    end
    
    % --- 2. メインループ ---
    for subject_id = SUBJECT_IDS
        % 被験者番号を2桁の文字列にフォーマット (例: 1 -> '01')
        subject_str = sprintf('%02d', subject_id);
        
        rri_filename = ['raw_HR_', subject_str, '.csv'];
        sbp_filename = ['raw_sBP_', subject_str, '.csv'];
        
        rri_path = fullfile(DATA_DIR, rri_filename);
        sbp_path = fullfile(DATA_DIR, sbp_filename);
        
        % ファイルの存在チェック
        if ~exist(rri_path, 'file') || ~exist(sbp_path, 'file')
            warning('File(s) not found for subject %s. Skipping. Check subject ID and filenames.', subject_str);
            continue;
        end
        
        fprintf('Processing Subject %s... ', subject_str);
        
        try
            % ペアリング関数を呼び出し
            paired_data = pair_rri_sbp_beats(rri_path, sbp_path);
            
            % 結果を保存
            output_filename = ['paired_beats_', subject_str, '.csv'];
            output_path = fullfile(OUT_DIR, output_filename);
            
            writetable(paired_data, output_path);
            fprintf('Success! Paired %d beats. Saved to %s\n', height(paired_data), output_path);
            
        catch ME
            fprintf('\n  -> ERROR processing subject %s: %s\n', subject_str, ME.message);
        end
    end
    
    fprintf('\nAll subjects processing complete. Paired data is in the "%s" folder.\n', OUT_DIR);
end

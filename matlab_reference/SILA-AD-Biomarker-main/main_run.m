%% 초기화
clear; clc; close all;

filename = 'C:\Users\NXTDA0\Desktop\Git\code_conversion\data\sila_input_amyloid.csv';
SILAInput = readtable(filename);

% 필터링 FBB: 1.08, FBP: 1.11 non-composite
% 필터링 FBB: 0.74, FBP: 0.78 composite
tracer_filter = strcmp(SILAInput.("AMY_TRACER"), 'FBP');
SILAInput = SILAInput(tracer_filter, :);

age = table2array(SILAInput(:, 'AGE'));
amy_global = table2array(SILAInput(:, 'AMY_GLOBAL_COMPOSITE'));
id = table2array(SILAInput(:, 'RID'));


[tsila, tdrs] = SILA(age, amy_global, id, 0.25, 0.78, 200);

est = SILA_estimate(tsila, ...
                     age, ...
                     amy_global, ...
                     id, ...
                     'align_event','all', ...
                     'truncate_aget0','yes', ...
                     'extrap_years',3);

% for save
SILAInput = sortrows(SILAInput, {'RID', 'AGE'});
SILAInput.Amyloid_age = est.estdtt0;
SILAInput.truncated = est.truncated;

% writetable(SILAInput, 'C:\Users\NXTDA0\Desktop\Git\code_conversion\data\ADNI\ADNI.uc.SILA.output.csv')


%% Visualization
results = struct();

subplot(2,1,1); hold on;
plot(tsila.adtime, tsila.val, 'Color', 'b')
plot(tsila.adtime, tsila.val - tsila.ci95, 'Color', 'r', 'LineStyle', '--')
plot(tsila.adtime, tsila.val + tsila.ci95, 'Color', 'r', 'LineStyle', '--')
hold off;
title('Amyloid vs Years A+ (thr=0.78)')
xlabel('Years A+')
ylabel('Global SUVR')

results.x = tsila.adtime;
results.y = tsila.val;
results.y_ci_95_p = tsila.val + tsila.ci95;
results.y_ci_95_n = tsila.val - tsila.ci95;
% save('./results/SILA_Nonparametric_Amyloid_Curve.mat','results');

%
results = struct();

subplot(2,1,2); hold on;
plot(tdrs.val, tdrs.rate, 'Color', 'b')
plot(tdrs.val, tdrs.rate - tdrs.ci, 'Color', 'r', 'LineStyle', '--')
plot(tdrs.val, tdrs.rate + tdrs.ci, 'Color', 'r', 'LineStyle', '--')
title('Rate vs. Value Curve')
xlabel('Global SUVR')
ylabel('Rate (Global SUVR / Year)')

results.x = tdrs.val;
results.y = tdrs.rate;
results.y_ci_95_p = tdrs.rate + tdrs.ci;
results.y_ci_95_n = tdrs.rate - tdrs.ci;
% save('./results/SILA_Nonparametric_AmyloidAccumulation_Curve.mat','results');

%% estimate (inference)
% 1) 원본 테이블 불러오기
origData = readtable('C:\Users\NXTDA0\Desktop\Git\code_conversion\data\filtered_data.csv');


%%
% (RID, AGE) 기준 정렬
origData = sortrows(origData, {'RID','AGE'});


%%
% 3) SILA_estimate 실행
test = SILA_estimate(tsila, ...
    origData.AGE, ...
    origData.AMY_GLOBAL_COMPOSITE, ...
    origData.RID, ...
    'align_event','all', ...
    'truncate_aget0','yes', ...
    'extrap_years',3);

% subData에 Amyloid_age, truncated 컬럼 추가
origData.Amyloid_age = test.estdtt0;
origData.truncated = test.truncated;

% 4) 요약 테이블 생성 (필요한 컬럼만 선택)
summaryData = origData(:, {'PTID', 'TIMEPOINTS', 'FULL_ID', 'AGE', 'AMY_GLOBAL_COMPOSITE', ...
                           'Amyloid_age', 'truncated'});

% 6) 최종적으로 CSV 저장
% 전체 데이터 저장
% writetable(origData, 'C:\Users\NXTDA0\Desktop\Git\code_conversion\data\filtered_data_with_sila.csv');

% 요약 테이블 저장 (PTID, RID, AGE, AMY_GLOBAL_COMPOSITE, Amyloid_age, truncated)
writetable(summaryData, 'C:\Users\NXTDA0\Desktop\Git\code_conversion\data\amy_age.csv');

% 7) 결과 출력
fprintf('처리 완료!\n');
fprintf('저장된 파일:\n');
fprintf('  - filtered_data_with_sila.csv (전체 데이터)\n');
fprintf('  - sila_summary.csv (요약 테이블)\n');


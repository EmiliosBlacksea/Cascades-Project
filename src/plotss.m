%% Ablation plots + mean-importance ranking + plots excluding drop-all
% Expects columns: k, baseline_acc, baseline_auc, and drop variants like drop_x_acc, drop_x_auc

clear; clc; close all;

%% ---- Inputs ----
xlsxFile  = "ablation_summary_mlp.xlsx";   % <-- change if needed
sheetName = "metrics";                     % <-- change if needed

outDir = "ablation_plots_test";
if ~exist(outDir, "dir"), mkdir(outDir); end

% --- Drop variants (prefix before _acc/_auc) ---
dropNames = [
    "drop_burstiness_cv"
    "drop_duplicate_leaf_users_k"
    "drop_interarrival_trend"
    "drop_max_depth_k"
    "drop_max_outdeg_k"
    "drop_mean_inter_second_half"
    "drop_mean_inter_first_half"
    "drop_mean_interarrival"
    "drop_num_leaves_k"
    "drop_p90_depth_k"
    "drop_root_outdeg_k"
    "drop_std_interarrival"
    "drop_time_k"
    "drop_time_k_plus_many"
    "drop_unique_leaf_users_k"
    "drop_all"              % <-- INCLUDE your drop-all name here if you have it
];

% If your drop-all row has a different name, set it here exactly (prefix before _acc/_auc).
% If left "", we will auto-remove the single largest outlier in the ranking plots.
dropAllName = "drop_all";

% Force column vector
dropNames = string(dropNames(:));

%% ---- Load & prepare ----
T = readtable(xlsxFile, "Sheet", sheetName);
T = sortrows(T, "k");

k       = T.k;
baseAcc = T.baseline_acc;
baseAuc = T.baseline_auc;

varNames = string(T.Properties.VariableNames);

%% ---- Plot baseline once ----
baseAccPct = baseAcc * 100;
baseAucPct = baseAuc * 100;

validAcc = ~isnan(k) & ~isnan(baseAccPct);
validAuc = ~isnan(k) & ~isnan(baseAucPct);

fig = figure("Name", "baseline", "Color", "w");
tiledlayout(2,1, "Padding","compact", "TileSpacing","compact");

nexttile;
plot(k(validAcc), baseAccPct(validAcc), "-", "LineWidth", 1);
grid on; xlabel("k"); ylabel("Accuracy (%)");
title("Baseline Accuracy");

nexttile;
plot(k(validAuc), baseAucPct(validAuc), "-", "LineWidth", 1);
grid on; xlabel("k"); ylabel("AUC (%)");
title("Baseline AUC");

saveas(fig, fullfile(outDir, "base.png"));
close(fig);

%% ---- Per-drop degradation plots + collect mean degradations ----
meanWorseAcc = nan(numel(dropNames), 1);
meanWorseAuc = nan(numel(dropNames), 1);

for i = 1:numel(dropNames)
    name = dropNames(i);
    accCol = name + "_acc";
    aucCol = name + "_auc";

    if ~ismember(accCol, varNames) || ~ismember(aucCol, varNames)
        warning("Missing columns for %s (expected %s and %s). Skipping.", name, accCol, aucCol);
        continue;
    end

    dropAcc = T.(accCol);
    dropAuc = T.(aucCol);

    % "How much worse" in percentage points (pp): baseline - drop
    worseAcc = (baseAcc - dropAcc) * 100;
    worseAuc = (baseAuc - dropAuc) * 100;

    meanWorseAcc(i) = mean(worseAcc, "omitnan");
    meanWorseAuc(i) = mean(worseAuc, "omitnan");

    validAcc = ~isnan(k) & ~isnan(worseAcc);
    validAuc = ~isnan(k) & ~isnan(worseAuc);

    fig = figure("Name", name, "Color", "w");
    tiledlayout(2,1, "Padding","compact", "TileSpacing","compact");

    nexttile;
    plot(k(validAcc), worseAcc(validAcc), "-", "LineWidth", 1);
    yline(0, "--");
    grid on;
    xlabel("k");
    ylabel("baseline - drop (pp)");
    title("Accuracy degradation dropping: " + strrep(name, "_", "\_"));

    nexttile;
    plot(k(validAuc), worseAuc(validAuc), "-", "LineWidth", 1);
    yline(0, "--");
    grid on;
    xlabel("k");
    ylabel("baseline - drop (pp)");
    title("AUC degradation dropping: " + strrep(name, "_", "\_"));

    saveas(fig, fullfile(outDir, name + ".png"));
    close(fig);
end

%% ---- Build ranking table (remove rows with both NaN means) ----
ranking = table(dropNames, meanWorseAcc, meanWorseAuc, ...
    'VariableNames', {'feature','mean_worse_acc_pp','mean_worse_auc_pp'});

keep = ~(isnan(ranking.mean_worse_acc_pp) & isnan(ranking.mean_worse_auc_pp));
ranking = ranking(keep, :);

%% ---- Sort and save rankings ----
rankingAcc = sortrows(ranking, 'mean_worse_acc_pp', 'descend');
rankingAuc = sortrows(ranking, 'mean_worse_auc_pp', 'descend');

disp("=== Feature ranking by mean Accuracy degradation (pp) ===");
disp(rankingAcc);

writetable(rankingAcc, fullfile(outDir, "feature_ranking_by_mean_worseAcc.csv"));
writetable(rankingAuc, fullfile(outDir, "feature_ranking_by_mean_worseAuc.csv"));

%% ---- Plot ranked mean degradations (with drop-all) ----
% Accuracy ranked
fig = figure("Name", "mean_worseAcc_ranked", "Color", "w");
barh(rankingAcc.mean_worse_acc_pp);
grid on;
set(gca, "YDir", "reverse");
yticks(1:height(rankingAcc));
yticklabels(strrep(rankingAcc.feature, "_", "\_"));
xlabel("Mean (baseline - drop) Accuracy (pp)");
title("Feature importance (mean Accuracy degradation)");
saveas(fig, fullfile(outDir, "mean_worseAcc_ranked.png"));
close(fig);

% AUC ranked
fig = figure("Name", "mean_worseAuc_ranked", "Color", "w");
barh(rankingAuc.mean_worse_auc_pp);
grid on;
set(gca, "YDir", "reverse");
yticks(1:height(rankingAuc));
yticklabels(strrep(rankingAuc.feature, "_", "\_"));
xlabel("Mean (baseline - drop) AUC (pp)");
title("Feature importance (mean AUC degradation)");
saveas(fig, fullfile(outDir, "mean_worseAuc_ranked.png"));
close(fig);

% Grouped (Accuracy-sorted order)
fig = figure("Name", "mean_worse_grouped", "Color", "w");
vals = [rankingAcc.mean_worse_acc_pp, rankingAcc.mean_worse_auc_pp];
barh(vals);
grid on;
set(gca, "YDir", "reverse");
yticks(1:height(rankingAcc));
yticklabels(strrep(rankingAcc.feature, "_", "\_"));
xlabel("Mean degradation (pp)");
title("Mean degradation by feature (Accuracy & AUC)");
legend({'Accuracy','AUC'}, "Location", "best");
saveas(fig, fullfile(outDir, "mean_worse_grouped.png"));
close(fig);

%% ---- Extra ranked plots excluding drop-all (to avoid squashing everything) ----
% Helper to remove drop-all robustly
removeDropAll = @(tbl, metricCol) ...
    ( ...
      (dropAllName ~= "" && any(tbl.feature == dropAllName)) .* tbl(tbl.feature ~= dropAllName, :) + ...
      (dropAllName == "" || ~any(tbl.feature == dropAllName)) .* ( ...
          tbl( setdiff(1:height(tbl), find(tbl.(metricCol) == max(tbl.(metricCol), [], "omitnan"), 1, "first")), : ) ...
      ) ...
    );

% The anonymous function trick above is messy for older MATLABs; do it explicitly instead:

% --- Accuracy no-drop-all ---
rankingAccNoAll = rankingAcc;
if dropAllName ~= "" && any(rankingAccNoAll.feature == dropAllName)
    rankingAccNoAll = rankingAccNoAll(rankingAccNoAll.feature ~= dropAllName, :);
else
    [~, j] = max(rankingAccNoAll.mean_worse_acc_pp, [], "omitnan");
    rankingAccNoAll(j,:) = [];
end

fig = figure("Name", "mean_worseAcc_ranked_NO_DROPALL", "Color", "w");
barh(rankingAccNoAll.mean_worse_acc_pp);
grid on;
set(gca, "YDir", "reverse");
yticks(1:height(rankingAccNoAll));
yticklabels(strrep(rankingAccNoAll.feature, "_", "\_"));
xlabel("Mean (baseline - drop) Accuracy (pp)");
title("Feature importance (mean Accuracy degradation) — excluding drop-all");
saveas(fig, fullfile(outDir, "mean_worseAcc_ranked_no_dropall.png"));
close(fig);

% --- AUC no-drop-all ---
rankingAucNoAll = rankingAuc;
if dropAllName ~= "" && any(rankingAucNoAll.feature == dropAllName)
    rankingAucNoAll = rankingAucNoAll(rankingAucNoAll.feature ~= dropAllName, :);
else
    [~, j] = max(rankingAucNoAll.mean_worse_auc_pp, [], "omitnan");
    rankingAucNoAll(j,:) = [];
end

fig = figure("Name", "mean_worseAuc_ranked_NO_DROPALL", "Color", "w");
barh(rankingAucNoAll.mean_worse_auc_pp);
grid on;
set(gca, "YDir", "reverse");
yticks(1:height(rankingAucNoAll));
yticklabels(strrep(rankingAucNoAll.feature, "_", "\_"));
xlabel("Mean (baseline - drop) AUC (pp)");
title("Feature importance (mean AUC degradation) — excluding drop-all");
saveas(fig, fullfile(outDir, "mean_worseAuc_ranked_no_dropall.png"));
close(fig);

% --- Grouped no-drop-all (Accuracy order, no-drop-all) ---
fig = figure("Name", "mean_worse_grouped_NO_DROPALL", "Color", "w");
valsNoAll = [rankingAccNoAll.mean_worse_acc_pp, rankingAccNoAll.mean_worse_auc_pp];
barh(valsNoAll);
grid on;
set(gca, "YDir", "reverse");
yticks(1:height(rankingAccNoAll));
yticklabels(strrep(rankingAccNoAll.feature, "_", "\_"));
xlabel("Mean degradation (pp)");
title("Mean degradation by feature (Accuracy & AUC) — excluding drop-all");
legend({'Accuracy','AUC'}, "Location", "best");
saveas(fig, fullfile(outDir, "mean_worse_grouped_no_dropall.png"));
close(fig);

disp("Done. Saved per-drop plots + ranking plots to: " + outDir);

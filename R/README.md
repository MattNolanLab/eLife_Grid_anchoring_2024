The repository contains statistical analysis for figures in Clark et al.

* Figure 2F

Analysis is in CompareGroups.Rmd


* Figure 3G

Analysis is in Compare_by_trial.Rmd


* Figure 4F

Analysis is in GridvsNonGrid.Rmd


* Figure 3 Figure Supplement 4 *

Analysis is in CompareGroupsStableCats.Rmd


* Figure 4E

Analysis is in agreement_analysis.Rmd


* Figure 6B

Analysis is in behaviour_analysis_grid_cells_stable_sessions.Rmd


* Figure 6D

Analysis is in behaviour_analysis_grid_cells.Rmd


* Figure 6E

Analysis is in behaviour_analysis_hit_try_run.Rmd


* Figure 6, Figure Supplement 2F

Analysis is in behaviour_analysis_grid_cells_altmethod.Rmd


* Figure 6, Figure Supplement 2G

Analysis is in behaviour_analysis_hit_try_run_altmethod.Rmd


Additional notes:

- In the initial submission of the manuscipt, task-anchored and task-independent firing was formly named "position" and "distance" coding based on the type of information extractable from these grid codes. However, following comments by reviewers, we decided to change the name to what is now called task-anchored and task-independent firing. Where "position" and "distance" labels appear throughout the code can be considered task-anchored and task-independent respectively.

- You will need to first convert the .pkl python dataframes into .Rda format using ConvertPickletoRda.Rmd. Once done, all of the remaining Rmd files can be run

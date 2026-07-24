# bash scripts/run_qassr_experiments.sh sped suite 2>&1 | tee outputs/run_logs/sped_suit.log
# bash scripts/run_qassr_experiments.sh pitts30k-test suite 2>&1 | tee outputs/run_logs/pitts30k-test_suit.log
# bash scripts/run_qassr_experiments.sh amstertime suite 2>&1 | tee outputs/run_logs/amstertime_suit.log
# bash scripts/run_qassr_experiments.sh nordland suite 2>&1 | tee outputs/run_logs/nordland_suit.log
# bash scripts/run_qassr_experiments.sh svox-all suite 2>&1 | tee outputs/run_logs/svox-all_suit.log
# bash scripts/run_qassr_experiments.sh tokyo247 suite 2>&1 | tee outputs/run_logs/tokyo247_suit.log

bash scripts/run_qassr_experiments.sh nordland baseline 2>&1 | tee outputs/run_logs/nordland_baseline.log
bash scripts/run_qassr_experiments.sh nordland sparse 2>&1 | tee outputs/run_logs/nordland_sparse.log
bash scripts/run_qassr_experiments.sh nordland role 2>&1 | tee outputs/run_logs/nordland_role.log
bash scripts/run_qassr_experiments.sh nordland gate 2>&1 | tee outputs/run_logs/nordland_gate.log

bash scripts/run_qassr_experiments.sh svox-all baseline 2>&1 | tee outputs/run_logs/svox-all_baseline.log
bash scripts/run_qassr_experiments.sh svox-all sparse 2>&1 | tee outputs/run_logs/svox-all_sparse.log
bash scripts/run_qassr_experiments.sh svox-all role 2>&1 | tee outputs/run_logs/svox-all_role.log
bash scripts/run_qassr_experiments.sh svox-all gate 2>&1 | tee outputs/run_logs/svox-all_gate.log

bash scripts/run_qassr_experiments.sh tokyo247 baseline 2>&1 | tee outputs/run_logs/tokyo247_baseline.log
bash scripts/run_qassr_experiments.sh tokyo247 sparse 2>&1 | tee outputs/run_logs/tokyo247_sparse.log
bash scripts/run_qassr_experiments.sh tokyo247 role 2>&1 | tee outputs/run_logs/tokyo247_role.log
bash scripts/run_qassr_experiments.sh tokyo247 gate 2>&1 | tee outputs/run_logs/tokyo247_gate.log
from experiments.admg_adjustment import run_experiment as run_admg_experiment
from experiments.dag_adjustment import run_experiment as run_dag_experiment
from experiments.relation_variance_quotient import run_experiment as run_relation_experiment


if __name__ == "__main__":
    run_admg_experiment()
    run_dag_experiment()
    #run_relation_experiment()

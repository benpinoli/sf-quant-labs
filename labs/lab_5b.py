"""
Example code for submitting an MVO backtest as a job to the supercomputer.
Note that to actually submit the job you need to change dry_run to false.
Also note that you need to fill in your own project_root and byu_email.
"""

from sf_backtester import BacktestConfig, BacktestRunner, SlurmConfig

slurm_config = SlurmConfig(
    n_cpus=8,
    mem="32G",
    time="03:00:00",
    mail_type="BEGIN,END,FAIL",
    max_concurrent_jobs=30,
)

backtest_config = BacktestConfig(
    signal_name="momentum",
    data_path="momentum_alphas.parquet",
    gamma=50,
    project_root="/home/byunetid/path/to/your/project",
    byu_email="netid.byu.edu",
    constraints=["ZeroBeta", "ZeroInvestment"],
    slurm=slurm_config,
)

backtest_runner = BacktestRunner(backtest_config)
backtest_runner.submit(dry_run=True)

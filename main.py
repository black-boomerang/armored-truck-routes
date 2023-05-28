from src.task import Environment
from src.tools import evaluate
from src.solvers import ClusteringSolver
from src.utils import get_data, parse_args


def main(config=None):

    if config is None:
        config = parse_args()

    data_config = config['data']
    terminals, time_matrix = get_data(data_path=data_config['path'])

    task_config = config['environment']
    environment = Environment(
        non_cashable_loss_percent_per_year=task_config['non_cashable_loss_percent_per_year'],
        cashable_loss_percent=task_config['cashable_loss_percent'],
        cashable_min_loss=task_config['cashable_min_loss'],
        non_serviced_days=task_config['non_serviced_days'],
        terminal_limit=task_config['terminal_limit'],
        armored_price=task_config['armored_price'],
        working_day_time=task_config['working_day_time'],
        encashment_time=task_config['encashment_time']
    )

    solver = ClusteringSolver(terminals['start_value'].values, time_matrix, environment)
    evaluate(solver, terminals, 1, 91)


if __name__ == "__main__":
    main()

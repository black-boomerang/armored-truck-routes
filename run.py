from solvers import DensitySolver
from task import BusinessLogic
from utils import get_data, evaluate

if __name__ == "__main__":
    terminals, time_matrix = get_data()
    business_logic = BusinessLogic()
    solver = DensitySolver(terminals['start_value'].values, time_matrix, business_logic)
    evaluate(solver, terminals)

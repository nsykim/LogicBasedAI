import pulp
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class LPModel:
    """
    Logic programming model
    """

    def __init__ (self, name="Rule Based Model", sense=pulp.LpMaximize):
        """
        Constructor for LPModel class
        Initializes the model attribute.

        Input:
            - name (str) - name of the model
            - sense (obj) - optimization direction (pulp.LpMaximize or pulp.LpMinimize)
        """
        self.model = pulp.LpProblem(name, sense)
        self.variables = {}
        self.objective_value = None
        logging.info(f"Initialized LPModel with name {name} and sense {sense}")
    
    def set_logging(self, level=logging.INFO):
        """
        Sets the logging level for the model.

        Input:
            - level (str) - logging level (logging.INFO, logging.DEBUG, logging.WARNING, logging.ERROR, logging.CRITICAL)
        """
        num_level = getattr(logging, level.upper(), None)
        if not isinstance(num_level, int):
            logging.error("Invalid logging level")
            return False
        
        logging.getLogger().setLevel(num_level)
        logging.info(f"Set logging level to {level}")
    
    def define_vars(self, var_dict):
        """
        Defines the decision variables for the model.
        The variables that the model will solve for.

        Input:
            - var_dict (dict) - dictionary with type {var_name: (low, up, cat)}
        """
        if not isinstance(var_dict, dict):
            logging.error("var_dict must be a dictionary")
            return False

        for var, bounds in var_dict.items():
            if not isinstance(bounds, tuple) or len(bounds) != 3:
                logging.error(f"Invalid bounds for variable {var}")
                return False
            low, up, cat = bounds
            self.variables[var] = pulp.LpVariable(var, lowBound=low, upBound=up, cat=cat)

        logging.info(f"Defined variables: {var_dict}")

    def set_objective(self, obj_coeffs):
        """
        Sets the objective function for the model.
        The function that the model will minimize or maximize.

        Input:
            - obj_coeffs (dict) - dictionary with type {var_name: coeff} where coeff is the coefficient of the variable in the objective function
        """
        if not isinstance(obj_coeffs, dict):
            logging.error("obj_coeffs must be a dictionary")
            return False

        for var_name in obj_coeffs.keys():
            if var_name not in self.variables:
                logging.error(f"Variable {var_name} not defined")
                return False
        self.model+= pulp.lpSum([self.variables[var]*coeff for var, coeff in obj_coeffs.items()])
        logging.info(f"Set objective function: {obj_coeffs}")

    def add_constraints(self, constraints):
        """
        Adds constraints to the model.
        Constraints that the decision variables must satisfy (rules).

        Input:
            - constraints (list) - list of tuples to be added to the model, where each tuple has form:
                - constaint_expr (pulp.LpConstraint) - expression of the constraint
                - sense: (obj) - sense of the constraint (pulp.LpConstraintLE, pulp.LpConstraintGE, pulp.LpConstraintEQ)
                - rhs: (float) - right hand side of the constraint
        """
        if not isinstance(constraints, list):
            logging.error("constraints must be a list")
            return False
        
        for constraint in constraints:
            if not isinstance(constraint, tuple) or len(constraint) != 3:
                logging.error(f"Invalid constraint: {constraint}")
            constraint_expr, sense, rhs = constraint
            if not isinstance(constraint_expr, dict):
                logging.error(f"Invalid constraint expression: {constraint_expr}")
                return False
            for var_name in constraint_expr.keys():
                if var_name not in self.variables:
                    logging.error(f"Variable {var_name} not defined")
                    return False
            if sense not in [pulp.LpConstraintLE, pulp.LpConstraintGE, pulp.LpConstraintEQ]:
                logging.error(f"Invalid sense: {sense}")
                return False
            if not isintance(rhs, (int, float)):
                logging.error(f"Invalid rhs: {rhs}")
                return False

            self.model+= (pulp.lpSum([coeff * self.variables[var] for var, coeff in constraint_expr.items()]), sense, rhs)
            logging.info("Constraints added successfully")

    def solve(self):
        """
        Solves the model.
        """
        try:
            self.model.solve()
            self.objective_value = pulp.value(self.model.objective)
            logging.info("Model solved successfully")
        except Exception as e:
            logging.error(f"Error solving model: {e}")
    
    def get_results(self):
        """
        Returns the results of the model.

        Output:
            - dict with type {var_name: value} - dictionary with the values of the decision variables
        """
        try: 
            results = {v.name: v.varValue for v in self.model.variables()}
            logging.info("Results retrieved successfully")
            return results
        except Exception as e:
            logging.error(f"Error retrieving results: {e}")
            return None

    def run(self, var_dict, obj_coeffs, constraints):
        """
        Runs the model.

        Input:
            - var_dict (dict) - dictionary with type {var_name: (low, up, cat)}
            - obj_coeffs (dict) - dictionary with type {var_name: coeff}
            - constraints (list) - list of tuples to be added to the model, where each tuple has form:
                - constaint_expr (pulp.LpConstraint) - expression of the constraint
                - sense: (obj) - sense of the constraint (pulp.LpConstraintLE, pulp.LpConstraintGE, pulp.LpConstraintEQ)
                - rhs: (float) - right hand side of the constraint
            
        Output:
            - dict with type {var_name: value} - dictionary with the values of the decision variables
        """
        if not self.define_vars(var_dict):
            logging.error("Error defining variables")
            return None
        if not self.set_objective(obj_coeffs):
            logging.error("Error setting objective function")
            return None
        if not self.add_constraints(constraints):
            logging.error("Error adding constraints")
            return None
        self.solve()
        return self.get_results()
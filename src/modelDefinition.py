import pulp
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class RuleBasedModel:
    """
    RuleBasedModel is a class for creating and solving linear programming models using the PuLP library. 
    It allows users to define decision variables, set objective functions, add constraints, and solve the model.
    Methods:
        __init__(self, name="Rule Based Model", sense=pulp.LpMaximize):
            Initializes the RuleBasedModel with a given name and optimization direction.
        set_logging(self, level=logging.INFO):
        define_vars(self, var_dict):
        set_objective(self, obj_coeffs):
        add_constraints(self, constraints):
        solve(self):
        get_results(self):
        run(self, var_dict, obj_coeffs, constraints):
            Runs the model by defining variables, setting the objective function, adding constraints, solving the model, and returning the results.
    """

    def __init__ (self, name="Rule_Based_Model", sense=pulp.LpMaximize):
        """
        Initializes a new instance of the LPModel class.

        Args:
            name (str): The name of the linear programming model. Defaults to "Rule Based Model".
            sense (pulp.LpSense): The sense of the optimization problem (e.g., pulp.LpMaximize or pulp.LpMinimize). Defaults to pulp.LpMaximize.

        Attributes:
            model (pulp.LpProblem): The linear programming problem instance.
            variables (dict): A dictionary to store the variables of the model.
            objective_value (float or None): The value of the objective function after solving the model. Initialized to None.
        """
        self.model = pulp.LpProblem(name, sense)
        self.variables = {}
        self.objective_value = None
        logging.info(f"Initialized LPModel with name {name} and sense {sense}")
    
    def set_logging(self, level=logging.INFO):
        """
        Sets the logging level for the application.
        Parameters:
        level (str): The logging level to set. Default is logging.INFO. 
                     Valid levels are 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'.
        Returns:
        bool: True if the logging level was set successfully, False otherwise.
        """
        num_level = getattr(logging, level.upper(), None)
        if not isinstance(num_level, int):
            logging.error("Invalid logging level")
            return False
        
        logging.getLogger().setLevel(num_level)
        logging.info(f"Set logging level to {level}")
        return True
    
    def define_vars(self, var_dict):
        """
        Define variables for the model using the provided dictionary.

        Parameters:
        var_dict (dict): A dictionary where keys are variable names and values are tuples 
                 containing the lower bound, upper bound, and category of the variable.

        Returns:
        bool: False if the input is invalid, otherwise None.

        The method logs an error and returns False if var_dict is not a dictionary or if any 
        of the bounds are not tuples of length 3. Otherwise, it defines the variables using 
        the provided bounds and logs the defined variables.
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
        Sets the objective function for the linear programming model.
        Parameters:
        obj_coeffs (dict): A dictionary where keys are variable names (str) and values are their corresponding coefficients (float or int).
        Returns:
        bool: Returns False if obj_coeffs is not a dictionary or if any variable name in obj_coeffs is not defined in the model. Otherwise, sets the objective function and logs the action.
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
        Adds constraints to the linear programming model.
        Parameters:
        constraints (list): A list of constraints to be added to the model. Each constraint should be represented as a tuple (constraint_expr, sense, rhs).
        Returns:
        bool: Returns False if constraints is not a list or if any constraint expression is invalid. Otherwise, adds the constraints to the model and logs the action."""

        if not isinstance(constraints, list):
            logging.error("constraints must be a list")
            return False

        try:
            for constraint in  constraints:
                if not isinstance(constraint, tuple) or len(constraint) != 3:
                    logging.error("Each contraint must be a tuple of length 3")
                    return False

                constraint_expr, sense, rhs = constraint

                if not isinstance(constraint_expr, dict):
                    logging.error("Constraint expression must be a dictionary")
                    return False

                for var in constraint_expr.keys():
                    if var not in self.variables:
                        logging.error(f"Variable {var} not defined")
                        return False

                if sense not in [pulp.LpConstraintLE, pulp.LpConstraintGE, pulp.LpConstraintEQ]:
                    logging.error("Invalid constraint sense")
                    return False

                if not isinstance(rhs, (int, float)):
                    logging.error("Invalid right-hand side value")
                    return False

                if sense == pulp.LpConstraintLE:
                    self.model += pulp.lpSum([self.variables[var]*coeff for var, coeff in constraint_expr.items()]) <= rhs
                elif sense == pulp.LpConstraintGE:
                    self.model += pulp.lpSum([self.variables[var]*coeff for var, coeff in constraint_expr.items()]) >= rhs
                elif sense == pulp.LpConstraintEQ:
                    self.model += pulp.lpSum([self.variables[var]*coeff for var, coeff in constraint_expr.items()]) == rhs
                
            logging.info("Constraints added successfully")
            return True
        except KeyError:
            logging.error("Invalid constraint expression")
            return False 


    def solve(self):
        """
        Solves the optimization model and updates the objective value.

        This method attempts to solve the optimization model defined in the instance.
        If the model is solved successfully, the objective value is updated and a success
        message is logged. If an error occurs during the solving process, an error message
        is logged with the exception details.

        Raises:
            Exception: If an error occurs during the solving process.
        """
        try:
            self.model.solve()
            self.objective_value = pulp.value(self.model.objective)
            logging.info("Model solved successfully")
        except Exception as e:
            logging.error(f"Error solving model: {e}")
    
    def get_results(self):
        """
        Retrieves the results of the model's variables.

        This method attempts to extract the variable names and their corresponding values
        from the model and returns them as a dictionary. If an error occurs during this 
        process, it logs the error and returns None.

        Returns:
            dict: A dictionary where the keys are variable names and the values are 
              the variable values, or None if an error occurs.
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
        Executes the optimization model with the given variables, objective coefficients, and constraints.

        Parameters:
        var_dict (dict): A dictionary where keys are variable names and values are their respective bounds or initial values.
        obj_coeffs (dict): A dictionary where keys are variable names and values are their respective coefficients in the objective function.
        constraints (list): A list of constraints to be added to the model. Each constraint should be represented in a suitable format for the solver.

        Returns:
        dict or None: The results of the optimization if successful, otherwise None.
        """
        if not self.define_vars(var_dict) is None:
            logging.error("Error defining variables")
            return None
        if not self.set_objective(obj_coeffs) is None:
            logging.error("Error setting objective function")
            return None
        if not self.add_constraints(constraints):
            logging.error("Error adding constraints")
            return None
        self.solve()
        return self.get_results()
import unittest
import pandas as pd
import sys
import os
from io import StringIO
import pulp

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from modelDefinition import RuleBasedModel

class TestModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load the Titanic dataset
        cls.file_path = 'datasets/titanic/train.csv'
        cls.data = pd.read_csv(cls.file_path)
        cls.data.columns = cls.data.columns.str.strip()  # Remove leading/trailing whitespaces from column names

    def setUp(self):
        self.model = RuleBasedModel()

    def test_model_initialization(self):
        self.assertIsInstance(self.model, RuleBasedModel)
        self.assertIsInstance(self.model.model, pulp.LpProblem)
        self.assertEqual(self.model.model.name, "Rule_Based_Model")
        self.assertEqual(self.model.model.sense, pulp.LpMaximize)

    def test_set_logging(self):
        self.assertTrue(self.model.set_logging(level='DEBUG'))
        self.assertFalse(self.model.set_logging(level='INVALID'))

    def test_define_vars(self):
        var_dict = {
            'x1': (0, None, pulp.LpContinuous),
            'x2': (0, None, pulp.LpContinuous)
        }
        self.assertIsNone(self.model.define_vars(var_dict))
        self.assertIn('x1', self.model.variables)
        self.assertIn('x2', self.model.variables)
        self.assertFalse(self.model.define_vars("invalid"))

    def test_define_vars_invalid(self):
        self.assertFalse(self.model.define_vars(None))
        self.assertFalse(self.model.define_vars({'x1': (0, None)}))

    def test_set_objective(self):
        var_dict = {
            'x1': (0, None, pulp.LpContinuous),
            'x2': (0, None, pulp.LpContinuous)
        }
        self.model.define_vars(var_dict)
        obj_coeffs = {'x1': 1, 'x2': 2}
        self.assertIsNone(self.model.set_objective(obj_coeffs))
        self.assertFalse(self.model.set_objective("invalid"))
        self.assertFalse(self.model.set_objective({'x3': 1}))

    def test_set_objective_invalid(self):
        self.assertFalse(self.model.set_objective(None))
        self.assertFalse(self.model.set_objective({'x3': 1}))

    def test_add_constraints(self):
        var_dict = {
            'x1': (0, None, pulp.LpContinuous),
            'x2': (0, None, pulp.LpContinuous)
        }
        self.model.define_vars(var_dict)
        constraints = [
            ({'x1': 1, 'x2': 1}, pulp.LpConstraintLE, 10),
            ({'x1': 1}, pulp.LpConstraintGE, 0)
        ]
        self.assertTrue(self.model.add_constraints(constraints))
        self.assertFalse(self.model.add_constraints("invalid"))
        self.assertFalse(self.model.add_constraints([("invalid", pulp.LpConstraintLE, 10)]))
        self.assertFalse(self.model.add_constraints([({'x1': 1}, 'invalid_sense', 10)]))
        self.assertFalse(self.model.add_constraints([({'x1': 1}, pulp.LpConstraintLE, 'invalid_rhs')]))
        self.assertFalse(self.model.add_constraints([({'x3': 1}, pulp.LpConstraintLE, 10)]))

    def test_add_constraints_invalid(self):
        self.assertFalse(self.model.add_constraints(None))
        self.assertFalse(self.model.add_constraints([({'x1': 1}, pulp.LpConstraintLE)]))

    def test_solve(self):
        var_dict = {
            'x1': (0, None, pulp.LpContinuous),
            'x2': (0, None, pulp.LpContinuous)
        }
        self.model.define_vars(var_dict)
        obj_coeffs = {'x1': 1, 'x2': 2}
        constraints = [
            ({'x1': 1, 'x2': 1}, pulp.LpConstraintLE, 10),
            ({'x1': 1}, pulp.LpConstraintGE, 0)
        ]
        self.model.set_objective(obj_coeffs)
        self.model.add_constraints(constraints)
        self.model.solve()
        self.assertIsNotNone(self.model.objective_value)

    def test_get_results(self):
        var_dict = {
            'x1': (0, None, pulp.LpContinuous),
            'x2': (0, None, pulp.LpContinuous)
        }
        self.model.define_vars(var_dict)
        obj_coeffs = {'x1': 1, 'x2': 2}
        constraints = [
            ({'x1': 1, 'x2': 1}, pulp.LpConstraintLE, 10),
            ({'x1': 1}, pulp.LpConstraintGE, 0)
        ]
        self.model.set_objective(obj_coeffs)
        self.model.add_constraints(constraints)
        self.model.solve()
        results = self.model.get_results()
        self.assertIsInstance(results, dict)
        self.assertIn('x1', results)
        self.assertIn('x2', results)

    def test_run(self):
        var_dict = {
            'x1': (0, None, pulp.LpContinuous),
            'x2': (0, None, pulp.LpContinuous)
        }
        obj_coeffs = {'x1': 1, 'x2': 2}
        constraints = [
            ({'x1': 1, 'x2': 1}, pulp.LpConstraintLE, 10),
            ({'x1': 1}, pulp.LpConstraintGE, 0)
        ]
        results = self.model.run(var_dict, obj_coeffs, constraints)
        self.assertIsInstance(results, dict)
        self.assertIn('x1', results)
        self.assertIn('x2', results)

    def test_run_invalid(self):
        self.assertIsNone(self.model.run(None, None, None))
        self.assertIsNone(self.model.run({'x1': (0, None)}, None, None))
        self.assertIsNone(self.model.run({'x1': (0, None)}, {'x1': 1}, None))
        self.assertIsNone(self.model.run({'x1': (0, None)}, {'x1': 1}, [({'x1': 1}, pulp.LpConstraintLE)]))

    def test_solve_exception(self):
        self.model.model = None
        with self.assertLogs(level='ERROR') as log:
            self.model.solve()
            self.assertIn('Error solving model', log.output[0])

    def test_get_results_exception(self):
        self.model.model = None
        with self.assertLogs(level='ERROR') as log:
            results = self.model.get_results()
            self.assertIsNone(results)
            self.assertIn('Error retrieving results', log.output[0])

    def test_invalid_logging_level(self):
        self.assertFalse(self.model.set_logging(level='INVALID'))

if __name__ == '__main__':
    unittest.main()
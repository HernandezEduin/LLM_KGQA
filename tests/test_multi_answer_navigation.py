import unittest
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from utils.graph_utils import RelationEntityGrapher
from utils.kgqa_navigation_metrics import (
    score_path_fidelity_against_references,
    score_single_final_entity,
)

class MultiAnswerNavigationTests(unittest.TestCase):
    def setUp(self):
        self.graph = {
            ('s', 'r1', 'a'),
            ('s', 'r1', 'b'),
            ('a', 'r2', 'x'),
            ('b', 'r2', 'x'),
            ('b', 'r2', 'z'),
            ('a', 'wrong', 'y'),
            ('s', 'other', 'b'),
        }
        self.grapher = RelationEntityGrapher(self.graph)
        self.relation_chain = ['r1', 'r2']

    def test_relation_index_is_lazy_and_releasable(self):
        self.assertIsNone(self.grapher._relation_index)

        self.grapher.find_paths_by_relation_chain(
            start_entity='s',
            relation_chain=self.relation_chain,
            target_entities={'x', 'z'},
        )
        self.assertIsNotNone(self.grapher._relation_index)

        self.grapher.clear_relation_index()
        self.assertIsNone(self.grapher._relation_index)

    def test_multiple_answers_and_multiple_entity_realizations(self):
        paths = self.grapher.find_paths_by_relation_chain(
            start_entity='s',
            relation_chain=self.relation_chain,
            target_entities={'x', 'z'},
        )

        self.assertEqual(len(paths), 3)
        self.assertEqual({path[-1][2] for path in paths}, {'x', 'z'})
        for path in paths:
            self.assertEqual([edge[1] for edge in path], self.relation_chain)
            self.assertEqual(path[0][0], 's')

    def test_one_answer_can_have_multiple_valid_paths(self):
        paths = self.grapher.find_paths_by_relation_chain(
            start_entity='s',
            relation_chain=self.relation_chain,
            target_entities={'x'},
        )

        self.assertEqual(len(paths), 2)
        self.assertTrue(all(path[-1][2] == 'x' for path in paths))

    def test_multiple_answers_can_leave_only_one_valid_path(self):
        paths = self.grapher.find_paths_by_relation_chain(
            start_entity='s',
            relation_chain=self.relation_chain,
            target_entities={'z', 'missing'},
        )

        self.assertEqual(paths, [[('s', 'r1', 'b'), ('b', 'r2', 'z')]])

    def test_wrong_relation_sequence_is_not_a_gold_path(self):
        paths = self.grapher.find_paths_by_relation_chain(
            start_entity='s',
            relation_chain=self.relation_chain,
            target_entities={'y', 'z'},
        )

        self.assertEqual(paths, [[('s', 'r1', 'b'), ('b', 'r2', 'z')]])
        self.assertNotIn([('s', 'r1', 'a'), ('a', 'wrong', 'y')], paths)

    def test_path_metrics_select_best_valid_reference(self):
        reference_paths = self.grapher.find_paths_by_relation_chain(
            start_entity='s',
            relation_chain=self.relation_chain,
            target_entities={'x', 'z'},
        )
        predicted = [('s', 'r1', 'b'), ('b', 'r2', 'z')]

        score = score_path_fidelity_against_references(
            predicted_path=predicted,
            reference_paths=reference_paths,
            reference_relation_chain=self.relation_chain,
        )

        self.assertEqual(score['PED'], 0.0)
        self.assertEqual(score['F1_SG'], 1.0)
        self.assertEqual(score['RED'], 0.0)
        self.assertEqual(score['F1_REL'], 1.0)

    def test_correct_answer_through_wrong_relation_sequence_is_not_perfect_path(self):
        reference_paths = self.grapher.find_paths_by_relation_chain(
            start_entity='s',
            relation_chain=self.relation_chain,
            target_entities={'z'},
        )
        predicted = [('s', 'other', 'b'), ('b', 'r2', 'z')]

        answer_score = score_single_final_entity('z', {'x', 'z'})
        path_score = score_path_fidelity_against_references(
            predicted_path=predicted,
            reference_paths=reference_paths,
            reference_relation_chain=self.relation_chain,
        )

        self.assertEqual(answer_score['Hits1'], 1.0)
        self.assertGreater(path_score['PED'], 0.0)
        self.assertLess(path_score['F1_SG'], 1.0)
        self.assertGreater(path_score['RED'], 0.0)
        self.assertLess(path_score['F1_REL'], 1.0)

    def test_invalid_terminal_answer_is_incorrect(self):
        score = score_single_final_entity('y', {'x', 'z'})
        self.assertEqual(score['Hits1'], 0.0)
        self.assertEqual(score['final_entity_correct'], 0.0)


if __name__ == '__main__':
    unittest.main()

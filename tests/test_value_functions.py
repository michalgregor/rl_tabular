import unittest
import numpy as np
from rl_tabular.value_functions import ValueTable
from rl_tabular import collect_states
from gym_plannable.env import MazeEnv

class ValueTableIndexing(unittest.TestCase):
    def setUp(self):
        self.vtable = ValueTable(np.zeros(0))
        env = MazeEnv()
        self.states = collect_states(env.single_plannable_state())

    def tearDown(self):
        del self.vtable

    def testGetDefault(self):
        self.assertEqual(self.vtable[self.states[0]], 0)
        # assert still empty
        self.assertEqual(len(self.vtable.values), 0)
        self.assertEqual(len(self.vtable.index), 0)

    def testGetMultipleDefault(self):
        vals = self.vtable[self.states[:5]]
        self.assertEqual(len(vals.shape), 1)
        self.assertEqual(vals.shape[0], 5)
        self.assertTrue((self.vtable[self.states[0]] == np.zeros(5)).all())
        # assert still empty
        self.assertEqual(len(self.vtable.values), 0)
        self.assertEqual(len(self.vtable.index), 0)

    def testGetDefaultWithRest(self):
        with self.assertRaises(IndexError):
            self.vtable[self.states[0], 2]

    def testSet(self):
        self.vtable[self.states[0]] = 11
        self.assertEqual(self.vtable[self.states[0]], 11)

    def testSetMultiple(self):
        self.vtable[[self.states[0], self.states[1]]] = [11, 12]
        self.assertEqual(self.vtable[self.states[0]], 11)
        self.assertEqual(self.vtable[self.states[1]], 12)
        self.assertTrue((self.vtable[[self.states[0], self.states[1]]] == [11, 12]).all())

    def testIn(self):
        self.vtable[self.states[1]] = 11
        self.assertTrue(not self.states[0] in self.vtable)
        self.assertTrue(self.states[1] in self.vtable)

    def testDel(self):
        self.vtable[self.states[:2]] = 11
        del self.vtable[self.states[1:3]]
        self.assertEqual(len(self.vtable), 1)

class ValueTableAutoAdd(unittest.TestCase):
    def setUp(self):
        self.vtable = ValueTable(np.zeros(0), auto_add_missing=True)
        env = MazeEnv()
        self.states = collect_states(env.single_plannable_state())

    def tearDown(self):
        del self.vtable

    def testGetDefault(self):
        self.assertEqual(self.vtable[self.states[0]], 0)
        # assert auto inserted
        self.assertEqual(len(self.vtable.values), 1)
        self.assertEqual(len(self.vtable.index), 1)
        self.assertEqual(self.vtable[self.states[0]], 0)

    def testGetMultipleDefault(self):
        vals = self.vtable[self.states[:5]]
        self.assertEqual(len(vals.shape), 1)
        self.assertEqual(vals.shape[0], 5)
        self.assertTrue((self.vtable[self.states[0]] == np.zeros(5)).all())
        # assert auto inserted
        self.assertEqual(len(self.vtable.values), 5)
        self.assertEqual(len(self.vtable.index), 5)
        self.assertTrue(
            (self.vtable[self.states[:5]] == [0, 0, 0, 0, 0]).all()
        )

class ValueTableIndexingND(unittest.TestCase):
    def setUp(self):
        self.vtable = ValueTable(np.zeros((0, 4, 2)))
        env = MazeEnv()
        self.states = collect_states(env.single_plannable_state())

    def tearDown(self):
        del self.vtable

    def testGetDefault(self):
        self.assertTrue((self.vtable[self.states[0]] == np.zeros((4, 2))).all())
        # assert still empty
        self.assertEqual(len(self.vtable.values), 0)
        self.assertEqual(len(self.vtable.index), 0)

    def testGetMultipleDefault(self):
        vals = self.vtable[self.states[:5]]
        self.assertEqual(len(vals.shape), 3)
        self.assertEqual(vals.shape, (5, 4, 2))
        self.assertTrue((self.vtable[self.states[0]] == np.zeros((5, 4, 2))).all())
        # assert still empty
        self.assertEqual(len(self.vtable.values), 0)
        self.assertEqual(len(self.vtable.index), 0)

    def testGetDefaultWithRest(self):
        self.assertEqual(self.vtable[self.states[0], 2, 0], 0)
        self.assertTrue((self.vtable[self.states[0], 2] == [0, 0]).all())
        self.assertTrue((self.vtable[self.states[0], [0, 1], 0] == [0, 0]).all())
        self.assertTrue((self.vtable[self.states[:2], [0, 1], 0] == 
            [[0, 0], [0, 0]]).all())

    def testSet(self):
        self.vtable[self.states[0]] = 11
        self.assertTrue((self.vtable[self.states[0]] == 11).all())
        # indexing multiple values using lists
        self.vtable[[self.states[0], self.states[1]], [0, 1], [0, 0]] = [11, 12]
        self.assertTrue((self.vtable[[self.states[0], self.states[1]], [0, 1], [0, 0]] == [11, 12]).all())

    def testSetSlice(self):
        vals = np.array(range(8)).reshape(2, 4)
        self.vtable[self.states[:2], :, 0] = vals
        self.assertTrue((self.vtable[self.states[:2], :, 0] == vals).all())

    def testSetDimAfterDim(self):
        vals = np.array(range(4)).reshape(2, 2)
        self.vtable[self.states[:2]] = 11

        tab = self.vtable[self.states[:2]]
        tab[:, [0, 1], 0] = vals
        self.vtable[self.states[:2]] = tab

        self.assertTrue((self.vtable[self.states[:2]][:, [0, 1], 0] == vals).all())

class ValueTableNumpy(unittest.TestCase):
    def setUp(self):
        self.vtable1 = ValueTable(np.zeros(0))
        self.vtable2 = ValueTable(np.zeros(0))
        env = MazeEnv()
        self.states = collect_states(env.single_plannable_state())

    def tearDown(self):
        del self.vtable1
        del self.vtable2

    def testAsArray(self):
        ar = np.asarray(self.vtable1)
        self.assertEqual(ar.shape, (0,))

    def testAdd(self):
        self.vtable1[self.states[:2]] = [1, 1]
        self.vtable2[self.states[1:3]] = [1, 1]
        vtable3 = self.vtable1 + self.vtable2
        self.assertTrue((vtable3[self.states[:3]] == [1, 2, 1]).all())

    def testReduce(self):
        self.vtable1[self.states[:3]] = [1, 2, 3]
        self.assertEqual(np.max(self.vtable1), 3)
        self.assertEqual(np.sum(self.vtable1), 6)

class ValueTableNumpyND(unittest.TestCase):
    def setUp(self):
        self.vtable = ValueTable(np.zeros((0, 4)))
        env = MazeEnv()
        self.states = collect_states(env.single_plannable_state())

    def tearDown(self):
        del self.vtable

    def testReduce(self):
        self.vtable[self.states[:3]] = np.array(range(12)).reshape(3, 4)
        self.assertTrue((np.max(self.vtable[self.states[:3]], axis=1) == [3, 7, 11]).all())
from .utils import OrderedSet
import numpy as np
import copy

def compute_action_value(state, action, vtable, discount=0.9):
    if not action in state.legal_actions():
        return 0
    
    val = 0
    
    for next_state, prob in state.all_next(action):
        val += prob * (next_state.reward() + discount * vtable[next_state])
        
    return val

def compute_state_value(state, qtable, policy=None):
    if policy is None:
        return np.max(qtable[state, :])
    else:
        proba = policy.proba(state)
        return (qtable[state, :] * proba).sum()
        
class NumpyMixin(np.lib.mixins.NDArrayOperatorsMixin):
    def __array__(self, dtype=None):
        assert dtype is None
        return self.values
        
    def _join_index(self, table, index):
        if isinstance(table, self.__class__):
            index = index.union(table.index)
        return index

    def _ufunc_unwrap(self, inputs, index=None, inplace=False):
        if index is None:
            index = OrderedSet()
            for inp in inputs:
                index = self._join_index(inp, index)

        inputs_unwrapped = [
            inp.reindex(index, inplace=inplace)[1]
                if isinstance(inp, self.__class__)
                else inp for inp in inputs
        ]

        return inputs_unwrapped, index

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):        
        if method == '__call__' or method == 'accumulate':
            inputs, index = self._ufunc_unwrap(inputs)
            func = getattr(ufunc, method)
            
            outputs = kwargs.pop('out', None)
            if outputs:
                assert(len(outputs) == 1)
                table = outputs[0]
                table.reindex(index, inplace=True)
            else:
                tmp_values, tmp_index = self.values, self.index
                self.values, self.index = np.zeros(0), []
                table = self.copy()
                self.values, self.index = tmp_values, tmp_index
                table.index = index

            table.values = func(*inputs, **kwargs)
            return table
        
        elif method == 'reduce':
            inputs, index = self._ufunc_unwrap(inputs)
            res = ufunc.reduce(*inputs, **kwargs)
            return res
            
        else:
            return NotImplemented
    
class StateKeyFunc:
    def __init__(self, obs_dtype=np.int32):
        self.dtype = obs_dtype

    def __call__(self, s):
        if hasattr(s, "observation"):
            return np.asarray(s.observation(), dtype=self.dtype).tobytes()
        elif isinstance(s, bytes):
            return s
        else:
            return np.asarray(s, dtype=self.dtype).tobytes()

class ValueTable(NumpyMixin):
    def __init__(
        self,
        values,
        index=None,
        default_value=0,
        state_key_func=StateKeyFunc(),
        auto_add_missing=False
    ):
        if index is None:
            self.index = OrderedSet()
        else:
            self.index = OrderedSet(index)

        self.values = values
        self.default_value = default_value
        self.state_key_func = state_key_func
        self.auto_add_missing = auto_add_missing

    @property
    def shape(self):
        return self.values.shape

    def _proc_key(self, key, recurse=True):
        rest = tuple()

        if isinstance(key, list):
            if not recurse: raise IndexError("A single key, not a list expected.")
            key = [self._proc_key(k, recurse=False)[0] for k in key]
        elif isinstance(key, tuple):
            if not recurse: raise IndexError("Too many indices.")
            key, rest = self._proc_key(key[0], recurse=True)[0], key[1:]
            if not isinstance(rest, tuple): rest = tuple(rest)
        else:
            key = self.state_key_func(key)
        
        return key, rest

    def reindex(self, keys, inplace=False):
        ar_shape = tuple(self.values.shape[1:])
        index = OrderedSet(keys)
        values = np.empty(((len(index),) + ar_shape), dtype=self.values.dtype)

        for i, k in enumerate(index):
            ind = self.index.get(k)
            if ind is None:
                values[i] = np.full(ar_shape, self.default_value)
            else:
                values[i] = self.values[ind]

        if inplace:
            self.index = index
            self.values = values

        return index, values

    def _union_reindex(self, keys):
        keys_set = OrderedSet(keys)
        missing_keys = keys_set.difference(self.index)

        if len(missing_keys):
            new_keys = missing_keys.union(self.index)
            return self.reindex(new_keys)
        else:
            return self.index, self.values
    
    def _translate_keys(self, keys, index):
        if isinstance(keys, list):
            return [index.index(k) for k in keys]
        else:
            return index.index(keys)

    def __getitem__(self, key):
        keys, rest = self._proc_key(key)
        keys_lst = keys if isinstance(keys, list) else [keys]
        
        if self.auto_add_missing:
            self.index, self.values = index, values = self._union_reindex(keys_lst)
        else:
            index, values = self.reindex(keys_lst)

        return values[(self._translate_keys(keys, index),) + rest]

    def __setitem__(self, key, value):
        keys, rest = self._proc_key(key)
        keys_lst = keys if isinstance(keys, list) else [keys]
        
        self.index, self.values = self._union_reindex(keys_lst)
        self.values[(self._translate_keys(keys, self.index),) + rest] = value

    def __contains__(self, key):
        key, _ = self._proc_key(key, recurse=False)
        return key in self.index

    def __delitem__(self, key):
        keys, rest = self._proc_key(key)
        if len(rest) != 0: raise IndexError("Too many index dimensions.")
        keys = keys if isinstance(keys, list) else [keys]
        keys = OrderedSet(keys)
        remaining_index = OrderedSet(self.index).difference(keys)
        self.reindex(remaining_index, inplace=True)

    def copy(self):
        return copy.deepcopy(self)

    def __repr__(self):
        repre = ""
        for k, val in zip(self.index, self.values):
            repre += k.__repr__() + ": " + val.__repr__() + "\n"
        return repre
    
    def __len__(self):
        return len(self.values)

    def __iter__(self):
        return zip(self.index, self.values)

class StateValueTable(ValueTable):
    def __init__(
        self,
        default_value=0,
        state_key_func=StateKeyFunc(),
        auto_add_missing=False,
        dtype=float
    ):
        super().__init__(np.zeros(0, dtype=dtype),
                         default_value=default_value,
                         state_key_func=state_key_func,
                         auto_add_missing=auto_add_missing)

    def to_action_values(self, states, discount=0.9):
        n_actions = states[0].action_space.n

        qtable = ActionValueTable(n_actions)
        for s in states:
            if not s in self: continue
            qtable[s] = [compute_action_value(s, a, self, discount) for a in range(n_actions)] 

        return qtable

class ActionValueTable(ValueTable):
    def __init__(
        self,
        n_actions,
        default_value=0,
        state_key_func=StateKeyFunc(),
        auto_add_missing=False,
        dtype=float
    ):
        super().__init__(np.zeros((0, n_actions), dtype=dtype),
                         default_value=default_value,
                         state_key_func=state_key_func,
                         auto_add_missing=auto_add_missing)
        self.n_actions = n_actions

    def to_state_values(self, policy=None):
        vtable = StateValueTable()

        for obs in self.index:
            vtable[obs] = compute_state_value(obs, self, policy=policy)

        return vtable

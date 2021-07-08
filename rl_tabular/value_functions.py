import numpy as np
import pandas as pd
import copy

def compute_action_value(state, action, vtable, discount=0.9):
    if not action in state.legal_actions():
        return 0
    
    val = 0
    
    for next_state, prob in state.all_next(action):
        val += prob * (next_state.rewards() + discount * vtable[next_state])
        
    return val

def compute_opt_state_value(state, qtable):
    return np.max(qtable[state, :])
    
class NumpyMixin(np.lib.mixins.NDArrayOperatorsMixin):
    def __array__(self, dtype=None):
        assert dtype is None
        return self.values()
    
    def _reindex_table(self, table, index):
        if isinstance(table, self.__class__):
            if index is None:
                index = table.data.index
                table = table.data.values
            else:
                table = table.data.loc[index].values
        
        return table, index
    
    def _ufunc_unwrap(self, inputs, kwargs):
        index = None
        inputs2 = []
        
        for inp in inputs:
            inp, index = self._reindex_table(inp, index)
            inputs2.append(inp)
                        
        outputs = kwargs.pop('out', None)
        
        if outputs:
            outputs2 = []
            for out in outputs:
                out, index = self._reindex_table(out, index)
                outputs2.append(out)

            kwargs['out'] = tuple(outputs2)
        
        return inputs2, kwargs

class ActionValueTable(NumpyMixin):
    def __init__(
        self, n_actions,
        data=None, default_value=0,
        state_key_func=lambda s: np.asarray(s.observations()).tobytes() if hasattr(s, "observations")
            else np.asarray(s).tobytes()
    ):
        self.n_actions = n_actions
        self.state_key_func = state_key_func
        self.default_value = default_value
        
        if data is None:
            self._action_values = pd.DataFrame(
                data=[], index=[],
                columns=range(self.n_actions),
                dtype=float
            )
        else:
            self._action_values = data
    
    def copy(self):
        return copy.deepcopy(self)

    @property
    def data(self):
        return self._action_values

    @data.setter
    def data(self, value):
        self._action_values = value

    def keys(self):
        return self._action_values.index

    def values(self):
        return self._action_values.values
    
    def items(self):
        return self._action_values.items()
    
    def _proc_slice(self, s, key_func):
        if not s.start is None:
            start = key_func(s.start)
        else:
            start = s.start
            
        if not s.stop is None:
            stop = key_func(s.stop)
        else:
            stop = s.stop
            
        if not s.step is None:
            step = key_func(s.step)
        else:
            step = s.step
        
        return slice(start, stop, step)
    
    def _proc_key(self, key):
        if key is None:
            key = slice(None)

        if isinstance(key, tuple):
            assert len(key) == 2
            s_key, a_key = key
            if s_key is None: s_key = slice(None)
            if a_key is None: a_key = slice(None)
            
            if isinstance(s_key, slice):
                s_key = self._proc_slice(s_key, self.state_key_func)
            else:
                s_key = self.state_key_func(s_key)
            
            if isinstance(a_key, slice):
                a_key = self._proc_slice(a_key, lambda x: x)
            
            key = s_key, a_key
                        
        elif isinstance(key, slice):
            key = self._proc_slice(key, self.state_key_func)

        else:
            key = self.state_key_func(key)
        
        return key
    
    def get(self, state=None, action=None, default=None):
        key = self._proc_key((state, action))
        
        try:
            return self._action_values.loc[key]
        except KeyError:
            return default
    
    def __contains__(self, key):
        return self._proc_key(key) in self._action_values.index

    def __getitem__(self, key):
        if self.default_value is None:
            return self._action_values.loc[self._proc_key(key)]
        else:
            try:
                return self._action_values.loc[self._proc_key(key)]
            except KeyError:
                return self.default_value
    
    def __setitem__(self, key, value):
        self._action_values.loc[self._proc_key(key)] = value
        if not self.default_value is None:
            self._action_values.fillna(self.default_value, inplace=True)
    
    def __delitem__(self, key):
        key = self._proc_key(key)
        self._action_values.drop(index=[key], inplace=True)
        
    def remove(self, keys):
        self._action_values.drop(index=[self._proc_key(key) for key in keys], inplace=True)

    def __repr__(self):
        return self._action_values.__repr__()
    
    def __len__(self):
        return len(self._action_values)

    def to_state_values(self):
        vtable = StateValueTable()

        for obs in self.keys():
            vtable[obs] = compute_opt_state_value(obs, self)

        return vtable

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):        
        if method == '__call__' or method == 'accumulate':
            inputs, kwargs = self._ufunc_unwrap(inputs, kwargs)
            func = getattr(ufunc, method)
                    
            return self.__class__(
                self.n_actions,
                states=None, values=None,
                state_key_func=self.state_key_func,
                init_val_func=None,
                default_value=self.default_value,
                action_values=pd.DataFrame(
                    data=func(*inputs, **kwargs),
                    index=self.data.index,
                    columns=range(self.n_actions),
                    dtype=float
                )
            )
        
        elif method == 'reduce':
            inputs, kwargs = self._ufunc_unwrap(inputs, kwargs)
            res = ufunc.reduce(*inputs, **kwargs)
            return res
            
        else:
            return NotImplemented
    
class StateValueTable(NumpyMixin):
    def __init__(
        self,
        data=None, default_value=0, 
        state_key_func=lambda s: np.asarray(s.observations()).tobytes() if hasattr(s, "observations")
            else np.asarray(s).tobytes()
    ):
        self.state_key_func = state_key_func
        self.default_value = default_value
        
        if data is None:
            self._state_values = pd.Series(data=[], index=[], dtype=float)
        else:
            self._state_values = data

    def copy(self):
        return copy.deepcopy(self)

    @property
    def data(self):
        return self._state_values

    @data.setter
    def data(self, value):
        self._state_values = value

    def keys(self):
        return self._state_values.index

    def values(self):
        return self._state_values.values
    
    def items(self):
        return self._state_values.items()
    
    def _proc_slice(self, s, key_func):
        if not s.start is None:
            start = key_func(s.start)
        else:
            start = s.start
            
        if not s.stop is None:
            stop = key_func(s.stop)
        else:
            stop = s.stop
            
        if not s.step is None:
            step = key_func(s.step)
        else:
            step = s.step
        
        return slice(start, stop, step)
    
    def _proc_key(self, key):
        if key is None:
            key = slice(None)

        if isinstance(key, slice):
            key = self._proc_slice(key, self.state_key_func)

        else:
            key = self.state_key_func(key)
        
        return key
    
    def get(self, state=None, default=None):
        return self._state_values.get(self._proc_key(state), default=default)

    def __contains__(self, key):
        return self._proc_key(key) in self._state_values.index

    def __getitem__(self, state):
        if self.default_value is None:
            return self._state_values.loc[self._proc_key(state)]
        else:
            try:
                return self._state_values.loc[self._proc_key(state)]
            except KeyError:
                return self.default_value
    
    def __setitem__(self, state, value):
        self._state_values.loc[self._proc_key(state)] = value
    
    def __delitem__(self, state):
        self._state_values.drop([self._proc_key(state)], inplace=True)
        
    def remove(self, states):
        self._state_values.drop([self._proc_key(state) for state in states], inplace=True)

    def __repr__(self):
        return self._state_values.__repr__()
    
    def __len__(self):
        return len(self._state_values)
    
    def to_action_values(self, states, discount=0.9):
        n_actions = states[0].action_space.n

        qtable = ActionValueTable(n_actions)
        for s in states:
            if not s in self: continue
            qtable[s] = [compute_action_value(s, a, self, discount) for a in range(n_actions)] 

        return qtable
    
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):        
        if method == '__call__' or method == 'accumulate':
            inputs, kwargs = self._ufunc_unwrap(inputs, kwargs)
            func = getattr(ufunc, method)
            
            return self.__class__(
                states=None, values=None,
                state_key_func=self.state_key_func,
                init_val_func=None,
                default_value=self.default_value,
                state_values=pd.Series(
                    data=func(*inputs, **kwargs),
                    index=self.data.index,
                    dtype=float
                )
            )
        
        elif method == 'reduce':
            inputs, kwargs = self._ufunc_unwrap(inputs, kwargs)
            res = ufunc.reduce(*inputs, **kwargs)
            return res
            
        else:
            return NotImplemented
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
        
    def _join_index(self, table, index):
        if isinstance(table, self.__class__):
            index = index.union(table.data.index)
        return index
    
    def _ufunc_unwrap(self, inputs, kwargs):
        index = pd.Index([])
        for inp in inputs:
            index = self._join_index(inp, index)

        inputs2 = [
            inp.data.reindex(index, fill_value=inp.default_value, copy=False).values
                if isinstance(inp, self.__class__) else inp for inp in inputs
        ]
                                
        outputs = kwargs.pop('out', None)
        
        if outputs:
            outputs2 = []
            for out in outputs:
                if isinstance(out, self.__class__):
                    out.data = out.data.reindex(index,
                        fill_value=out.default_value, copy=False
                    )
                    out = out.data.values
                outputs2.append(out)

            kwargs['out'] = tuple(outputs2)
        
        return inputs2, kwargs, index

class StateKeyFunc:
    def __init__(self, obs_dtype=np.int32):
        self.dtype = obs_dtype

    def __call__(self, s):
        if hasattr(s, "observations"):
            return np.asarray(s.observations(), dtype=self.dtype).tobytes()
        elif isinstance(s, bytes):
            return s
        else:
            return np.asarray(s, dtype=self.dtype).tobytes()

class ActionValueTable(NumpyMixin):
    def __init__(
        self, n_actions,
        data=None, default_value=0,
        state_key_func=StateKeyFunc(),
        dtype=float
    ):
        self.n_actions = n_actions
        self.state_key_func = state_key_func
        self.default_value = default_value
        self.dtype = dtype
        
        if data is None:
            self._action_values = pd.DataFrame(
                data=[], index=[],
                columns=range(self.n_actions),
                dtype=self.dtype
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

        if isinstance(key, list):
            key = [self._proc_key(k) for k in key]

        elif isinstance(key, tuple):
            assert len(key) == 2
            s_key, a_key = key
            if s_key is None: s_key = slice(None)
            if a_key is None: a_key = slice(None)
            
            if isinstance(s_key, list):
                s_key = [
                    self._proc_slice(k, self.state_key_func)
                    if isinstance(k, slice)
                    else self.state_key_func(k)
                    for k in s_key
                ]

            elif isinstance(s_key, slice):
                s_key = self._proc_slice(s_key, self.state_key_func)
            else:
                s_key = self.state_key_func(s_key)
            
            if isinstance(a_key, list):
                a_key = [
                    self._proc_slice(k, lambda x: x)
                    if isinstance(k, slice) else k
                    for k in a_key
                ]
            
            elif isinstance(a_key, slice):
                a_key = self._proc_slice(a_key, lambda x: x)
            
            key = s_key, a_key
                        
        elif isinstance(key, slice):
            key = self._proc_slice(key, self.state_key_func)

        else:
            key = self.state_key_func(key)
        
        return key
    
    def __contains__(self, key):
        return self._proc_key(key) in self._action_values.index

    def _reindex(self, keys):
        keys = self._proc_key(keys)

        if isinstance(keys, tuple):
            assert(len(keys) == 2)
            state_keys, action_keys = keys
            state_keys_orig = state_keys
        else:
            state_keys = state_keys_orig = keys
            action_keys = slice(None)
            
        if not isinstance(state_keys, list):
            state_keys = [state_keys]
        else:
            state_keys_orig = np.asarray(state_keys_orig, dtype=object)

        state_keys = self._action_values.index.__class__(state_keys)
        missing_keys = state_keys.difference(self._action_values.index)

        if len(missing_keys):
            new_entries = pd.DataFrame(
                np.full(
                    (len(missing_keys), self._action_values.shape[1]),
                    self.default_value
                ), columns=self._action_values.columns, index=missing_keys
            )

            action_values = pd.concat([self._action_values, new_entries])
        else:
            action_values = self._action_values
            
        return action_values, state_keys_orig, action_keys

    def __getitem__(self, key):
        if self.default_value is None:
            return self._action_values.loc[self._proc_key(key)]
        else:
            action_values, state_keys, action_keys = self._reindex(key)
            return action_values.loc[state_keys, action_keys]

    def __setitem__(self, key, value):
        self._action_values, state_keys, action_keys = self._reindex(key)
        self._action_values.loc[state_keys, action_keys] = np.asarray(value, dtype=self.dtype)
    
    def __delitem__(self, key):
        keys = self._proc_key(key)
        if isinstance(keys, tuple): raise ValueError("Only entire items can be deleted: i.e. key must not be a tuple.")
        if not isinstance(keys, list): keys = [keys]
        self._action_values.drop(index=keys, inplace=True)
        
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
            inputs, kwargs, index = self._ufunc_unwrap(inputs, kwargs)
            func = getattr(ufunc, method)
                    
            return self.__class__(
                self.n_actions,
                data=pd.DataFrame(
                    data=func(*inputs, **kwargs),
                    index=index,
                    columns=range(self.n_actions),
                    dtype=self.dtype
                ),
                default_value=self.default_value,
                state_key_func=self.state_key_func
            )
        
        elif method == 'reduce':
            inputs, kwargs, index = self._ufunc_unwrap(inputs, kwargs)
            res = ufunc.reduce(*inputs, **kwargs)
            return res
            
        else:
            return NotImplemented
    
class StateValueTable(NumpyMixin):
    def __init__(
        self,
        data=None, default_value=0, 
        state_key_func=StateKeyFunc(),
        dtype=float
    ):
        self.state_key_func = state_key_func
        self.default_value = default_value
        self.dtype = dtype
        
        if data is None:
            self._state_values = pd.Series(data=[], index=[], dtype=self.dtype)
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

        if isinstance(key, list):
            key = [
                self._proc_slice(k, self.state_key_func)
                if isinstance(k, slice) else self.state_key_func(k)
                for k in key
            ]

        elif isinstance(key, slice):
            key = self._proc_slice(key, self.state_key_func)

        else:
            key = self.state_key_func(key)
        
        return key

    def __contains__(self, key):
        return self._proc_key(key) in self._state_values.index

    def __getitem__(self, state):
        if self.default_value is None:
            return self._state_values.loc[self._proc_key(state)]
        else:
            keys = self._proc_key(state)
            keys = self._state_values.index.__class__(keys)
            missing_keys = keys.difference(self._state_values.index)

            if len(missing_keys):
                new_entries = pd.Series(self.default_value, index=missing_keys)
                df = pd.concat([self._state_values, new_entries])
                return df.loc[keys]
            else:
                return self._state_values.loc[keys]

    def __setitem__(self, state, value):
        keys = self._proc_key(state)

        if isinstance(keys, list) and not self.default_value is None:
            keys = self._state_values.index.__class__(keys)
            missing_keys = keys.difference(self._state_values.index)
            new_entries = pd.Series(self.default_value, index=missing_keys)
            self._state_values = pd.concat([self._state_values, new_entries])
            self._state_values.loc[keys] = np.asarray(value, dtype=self.dtype)
        else:
            self._state_values.loc[keys] = np.asarray(value, dtype=self.dtype)
    
    def __delitem__(self, state):
        keys = self._proc_key(state)
        if not isinstance(keys, list): keys = [keys]
        self._state_values.drop(keys, inplace=True)
        
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
            inputs, kwargs, index = self._ufunc_unwrap(inputs, kwargs)
            func = getattr(ufunc, method)
            
            return self.__class__(
                data=pd.Series(
                    data=func(*inputs, **kwargs),
                    index=index,
                    dtype=self.dtype
                ),
                default_value=self.default_value,
                state_key_func=self.state_key_func
            )
        
        elif method == 'reduce':
            inputs, kwargs, index = self._ufunc_unwrap(inputs, kwargs)
            res = ufunc.reduce(*inputs, **kwargs)
            return res
            
        else:
            return NotImplemented
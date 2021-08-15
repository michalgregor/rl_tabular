import abc

class Callback:
    @abc.abstractmethod
    def __call__(self, step, train_info):
        raise NotImplementedError()

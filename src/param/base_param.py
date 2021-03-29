class BaseParam(object):

    @classmethod
    def _default_values(cls):
        data = {key: cls.__dict__[key] for key in cls.__dict__.keys() if not key.startswith('_')}
        return data

    def __str__(self):
        inKeys = set([key for key in self.__dict__.keys() if not key.startswith('__')])
        clzKeys = set([key for key in self.__class__.__dict__.keys() if not key.startswith('__')])
        keys = inKeys.union(clzKeys)
        out = ''
        for key in keys:
            if key in self.__dict__:
                out += '%s:%s\n' % (key, self.__dict__[key])
            else:
                out += '%s:%s\n' % (key, self.__class__.__dict__[key])
        return out
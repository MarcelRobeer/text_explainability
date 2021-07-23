"""Default classes for all to inherit from."""


class Readable:
    """Ensure that a class has a readable representation."""

    def __repr__(self):
        public_vars = ', '.join([f'{k}={v}' for k, v in vars(self).items() if not k.startswith('_')])
        return f'{self.__class__.__name__}({public_vars})'

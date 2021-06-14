class Readable:
    
    def __repr__(self):
        public_vars = ', '.join([f'{k}={v}' for k, v in vars(self).items() if not k.startswith('_')])
        return f'{self.__class__.__name__}({public_vars})'

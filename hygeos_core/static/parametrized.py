def parametrized(dec):
    """
    Utility to parametrize a decorator
    """
    def layer(*args, **kwargs):
        def repl(f):
            return dec(f, *args, **kwargs)
        return repl
    return layer
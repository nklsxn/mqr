def bounded_error_msg(bounded):
    return f'Invalid bound "{bounded}". Use both, below or above.'

def compare_error_msg(compare):
    return f'Invalid comparison "{compare}". Use diff or ratio.'

def method_error_msg(method, available):
    msg = f'Method "{method}" is not available. '
    if len(available) == 1:
        msg += f'Use {available[0]}.'
    else:
        msg += f'Use {", ".join(available[:-1])} or {available[-1]}.'
    return msg

def alternative_error_msg(alternative):
    return f'Invalid alternative "{alternative}". Use two-sided, less or greater.'

def var_name(var, default):
    if hasattr(var, 'name') and (var.name is not None) and (len(var.name) > 0):
        return var.name
    else:
        return default

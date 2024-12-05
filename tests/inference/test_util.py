import pytest

from mqr.inference.lib import util

def test_bounded_error_msg():
    assert (util.bounded_error_msg('abcdef') ==
        'Invalid bound "abcdef". Use both, below or above.')

def test_compare_error_msg():
    assert (util.compare_error_msg('abcdef') ==
        'Invalid comparison "abcdef". Use diff or ratio.')

def test_method_error_msg():
    assert (util.method_error_msg('abcdef', ['123x']) ==
        'Method "abcdef" is not available. Use 123x.')
    assert (util.method_error_msg('abcdef', ['123x', '123y']) ==
        'Method "abcdef" is not available. Use 123x or 123y.')
    assert (util.method_error_msg('abcdef', ['123x', '123y', '123z']) ==
        'Method "abcdef" is not available. Use 123x, 123y or 123z.')

def test_alternative_error_msg():
    assert (util.alternative_error_msg('abcdef') ==
        'Invalid alternative "abcdef". Use two-sided, less or greater.')

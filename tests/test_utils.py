'''
Check call-throughs.
'''

import mqr.utils

def test_make_ordinal():
    assert mqr.utils.make_ordinal(1) == '1st'
    assert mqr.utils.make_ordinal(2) == '2nd'
    assert mqr.utils.make_ordinal(3) == '3rd'
    assert mqr.utils.make_ordinal(4) == '4th'
    assert mqr.utils.make_ordinal(5) == '5th'

    assert mqr.utils.make_ordinal(10) == '10th'
    assert mqr.utils.make_ordinal(11) == '11th'
    assert mqr.utils.make_ordinal(12) == '12th'
    assert mqr.utils.make_ordinal(13) == '13th'
    assert mqr.utils.make_ordinal(14) == '14th'

    assert mqr.utils.make_ordinal(21) == '21st'
    assert mqr.utils.make_ordinal(22) == '22nd'
    assert mqr.utils.make_ordinal(23) == '23rd'
    assert mqr.utils.make_ordinal(24) == '24th'
    assert mqr.utils.make_ordinal(25) == '25th'

    assert mqr.utils.make_ordinal(1.1) == '1.1th'
    assert mqr.utils.make_ordinal(5.1) == '5.1th'
    assert mqr.utils.make_ordinal(134.1) == '134.1th'

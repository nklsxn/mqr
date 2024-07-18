"""
Defaults for text-based tables.
"""

from rich import box

# The box from the documentation. Here to copy easily for new border styles
# (
# '┌─┬┐\n' # top
# '│ ││\n' # head
# '├─┼┤\n' # head_row
# '│ ││\n' # mid
# '├─┼┤\n' # row
# '├─┼┤\n' # foot_row
# '│ ││\n' # foot
# '└─┴┘'   # bottom
# )

def default_table_box():
    return box.Box(
    '────\n' # top
    '    \n' # head
    '────\n' # head_row
    '    \n' # mid
    '────\n' # row
    '    \n' # foot_row
    '    \n' # foot
    '────'   # bottom
    )

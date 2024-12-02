.. py:module:: {{ fullname }}

.. raw:: html

   <div class="prename">{{ fullname.rsplit('.', 1)[0] }}.</div>
   <div class="empty"></div>

{{ name }}
{{ underline }}

.. currentmodule:: {{ fullname }}


{% if attributes %}
.. rubric:: Module Attributes
.. autosummary::
   :toctree:

{% for item in attributes %}
   {{ item }}
{%- endfor %}
{% endif %}


{% if functions %}
.. rubric:: Functions
.. autosummary::
   :toctree:

{% for item in functions %}
   {{ item }}
{%- endfor %}
{% endif %}


{% if classes %}
.. rubric:: Classes
.. autosummary::
   :toctree:

{% for item in classes %}
   {{ item }}
{%- endfor %}
{% endif %}

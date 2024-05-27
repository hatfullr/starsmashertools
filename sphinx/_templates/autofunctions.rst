
{{ fullname | escape | underline }}


.. automodule:: {{ fullname }}
   :no-members:

{% block functions %}
{% if functions %}

Functions
---------

.. autosummary::
   :template: autosummary.rst
   :toctree:

{% for item in functions %}
   {{ item }}{% endfor %}

{% endif %}
{% endblock %}

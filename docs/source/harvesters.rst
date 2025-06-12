========================
Harvesters documentation
========================

This document will keep a track about different processes 
and problems encountered during using Harvesters module.

Refer to `harvesters`_ repository and `documentation`_ for more information.

.. _harvesters: https://github.com/genicam/harvesters
.. _documentation: https://harvesters.readthedocs.io/en/latest/?badge=latest 

Pixel type information 
----------------------

.. code-block::

    from harvesters.core import Harvester
    h = Harvester()
    h.add_file('/opt/sentech/lib/libstgentl.cti')
    h.update()
    ia = h.create(0)
    print(ia.remote_device.node_map.PixelFormat.symbolics)

The following output will be displayed::

    ('BayerRG8', 'BayerRG10', 'BayerRG10p', 'BayerRG12', 'BayerRG12p')
Dimagi Clockify CLI
===================

A Clockify command line interface, for the way Dimagi uses it.


Usage
-----

Start clocking time to the "gtd_meeting" bucket::

    $ dcl gtd_meeting

Stop clocking time::

    $ dcl stop

Start clocking time to the "jamaica" bucket since 14:00::

    $ dcl jamaica --since 14:00


Requirements
------------

* Python 3.8 or higher
* `Poetry <https://python-poetry.org/>`_


Installation
------------

Clone the repository, and install using Poetry::

    $ git clone https://github.com/kaapstorm/dimagi-clockify-cli.git
    $ cd dimagi-clockify-cli
    $ poetry install

``poetry install`` will create a virtualenv, install requirements, and
put the ``dcl`` command in the virtualenv's ``bin`` directory. Activate
the virtualenv using ::

    $ poetry shell


Configuration
-------------

1. Create a config directory::

       $ mkdir ~/.config/dimagi-clockify-cli

   To use a different config directory, set an environment
   variable named ``DCL_CONFIG_DIR`` to the directory you prefer.

2. Copy the template config file to
   ``~/.config/dimagi-clockify-cli/config.yaml`` (or
   ``$DCL_CONFIG_DIR/config.yaml``)::

       $ cp config.template.yaml ~/.config/dimagi-clockify-cli/config.yaml

3. Edit the new config.yaml file to set your team's projects and tasks,
   and to add your buckets.

An implementation of the Vogels et al 2011 network in PyNN


Running instructions
--------------------

- Set up a Python3 virtual environment using the `virtualenv` tool: `virtualenv-3 $HOME/pynn-virt`
- This has set up the virtual environment at `$HOME/pynn-virt`
- activate the virtual envivonment: `source $HOME/pynn-virt/bin/activate`
- install required python modules: `pip3 install -r requirements.txt`
- To use NEST, please update your `LD_LIBRARY_PATH` variable: `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/pynn-virt/lib/python3.6/site-packages/pyNN/nest/_build/`
- Copy the required NEST sli init file: `cp $HOME/pynn-virt/lib/python3.6/site-packages/pyNN/nest/extensions/sli/pynn_extensions-init.sli . `
- Now you should be able to run the simulation: `python3 vogelsEtAl2011.py nest`

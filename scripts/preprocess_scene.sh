source ~/virtual_envs/scannetpp_env/scannetpp_env/bin/activate
cd scannetpp_toolkit
python -m dslr.downscale dslr/configs/downscale.yml
python -m dslr.undistort dslr/configs/undistort.yml
python -m common.render common/configs/render.yml
cd .. && cd data_parser
python parse_scannetpp.py --data_base /home/raviteja/code/datasets/scannetpp/data/0a5c013435 --output_path /home/raviteja/code/datasets/scannetpp/parsed
python parse_scannetpp.py --data_base /home/raviteja/code/datasets/scannetpp/data/08bbbdcc3d --output_path /home/raviteja/code/datasets/scannetpp/parsed

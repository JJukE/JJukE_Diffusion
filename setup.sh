python setup.py sdist bdist_wheel && \
python -m twine upload dist/* && \
rm -r ./build && rm -r ./dist && rm -r ./jjuke_diffusion.egg-info

build: force
	pip install --no-build-isolation -e . -v 2>&1 | tee build_log.txt

validate: force
	python -c "import test; test.validate()"

bench: force
	python -c "import test; test.benchmark_all()"

profile: force 
	ncu -f -o ncu_reports/$v --target-processes all --set detailed python -c "import test; test.profile('$v')"
	ncu-ui ./ncu_reports/$v.ncu-rep

force:


# adding a new kernel:
# add def to TORCH_LIBRARY in base_kernel.cu
# add source file to setup.py
# add implementation in operators.py
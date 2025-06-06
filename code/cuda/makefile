
build: force
	pip install --no-build-isolation -e . -v 2>&1 | tee build_log.txt

validate: force
	python -c "import test; test.validate()"

bench: force
	python -c "import test; test.benchmark_all()"

force:
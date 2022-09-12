test:
	pytest tests/

quality_checks:
	isort .
	black .
	pylint --recursive=y .

build: quality_checks test
	cd prediction_service
	docker build -t pred_com:v1 .
	docker run -it --rm -p 9696:9696 pred_com:v1
	python test_monitoring.py


publish: build integration_test
	echo "result"

setup: publish
	pipenv install --dev
	pre-commit install

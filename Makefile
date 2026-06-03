.PHONY: setup lint test docker-up docker-down clean

setup:
	uv sync --all-extras
	uv run pre-commit install

lint:
	uv run ruff check src/ tests/
	uv run ruff format --check src/ tests/
	uv run mypy src/

test:
	uv run pytest tests/ -v

docker-up:
	docker compose up -d

docker-down:
	docker compose down

clean:
	rmdir /s /q .venv
	rmdir /s /q .mypy_cache
	rmdir /s /q .pytest_cache
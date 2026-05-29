.ONESHELL:
SHELL := /bin/bash

VENV := .venv
PYTHON := uv run python

NAME := src
DIRS_TO_CLEAN := __pycache__ .mypy_cache
FILES_TO_CLEAN := *.pyc
EXCLUDE_NAMES_FLAKE8 := $(VENV),llm_sdk,data
EXCLUDE_NAMES_MYPY := (^|/)($(VENV)|llm_sdk|data)/

define ENSURE_VENV
	if [ ! -d "$(VENV)" ]; then \
		echo "No virtual environment detected."; \
		echo "Creating .venv"; \
		uv venv $(VENV); \
	fi
endef

help:
	@echo "Start:"
	echo "                 make install"
	echo "                 make run"
	echo
	echo "Available targets:"
	echo "  install        Install dependencies and set up the virtual environment"
	echo "  run            Run the application"
	echo "  lint           Run flake8 and mypy checks"
	echo "  lint-strict    Run flake8 and strict mypy checks"
	echo "  clean          Remove caches and temporary files"
	echo "  fclean         clean + remove virtual environment folder"
	echo "  debug          Run the debugger "

install:
	@$(ENSURE_VENV)
	@uv sync

run:
	@$(ENSURE_VENV)
	@$(PYTHON) -m $(NAME) \
		--functions_definition data/input/functions_definition.json \
		--input data/input/function_calling_tests_copy.json \
		--output data/output/function_calls.json

debug:
	@$(ENSURE_VENV)
	echo "Entering debug mode..."
	$(PYTHON) -m pdb -m $(NAME)

clean:
	@echo "Cleaning..."
	echo "Removing $(DIRS_TO_CLEAN) $(FILES_TO_CLEAN)"
	find . -mindepth 1 \( \
		-type d \( $(foreach d,$(DIRS_TO_CLEAN),-name "$(d)" -o ) -false \) -o \
		-type f \( $(foreach f,$(FILES_TO_CLEAN),-name "$(f)" -o ) -false \) \
	\) -exec rm -rf {} +
	echo "Cleaning is finished!"

fclean: clean
	@rm -fr $(VENV)
	echo "$(VENV) directory is removed"

lint:
	@$(ENSURE_VENV)
	echo "Checking with flake8..."
	uv run flake8 . --exclude=$(EXCLUDE_NAMES_FLAKE8)
	echo "flake8 check finished"
	echo "Checking with mypy..."
	uv run mypy . --exclude "$(EXCLUDE_NAMES_MYPY)" --warn-return-any \
		--warn-unused-ignores --ignore-missing-imports \
		--disallow-untyped-defs --check-untyped-defs

lint-strict:
	@$(ENSURE_VENV)
	echo "Checking with flake8..."
	uv run flake8 . --exclude=$(EXCLUDE_NAMES_FLAKE8)
	echo "flake8 check finished"
	echo "Checking with mypy --strict..."
	uv run mypy . --exclude "$(EXCLUDE_NAMES_MYPY)" --strict

.PHONY: help install run debug clean fclean lint lint-strict

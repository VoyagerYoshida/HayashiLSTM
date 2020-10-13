IMAGE_NAME := sample/hayashi
WORKINGDIR := /var/www
PWD := $(shell pwd)

.PHONY: build
build:
	@docker build . -t $(IMAGE_NAME)

.PHONY: run
run:
	@docker run --gpus all -it -u root -v $(PWD):$(WORKINGDIR) $(IMAGE_NAME) bash
